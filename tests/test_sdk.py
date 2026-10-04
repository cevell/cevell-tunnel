"""
Unit & Integration Tests for Ox SDK
====================================
Tests Protobuf wire encoding, Ed25519 canonical signing, RFC 9180 HPKE,
real Intel TDX QE3/PCK quote signatures, NVIDIA DICE leaf SPDM signatures,
and live online vendor CRL verification against Intel and NVIDIA trust services.
"""

import os
import json
import base64
import hashlib
import unittest
import struct
from typing import Optional

from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cevell_tunnel.wire import (
    encode_encrypted_inference_request,
    decode_streaming_frame,
    decode_protobuf,
    decode_varint,
    MAX_FRAME_SIZE,
    FRAME_TYPE_TOKEN_DELTA,
    FRAME_TYPE_STREAM_END,
)
from cevell_tunnel.crypto import (
    load_auth_key,
    sign_canonical_request,
    create_client_hpke_session,
    zeroize_buffer,
    LOW_ORDER_X25519_BYTES,
    HPKEError,
    AuthKeyError,
)
from cevell_tunnel.attestation import (
    verify_attestation_document,
    verify_intel_tdx_quote,
    verify_nvidia_gpu_evidence,
    VendorCRLManager,
    ReleaseMeasurementManager,
    AttestationVerificationError,
    OFFICIAL_NVIDIA_ROOT_CA_PEM,
    OFFICIAL_INTEL_ROOT_CA_PEM,
)
from cevell_tunnel.client import CVMClient, CVMInferenceError


def get_attestation_path() -> Optional[str]:
    candidate = os.environ.get("OX_TEST_ATTESTATION_PATH")
    if candidate and os.path.exists(candidate):
        return candidate
    local_fixture = os.path.join(os.path.dirname(__file__), "fixtures", "attestation.json")
    if os.path.exists(local_fixture):
        return local_fixture
    legacy = "/home/trave/cevell/attestation.json"
    if os.path.exists(legacy):
        return legacy
    return None


class TestOxSDK(unittest.TestCase):

    def test_wire_protobuf_encoding_and_decoding(self):
        client_pub = b"\x11" * 32
        nonce = b"\x22" * 12
        payload = b"encrypted-data-payload"
        binding = b"\x33" * 32
        req_id = "req-test-999"

        proto_bytes = encode_encrypted_inference_request(
            version=1,
            cipher_suite=1,
            client_ephemeral_public_key=client_pub,
            nonce=nonce,
            encrypted_payload=payload,
            attestation_binding_hash=binding,
            request_id=req_id,
        )

        fields = decode_protobuf(proto_bytes)
        self.assertEqual(fields[1], 1)  # version
        self.assertEqual(fields[2], 1)  # cipher_suite
        self.assertEqual(fields[3], client_pub)
        self.assertEqual(fields[4], nonce)
        self.assertEqual(fields[5], payload)
        self.assertEqual(fields[6], binding)
        self.assertEqual(fields[7].decode("utf-8"), req_id)

    def test_wire_security_bounds(self):
        # 1. Test varint shift overflow rejection (DoS prevention)
        malicious_varint = b"\x80" * 10 + b"\x01"
        with self.assertRaises(ValueError):
            decode_varint(malicious_varint, 0)

        # 2. Test truncated length-delimited field
        bad_length_field = b"\x12\x50" + b"\x00" * 10  # field 2 (wire 2), len 80, but only 10 bytes
        with self.assertRaises(ValueError):
            decode_protobuf(bad_length_field)

        # 3. Test frame size ceiling in decode_streaming_frame
        huge_frame = struct.pack(">I", MAX_FRAME_SIZE + 1) + b"\x00" * 10
        with self.assertRaises(ValueError):
            decode_streaming_frame(huge_frame)

    def test_crypto_ed25519_key_loading_and_signing(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pem_bytes = priv_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        loaded = load_auth_key(pem_bytes)
        self.assertIsNotNone(loaded)

        body = b'{"test": 123}'
        headers = sign_canonical_request(
            priv_key=loaded,
            method="POST",
            path="/v1/chat/completions",
            body_bytes=body,
            request_id="req-test-1",
            timestamp="1789000000",
            nonce="nonce-123",
        )

        self.assertIn("Authorization", headers)
        self.assertIn("x-signature", headers)
        self.assertIn("x-cevell-body-sha256", headers)
        self.assertEqual(headers["x-cevell-body-sha256"], hashlib.sha256(body).hexdigest())

        canonical = f"POST:/v1/chat/completions:req-test-1:1789000000:nonce-123:{hashlib.sha256(body).hexdigest()}"
        sig_bytes = base64.b64decode(headers["x-signature"])
        loaded.public_key().verify(sig_bytes, canonical.encode("utf-8"))

    def test_crypto_ed25519_key_loading_broad_formats(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        raw_seed = priv_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        hex_seed = raw_seed.hex()
        expected_pub = priv_key.public_key().public_bytes_raw()

        # 1. Load from 64-char hex string
        k1 = load_auth_key(hex_seed)
        self.assertEqual(k1.public_key().public_bytes_raw(), expected_pub)

        # 2. Load from 64-char hex string in a file (with leading/trailing whitespace & newlines)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(f"  {hex_seed} \n")
            hex_file_path = f.name
        try:
            k2 = load_auth_key(hex_file_path)
            self.assertEqual(k2.public_key().public_bytes_raw(), expected_pub)
        finally:
            if os.path.exists(hex_file_path):
                os.remove(hex_file_path)

        # 3. Load from raw 32-byte binary seed (bytes)
        k3 = load_auth_key(raw_seed)
        self.assertEqual(k3.public_key().public_bytes_raw(), expected_pub)

        # 4. Load from raw 32-byte binary file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(raw_seed)
            bin_file_path = f.name
        try:
            k4 = load_auth_key(bin_file_path)
            self.assertEqual(k4.public_key().public_bytes_raw(), expected_pub)
        finally:
            if os.path.exists(bin_file_path):
                os.remove(bin_file_path)

        # 5. Load from 64-byte keypair (seed + pubkey, libsodium format)
        keypair_bytes = raw_seed + expected_pub
        k5 = load_auth_key(keypair_bytes)
        self.assertEqual(k5.public_key().public_bytes_raw(), expected_pub)

        # 6. Load from 128-char hex string (libsodium keypair hex)
        k6 = load_auth_key(keypair_bytes.hex())
        self.assertEqual(k6.public_key().public_bytes_raw(), expected_pub)

        # 7. Load from base64 encoded seed
        b64_seed = base64.b64encode(raw_seed).decode("ascii")
        k7 = load_auth_key(b64_seed)
        self.assertEqual(k7.public_key().public_bytes_raw(), expected_pub)

        # 8. Invalid key input raises AuthKeyError
        with self.assertRaises(AuthKeyError):
            load_auth_key("invalid-key-data")

    def test_crypto_hpke_encryption_and_decryption(self):
        server_priv = x25519.X25519PrivateKey.generate()
        server_pub_bytes = server_priv.public_key().public_bytes_raw()

        client_session, client_pub_bytes = create_client_hpke_session(server_pub_bytes)
        self.assertEqual(len(client_pub_bytes), 32)

        plaintext = b"Confidential AI Prompt"
        nonce, ciphertext = client_session.encrypt_request(plaintext)

        self.assertEqual(len(nonce), 12)
        self.assertTrue(len(ciphertext) > len(plaintext))

        # Server derives keys and decrypts request
        shared_secret = server_priv.exchange(x25519.X25519PublicKey.from_public_bytes(client_pub_bytes))
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cevell_tunnel.crypto import compute_frame_nonce, compute_frame_aad

        req_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"cevell-hpke-req-aes-gcm").derive(shared_secret)
        resp_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce, info=b"cevell-hpke-resp-aes-gcm").derive(shared_secret)
        resp_iv = HKDF(algorithm=hashes.SHA256(), length=12, salt=nonce, info=b"cevell-hpke-resp-base-iv").derive(shared_secret)

        decrypted_req = AESGCM(req_key).decrypt(nonce, ciphertext, None)
        self.assertEqual(decrypted_req, plaintext)

        # Server emits encrypted frame seq=0
        seq = 0
        frame_type = FRAME_TYPE_TOKEN_DELTA
        ts_ns = 1789000000000
        frame_nonce = compute_frame_nonce(resp_iv, seq)
        frame_aad = compute_frame_aad(seq, frame_type, ts_ns)
        token_plain = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        frame_ciphertext_with_tag = AESGCM(resp_key).encrypt(frame_nonce, token_plain, frame_aad)
        ct_only = frame_ciphertext_with_tag[:-16]
        tag_only = frame_ciphertext_with_tag[-16:]

        # Client decrypts frame seq=0
        decrypted_token = client_session.decrypt_frame(seq, frame_type, ct_only, tag_only, ts_ns)
        self.assertEqual(decrypted_token, token_plain)

        # Monotonic sequence enforcement: Replay of seq=0 must fail (expected 1)
        with self.assertRaises(HPKEError) as ctx:
            client_session.decrypt_frame(0, frame_type, ct_only, tag_only, ts_ns)
        self.assertIn("Out-of-order frame sequence", str(ctx.exception))

        # Monotonic sequence enforcement: Out-of-order seq=2 must fail (expected 1)
        with self.assertRaises(HPKEError) as ctx:
            client_session.decrypt_frame(2, frame_type, ct_only, tag_only, ts_ns)
        self.assertIn("Out-of-order frame sequence", str(ctx.exception))

        # Client decrypts frame seq=1
        seq_1 = 1
        frame_nonce_1 = compute_frame_nonce(resp_iv, seq_1)
        frame_aad_1 = compute_frame_aad(seq_1, FRAME_TYPE_STREAM_END, ts_ns + 1)
        end_plain = b"data: [DONE]\n\n"
        end_sealed = AESGCM(resp_key).encrypt(frame_nonce_1, end_plain, frame_aad_1)
        decrypted_end = client_session.decrypt_frame(seq_1, FRAME_TYPE_STREAM_END, end_sealed[:-16], end_sealed[-16:], ts_ns + 1)
        self.assertEqual(decrypted_end, end_plain)

        # Test real zeroization in HPKESession
        client_session.zeroize()
        self.assertEqual(client_session._req_key, bytearray(b"\x00" * 32))
        self.assertEqual(client_session._resp_key, bytearray(b"\x00" * 32))
        self.assertEqual(client_session._resp_base_iv, bytearray(b"\x00" * 12))
        self.assertIsNone(client_session.req_nonce)
        self.assertEqual(client_session.expected_seq, 0)
        with self.assertRaises(HPKEError):
            client_session.encrypt_request(b"test")
        with self.assertRaises(HPKEError):
            client_session.decrypt_frame(0, 1, b"ct", b"tag", 0)

    def test_crypto_hpke_nonce_binding_derives_unique_keys(self):
        server_priv = x25519.X25519PrivateKey.generate()
        server_pub_bytes = server_priv.public_key().public_bytes_raw()

        nonce1 = b"\x01" * 12
        nonce2 = b"\x02" * 12

        session1, _ = create_client_hpke_session(server_pub_bytes, nonce=nonce1)
        session2, _ = create_client_hpke_session(server_pub_bytes, nonce=nonce2)

        self.assertNotEqual(session1.resp_key, session2.resp_key)
        self.assertNotEqual(session1.resp_base_iv, session2.resp_base_iv)
        self.assertEqual(bytes(session1.req_nonce), nonce1)
        self.assertEqual(bytes(session2.req_nonce), nonce2)

        # Reusing the exact same client private key with two different nonces
        # guarantees distinct response encryption keys and IVs
        client_priv = x25519.X25519PrivateKey.generate()
        shared = client_priv.exchange(server_priv.public_key())
        k1 = HKDF(hashes.SHA256(), 32, salt=nonce1, info=b"cevell-hpke-resp-aes-gcm").derive(shared)
        k2 = HKDF(hashes.SHA256(), 32, salt=nonce2, info=b"cevell-hpke-resp-aes-gcm").derive(shared)
        self.assertNotEqual(k1, k2)

        # Test all RFC 7748 Table 6 low-order points rejection
        for low_order_pt in LOW_ORDER_X25519_BYTES:
            with self.assertRaises(HPKEError):
                create_client_hpke_session(low_order_pt)

    def test_real_attestation_full_cryptographic_verification(self):
        attestation_path = get_attestation_path()
        if not attestation_path or not os.path.exists(attestation_path):
            self.skipTest("attestation.json fixture not found")

        with open(attestation_path, "r") as f:
            doc = json.load(f)

        # Full verification including Intel QE3 ECDSA signature, NVIDIA SPDM P-384 signature,
        # and live vendor CRL validation against Intel and NVIDIA online endpoints!
        res = verify_attestation_document(
            doc=doc,
            expected_nonce_hex=doc["predicate"]["nonce"],
            expected_tls_fingerprint=doc["predicate"]["tls_fingerprint"],
            enforce_official_roots=True,
            check_online_vendor=True,
        )

        self.assertEqual(res.platform, "intel-tdx")
        self.assertTrue(res.has_gpu)
        self.assertEqual(res.gpu_model, "NVIDIA H100 80GB HBM3")
        self.assertEqual(res.hpke_public_key_hex, doc["predicate"]["hpke_public_key"])
        self.assertEqual(res.tls_fingerprint, doc["predicate"]["tls_fingerprint"])
        self.assertTrue(res.vendor_crl_verified)
        self.assertIsNotNone(res.mrtd)
        self.assertIsNotNone(res.td_attributes)

    def test_code_measurement_verification_and_24h_caching(self):
        attestation_path = get_attestation_path()
        if not attestation_path or not os.path.exists(attestation_path):
            self.skipTest("attestation.json fixture not found")

        with open(attestation_path, "r") as f:
            doc = json.load(f)

        # 1. Verification succeeds dynamically against official latest release
        res = verify_attestation_document(
            doc=doc,
            expected_nonce_hex=doc["predicate"]["nonce"],
            expected_tls_fingerprint=doc["predicate"]["tls_fingerprint"],
            enforce_official_roots=True,
            check_online_vendor=False,
            verify_code=True,
            expected_release="latest",
        )
        self.assertTrue(res.code_verified)
        self.assertIn("v1", res.code_release)
        self.assertIsNotNone(res.code_roothash)

        # 2. Mismatched RTMR1 measurement raises AttestationVerificationError
        with self.assertRaises(AttestationVerificationError) as ctx:
            verify_attestation_document(
                doc=doc,
                expected_nonce_hex=doc["predicate"]["nonce"],
                expected_tls_fingerprint=doc["predicate"]["tls_fingerprint"],
                enforce_official_roots=True,
                check_online_vendor=False,
                verify_code=True,
                expected_rtmr1="00" * 48,
            )
        self.assertIn("mismatch", str(ctx.exception).lower())

        # 3. Test ReleaseMeasurementManager disk caching with 24h TTL and auto-update refresh
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ReleaseMeasurementManager(cache_ttl=86400.0, cache_dir=tmp_dir)
            cached_data = mgr.fetch_measurements("latest")
            self.assertIn("version", cached_data)
            cache_file = os.path.join(tmp_dir, "latest.json")
            self.assertTrue(os.path.exists(cache_file))

            # Modify file content to prove cache is read without network hit
            with open(cache_file, "r") as cf:
                loaded = json.load(cf)
            loaded["custom_marker"] = "cached_disk_test"
            with open(cache_file, "w") as cf:
                json.dump(loaded, cf)

            # Re-instantiate manager with same cache dir
            mgr2 = ReleaseMeasurementManager(cache_ttl=86400.0, cache_dir=tmp_dir)
            re_read = mgr2.fetch_measurements("latest")
            self.assertEqual(re_read.get("custom_marker"), "cached_disk_test")

            # 4. Auto-update resilience test:
            # Simulate a stale cache with an outdated RTMR1
            loaded["measurements"]["intel_tdx"]["rtmr1"] = "11" * 48
            with open(cache_file, "w") as cf:
                json.dump(loaded, cf)

            # Verify that when verify_code_measurements runs, the mismatch triggers
            # a fresh fetch from GitHub and recovers successfully without error!
            mgr_recovery = ReleaseMeasurementManager(cache_ttl=86400.0, cache_dir=tmp_dir)
            recovered = mgr_recovery.verify_code_measurements(res, expected_release="latest")
            self.assertTrue(recovered["verified"])

    def test_intel_tdx_quote_cryptographic_verification(self):
        attestation_path = get_attestation_path()
        if not attestation_path or not os.path.exists(attestation_path):
            self.skipTest("attestation.json fixture not found")

        with open(attestation_path, "r") as f:
            doc = json.load(f)

        raw_quote = base64.b64decode(doc["predicate"]["raw_quote"])
        user_data = bytes.fromhex(doc["predicate"]["user_data"])

        # 1. Genuine quote verification succeeds
        hw_info = verify_intel_tdx_quote(
            quote_bytes=raw_quote,
            user_data=user_data,
            enforce_official_roots=True,
            check_online_vendor=True,
        )
        self.assertIn("mrtd", hw_info)

        # 2. Tampered quote body fails QE3 ECDSA signature verification
        tampered_quote = bytearray(raw_quote)
        tampered_quote[100] ^= 0xFF
        with self.assertRaises(AttestationVerificationError):
            verify_intel_tdx_quote(
                quote_bytes=bytes(tampered_quote),
                user_data=user_data,
                enforce_official_roots=False,
                check_online_vendor=False,
            )

        # 3. Debug-enabled TDX enclave is rejected (td_report offset 120 = quote offset 168)
        debug_quote = bytearray(raw_quote)
        debug_quote[168] |= 0x01  # set bit 0 (DEBUG) in td_attributes
        with self.assertRaises(AttestationVerificationError):
            verify_intel_tdx_quote(
                quote_bytes=bytes(debug_quote),
                user_data=user_data,
                enforce_official_roots=False,
                check_online_vendor=False,
            )

    def test_nvidia_spdm_evidence_cryptographic_verification(self):
        attestation_path = get_attestation_path()
        if not attestation_path or not os.path.exists(attestation_path):
            self.skipTest("attestation.json fixture not found")

        with open(attestation_path, "r") as f:
            doc = json.load(f)

        gpu_ev = doc["predicate"]["gpu_evidence"]
        nonce_bytes = bytes.fromhex(doc["predicate"]["nonce"])

        # 1. Genuine SPDM report verifies signature against leaf DICE cert
        info = verify_nvidia_gpu_evidence(
            gpu_ev=gpu_ev,
            nonce_bytes=nonce_bytes,
            enforce_official_roots=True,
            check_online_vendor=True,
        )
        self.assertEqual(info["model"], "NVIDIA H100 80GB HBM3")

        # 2. Tampered SPDM evidence report fails ECDSA P-384 signature
        ev_raw = bytearray(base64.b64decode(gpu_ev["evidence_report"]))
        ev_raw[50] ^= 0xFF
        tampered_gpu_ev = dict(gpu_ev)
        tampered_gpu_ev["evidence_report"] = base64.b64encode(ev_raw).decode()
        with self.assertRaises(AttestationVerificationError):
            verify_nvidia_gpu_evidence(
                gpu_ev=tampered_gpu_ev,
                nonce_bytes=nonce_bytes,
                enforce_official_roots=False,
                check_online_vendor=False,
            )

        # 3. Nonce mismatch in SPDM measurement header is rejected
        with self.assertRaises(AttestationVerificationError):
            verify_nvidia_gpu_evidence(
                gpu_ev=gpu_ev,
                nonce_bytes=b"\xFF" * 32,
                enforce_official_roots=False,
                check_online_vendor=False,
            )

    def test_attestation_tampering_rejected(self):
        attestation_path = get_attestation_path()
        if not attestation_path or not os.path.exists(attestation_path):
            self.skipTest("attestation.json fixture not found")

        with open(attestation_path, "r") as f:
            doc = json.load(f)

        # 1. Tamper with HPKE public key
        tampered_doc = json.loads(json.dumps(doc))
        tampered_doc["predicate"]["hpke_public_key"] = "00" * 32
        with self.assertRaises(AttestationVerificationError):
            verify_attestation_document(tampered_doc, enforce_official_roots=False, check_online_vendor=False)

        # 2. Tamper with Nonce
        tampered_doc = json.loads(json.dumps(doc))
        with self.assertRaises(AttestationVerificationError):
            verify_attestation_document(tampered_doc, expected_nonce_hex="ff" * 32, check_online_vendor=False)

        # 3. Tamper with UserData
        tampered_doc = json.loads(json.dumps(doc))
        tampered_doc["predicate"]["user_data"] = "ff" * 64
        with self.assertRaises(AttestationVerificationError):
            verify_attestation_document(tampered_doc, enforce_official_roots=False, check_online_vendor=False)

        # 4. Tamper with TLS fingerprint (Channel Binding)
        with self.assertRaises(AttestationVerificationError):
            verify_attestation_document(
                doc=doc,
                expected_tls_fingerprint="00" * 32,
                check_online_vendor=False,
            )

    def test_client_transport_security(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pem = priv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        # Plain HTTP without allow_insecure_http must be rejected
        with self.assertRaises(ValueError) as ctx:
            CVMClient(cvm_url="http://127.0.0.1:8080", auth_key=pem, verify_attestation=False)
        self.assertIn("Insecure HTTP connections are disabled", str(ctx.exception))

        # Explicit allow_insecure_http=True permits HTTP
        c = CVMClient(
            cvm_url="http://127.0.0.1:8080",
            auth_key=pem,
            verify_attestation=False,
            allow_insecure_http=True,
        )
        self.assertEqual(c.base_url, "http://127.0.0.1:8080")

    def test_client_frame_pre_allocation_dos_guard(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pem = priv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        client = CVMClient(cvm_url="https://127.0.0.1:8080", auth_key=pem, verify_attestation=False)

        class MockResp:
            def __init__(self, data: bytes):
                self.data = data
                self.pos = 0
                self.status = 200
                self.read_sizes = []

            def read(self, n=None):
                if n is not None:
                    self.read_sizes.append(n)
                    chunk = self.data[self.pos:self.pos + n]
                    self.pos += len(chunk)
                    return chunk
                chunk = self.data[self.pos:]
                self.pos = len(self.data)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockOpener:
            def __init__(self, resp):
                self.resp = resp

            def open(self, req, timeout=None):
                return self.resp

        # 1. Huge streaming frame prefix (e.g. 100 MB)
        oversized_len = MAX_FRAME_SIZE + 1024
        mock_resp = MockResp(struct.pack(">I", oversized_len) + b"\x00" * 16)
        client._opener = MockOpener(mock_resp)

        server_priv = x25519.X25519PrivateKey.generate()
        session, _ = create_client_hpke_session(server_priv.public_key().public_bytes_raw())

        from urllib.request import Request
        fake_req = Request("https://127.0.0.1:8080/v1/chat/completions")

        gen = client._execute_streaming_request(fake_req, session)
        with self.assertRaises(CVMInferenceError) as ctx:
            next(gen)
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))
        self.assertNotIn(oversized_len, mock_resp.read_sizes)

        # 2. Huge unary response frame prefix
        mock_resp_unary = MockResp(struct.pack(">I", oversized_len) + b"\x00" * 16)
        client._opener = MockOpener(mock_resp_unary)
        session2, _ = create_client_hpke_session(server_priv.public_key().public_bytes_raw())
        with self.assertRaises(CVMInferenceError) as ctx:
            client._execute_unary_request(fake_req, session2)
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))
        self.assertNotIn(oversized_len, mock_resp_unary.read_sizes)

    def test_client_nonce_replay_guard(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pem = priv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        client = CVMClient(
            cvm_url="https://127.0.0.1:8080",
            auth_key=pem,
            verify_attestation=False,
            allow_mock_attestation=True,
        )

        attestation_path = get_attestation_path()
        if attestation_path and os.path.exists(attestation_path):
            with open(attestation_path, "r") as f:
                doc = json.load(f)
        else:
            tls_fp = b"\x22" * 32
            hpke_pub = b"\x11" * 32
            stale_nonce = "00" * 32
            nonce_bytes = bytes.fromhex(stale_nonce)
            second_half = hashlib.sha256(hpke_pub + nonce_bytes + (b"\x00" * 32)).digest()
            user_data = tls_fp + second_half
            doc = {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": "https://in-toto.io/attestation/confidential-computing/v0.1",
                "subject": [],
                "predicate": {
                    "platform": "intel-tdx",
                    "tls_fingerprint": tls_fp.hex(),
                    "hpke_public_key": hpke_pub.hex(),
                    "user_data": user_data.hex(),
                    "nonce": stale_nonce,
                }
            }

        class MockAttestationResp:
            def __init__(self, doc_json):
                self.body = json.dumps(doc_json).encode("utf-8")
                self.status = 200

            def read(self, *args):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockAttestationOpener:
            def __init__(self, resp):
                self.resp = resp

            def open(self, req, timeout=None):
                return self.resp

        class MockHandler:
            last_conn = None

        mock_opener = MockAttestationOpener(MockAttestationResp(doc))
        client._create_attestation_opener = lambda: (mock_opener, MockHandler())
        fresh_nonce = "11" * 32
        with self.assertRaises(AttestationVerificationError) as ctx:
            client.audit_and_bind_attestation(nonce=fresh_nonce)
        self.assertIn("Nonce mismatch", str(ctx.exception))

    def test_verify_attestation_fail_closed_on_missing_raw_quote(self):
        hpke_pub = b"\x11" * 32
        tls_fp = b"\x22" * 32
        nonce_hex = "aa" * 32
        nonce_bytes = bytes.fromhex(nonce_hex)
        second_half = hashlib.sha256(hpke_pub + nonce_bytes + (b"\x00" * 32)).digest()
        user_data = tls_fp + second_half

        doc = {
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://in-toto.io/attestation/confidential-computing/v0.1",
            "subject": [],
            "predicate": {
                "platform": "intel-tdx",
                "tls_fingerprint": tls_fp.hex(),
                "hpke_public_key": hpke_pub.hex(),
                "user_data": user_data.hex(),
                "nonce": nonce_hex,
            }
        }

        # 1. By default (allow_mock_attestation=False), MUST raise AttestationVerificationError
        with self.assertRaises(AttestationVerificationError) as ctx:
            verify_attestation_document(doc, expected_nonce_hex=nonce_hex)
        self.assertIn("missing required silicon hardware quote", str(ctx.exception))

        # 2. When allow_mock_attestation=True, succeeds and extracts key
        res = verify_attestation_document(doc, expected_nonce_hex=nonce_hex, allow_mock_attestation=True)
        self.assertEqual(res.hpke_public_key, hpke_pub)

    def test_verify_attestation_rejects_unsupported_platform(self):
        hpke_pub = b"\x11" * 32
        tls_fp = b"\x22" * 32
        nonce_hex = "aa" * 32
        nonce_bytes = bytes.fromhex(nonce_hex)
        second_half = hashlib.sha256(hpke_pub + nonce_bytes + (b"\x00" * 32)).digest()
        user_data = tls_fp + second_half

        doc = {
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://in-toto.io/attestation/confidential-computing/v0.1",
            "subject": [],
            "predicate": {
                "platform": "rogue-hypervisor-platform",
                "raw_quote": base64.b64encode(b"\x00" * 636).decode("utf-8"),
                "tls_fingerprint": tls_fp.hex(),
                "hpke_public_key": hpke_pub.hex(),
                "user_data": user_data.hex(),
                "nonce": nonce_hex,
            }
        }

        with self.assertRaises(AttestationVerificationError) as ctx:
            verify_attestation_document(doc, expected_nonce_hex=nonce_hex)
        self.assertIn("Unsupported hardware platform", str(ctx.exception))

    def test_client_load_model_quantization_options(self):
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pem = priv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        client = CVMClient(cvm_url="https://127.0.0.1:8080", auth_key=pem, verify_attestation=False)

        captured_requests = []

        class MockResp:
            def __init__(self, body: bytes):
                self.body = body
            def read(self, n=None):
                return self.body
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockOpener:
            def open(self, req, timeout=None):
                captured_requests.append(req)
                return MockResp(b'{"status":"ok","model":"loaded"}')

        client._opener = MockOpener()

        # 1. Test BitsAndBytes flag
        res1 = client.load_model(model="unsloth/Llama-3.2-3B-Instruct", bitsandbytes=True)
        self.assertEqual(res1["status"], "ok")
        req1 = captured_requests[-1]
        self.assertEqual(req1.full_url, "https://127.0.0.1:8080/v1/models/load")
        payload1 = json.loads(req1.data.decode("utf-8"))
        self.assertEqual(payload1["model"], "unsloth/Llama-3.2-3B-Instruct")
        self.assertTrue(payload1["bitsandbytes"])
        self.assertNotIn("gguf", payload1)
        self.assertIn("X-signature", req1.headers)
        self.assertIn("Authorization", req1.headers)

        # 2. Test GGUF flag
        res2 = client.load_model(model="meta-llama/Llama-3.2-3B-Instruct-GGUF", gguf=True)
        self.assertEqual(res2["status"], "ok")
        req2 = captured_requests[-1]
        payload2 = json.loads(req2.data.decode("utf-8"))
        self.assertEqual(payload2["model"], "meta-llama/Llama-3.2-3B-Instruct-GGUF")
        self.assertTrue(payload2["gguf"])
        self.assertNotIn("bitsandbytes", payload2)

        # 3. Test explicit quantization override
        res3 = client.load_model(model="custom-model", quantization="bitsandbytes")
        self.assertEqual(res3["status"], "ok")
        req3 = captured_requests[-1]
        payload3 = json.loads(req3.data.decode("utf-8"))
        self.assertEqual(payload3["model"], "custom-model")
        self.assertEqual(payload3["quantization"], "bitsandbytes")


if __name__ == "__main__":
    unittest.main()
