"""
End-to-End Mock Server Integration Test for Ox SDK
===================================================
Spins up a lightweight local mock CVM HTTPS server over loopback and tests:
  - Fresh nonce attestation handshake
  - Header-first Ed25519 canonical request signature verification
  - Protobuf EncryptedInferenceRequest unwrapping
  - Real-time length-prefixed StreamingInferenceFrame encryption & decryption
"""

import ssl
import json
import time
import struct
import socket
import secrets
import hashlib
import base64
import unittest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cevell_tunnel import CVMClient
from cevell_tunnel.wire import (
    decode_protobuf,
    encode_varint,
    encode_field_varint,
    encode_field_bytes,
    FRAME_TYPE_TOKEN_DELTA,
    FRAME_TYPE_STREAM_END,
)
from cevell_tunnel.crypto import compute_frame_nonce, compute_frame_aad


def generate_self_signed_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(x509.NameOID.COMMON_NAME, "127.0.0.1"),
    ])
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    import ipaddress
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


class MockCVMHandler(BaseHTTPRequestHandler):
    server_x25519_priv: x25519.X25519PrivateKey
    server_x25519_pub: bytes
    authorized_ed25519_pub: ed25519.Ed25519PublicKey

    def do_GET(self):
        if self.path.startswith("/v1/attestation"):
            query_nonce = ""
            if "nonce=" in self.path:
                query_nonce = self.path.split("nonce=")[1].split("&")[0]

            tls_fp = getattr(self, "server_tls_fp", b"\xaa" * 32)
            hpke_pub = self.server_x25519_pub
            if query_nonce:
                nonce_bytes = bytes.fromhex(query_nonce) if len(query_nonce) == 64 else hashlib.sha256(query_nonce.encode()).digest()
                second_half = hashlib.sha256(hpke_pub + nonce_bytes + (b"\x00" * 32)).digest()
            else:
                second_half = hashlib.sha256(hpke_pub).digest()
            user_data = tls_fp + second_half

            doc = {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": "https://in-toto.io/attestation/confidential-computing/v0.1",
                "subject": [],
                "predicate": {
                    "platform": "mock-enclave",
                    "tls_fingerprint": tls_fp.hex(),
                    "hpke_public_key": hpke_pub.hex(),
                    "user_data": user_data.hex(),
                    "nonce": getattr(MockCVMHandler, "override_nonce", None) or query_nonce,
                }
            }
            body = json.dumps(doc).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            # 1. Verify Ed25519 signature
            sig_b64 = self.headers.get("x-signature", "")
            ts = self.headers.get("x-timestamp", "")
            nonce = self.headers.get("x-nonce", "")
            req_id = self.headers.get("x-request-id", "")
            claimed_hash = self.headers.get("x-cevell-body-sha256", "")

            actual_hash = hashlib.sha256(body_bytes).hexdigest()
            if claimed_hash != actual_hash:
                self.send_response(401)
                self.end_headers()
                return

            canonical = f"POST:/v1/chat/completions:{req_id}:{ts}:{nonce}:{actual_hash}"
            try:
                self.authorized_ed25519_pub.verify(base64.b64decode(sig_b64), canonical.encode())
            except Exception:
                self.send_response(401)
                self.end_headers()
                return

            # 2. Decode Protobuf & Decrypt HPKE
            fields = decode_protobuf(body_bytes)
            client_pub_bytes = fields[3]
            req_nonce = fields[4]
            ciphertext = fields[5]

            shared_secret = self.server_x25519_priv.exchange(x25519.X25519PublicKey.from_public_bytes(client_pub_bytes))
            req_key = HKDF(hashes.SHA256(), 32, None, b"cevell-hpke-req-aes-gcm").derive(shared_secret)
            resp_key = HKDF(hashes.SHA256(), 32, None, b"cevell-hpke-resp-aes-gcm").derive(shared_secret)
            resp_iv = HKDF(hashes.SHA256(), 12, None, b"cevell-hpke-resp-base-iv").derive(shared_secret)

            plaintext = AESGCM(req_key).decrypt(req_nonce, ciphertext, None)
            req_json = json.loads(plaintext.decode("utf-8"))
            is_stream = req_json.get("stream", False)

            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

            def make_frame(seq, frame_type, payload_bytes):
                ts_ns = int(time.time() * 1e9)
                nonce = compute_frame_nonce(resp_iv, seq)
                aad = compute_frame_aad(seq, frame_type, ts_ns)
                sealed = AESGCM(resp_key).encrypt(nonce, payload_bytes, aad)
                ct = sealed[:-16]
                tag = sealed[-16:]

                proto = b"".join([
                    encode_field_varint(1, seq),
                    encode_field_varint(2, frame_type),
                    encode_field_bytes(3, ct),
                    encode_field_bytes(4, tag),
                    encode_field_varint(5, ts_ns),
                ])
                return struct.pack(">I", len(proto)) + proto

            if is_stream:
                f0 = make_frame(0, FRAME_TYPE_TOKEN_DELTA, b'data: {"choices":[{"delta":{"content":"Pong"}}]}\n\n')
                if getattr(MockCVMHandler, "truncate_stream", False):
                    self.wfile.write(f0)
                    return
                f1 = make_frame(1, FRAME_TYPE_STREAM_END, b"data: [DONE]\n\n")
                self.wfile.write(f0 + f1)
            else:
                resp_payload = json.dumps({"choices": [{"message": {"content": "Pong"}}]}).encode("utf-8")
                f0 = make_frame(0, 2, resp_payload)
                self.wfile.write(f0)
        else:
            self.send_response(404)
            self.end_headers()


class TestE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cert_pem, key_pem = generate_self_signed_cert()
        import tempfile
        cls.cert_file = tempfile.NamedTemporaryFile(delete=False)
        cls.cert_file.write(cert_pem)
        cls.cert_file.flush()

        cls.key_file = tempfile.NamedTemporaryFile(delete=False)
        cls.key_file.write(key_pem)
        cls.key_file.flush()

        # Generate keys
        cls.server_priv = x25519.X25519PrivateKey.generate()
        cls.server_pub = cls.server_priv.public_key().public_bytes_raw()

        cls.client_ed_priv = ed25519.Ed25519PrivateKey.generate()
        cls.client_ed_pub = cls.client_ed_priv.public_key()

        cls.client_key_file = tempfile.NamedTemporaryFile(delete=False)
        cls.client_key_file.write(
            cls.client_ed_priv.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        cls.client_key_file.flush()

        # Configure handler
        MockCVMHandler.server_x25519_priv = cls.server_priv
        MockCVMHandler.server_x25519_pub = cls.server_pub
        MockCVMHandler.authorized_ed25519_pub = cls.client_ed_pub
        cert_obj = x509.load_pem_x509_certificate(cert_pem)
        spki_bytes = cert_obj.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        cls.real_tls_fp = hashlib.sha256(spki_bytes).digest()
        MockCVMHandler.server_tls_fp = cls.real_tls_fp

        cls.server = HTTPServer(("127.0.0.1", 0), MockCVMHandler)
        cls.port = cls.server.server_port

        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certfile=cls.cert_file.name, keyfile=cls.key_file.name)
        cls.server.socket = ctx.wrap_socket(cls.server.socket, server_side=True)

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for f in [cls.cert_file, cls.key_file, cls.client_key_file]:
            try:
                os.remove(f.name)
            except Exception:
                pass

    def test_e2e_client_ask_and_stream(self):
        MockCVMHandler.server_tls_fp = self.real_tls_fp
        client = CVMClient(
            ip="127.0.0.1",
            port=self.port,
            auth_key=self.client_key_file.name,
            verify_attestation=True,
            enforce_official_roots=False,
            check_online_vendor=False,
            allow_mock_attestation=True,
        )

        self.assertIsNotNone(client.attestation)
        self.assertEqual(client.attestation.hpke_public_key, self.server_pub)

        # 1. Test ask()
        reply = client.ask("Ping")
        self.assertEqual(reply, "Pong")

        # 2. Test stream()
        chunks = list(client.stream("Ping"))
        self.assertEqual("".join(chunks), "Pong")

        # 3. Test client.chat.completions.create(stream=False)
        comp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Ping"}],
            stream=False,
        )
        self.assertEqual(comp["choices"][0]["message"]["content"], "Pong")

    def test_e2e_fail_closed_without_mock_flag(self):
        from cevell_tunnel.attestation import AttestationVerificationError
        # Default allow_mock_attestation=False MUST reject mock-enclave without silicon quotes
        MockCVMHandler.server_tls_fp = self.real_tls_fp
        with self.assertRaises(AttestationVerificationError) as ctx:
            CVMClient(
                ip="127.0.0.1",
                port=self.port,
                auth_key=self.client_key_file.name,
                verify_attestation=True,
                enforce_official_roots=False,
                check_online_vendor=False,
                # allow_mock_attestation defaults to False
            )
        self.assertIn("missing required silicon hardware quote", str(ctx.exception))

    def test_e2e_stream_truncation_detection(self):
        from cevell_tunnel.client import CVMInferenceError
        MockCVMHandler.server_tls_fp = self.real_tls_fp
        client = CVMClient(
            ip="127.0.0.1",
            port=self.port,
            auth_key=self.client_key_file.name,
            verify_attestation=True,
            enforce_official_roots=False,
            check_online_vendor=False,
            allow_mock_attestation=True,
        )

        MockCVMHandler.truncate_stream = True
        try:
            with self.assertRaises(CVMInferenceError) as ctx:
                list(client.stream("Ping"))
            self.assertIn("Stream truncated prematurely", str(ctx.exception))
        finally:
            MockCVMHandler.truncate_stream = False

    def test_e2e_tls_channel_binding_mitm_rejection(self):
        from cevell_tunnel.attestation import AttestationVerificationError
        # Simulate a MITM adversary relaying an attestation with a mismatched TLS fingerprint
        MockCVMHandler.server_tls_fp = b"\x99" * 32
        try:
            with self.assertRaises(AttestationVerificationError):
                CVMClient(
                    ip="127.0.0.1",
                    port=self.port,
                    auth_key=self.client_key_file.name,
                    verify_attestation=True,
                    enforce_official_roots=False,
                    check_online_vendor=False,
                    allow_mock_attestation=True,
                )
        finally:
            MockCVMHandler.server_tls_fp = self.real_tls_fp

    def test_e2e_nonce_replay_rejection(self):
        from cevell_tunnel.attestation import AttestationVerificationError
        # Simulate an adversary relaying an authentic attestation document from a previous session (stale nonce)
        MockCVMHandler.override_nonce = "deadbeef" * 8
        try:
            with self.assertRaises(AttestationVerificationError) as ctx:
                CVMClient(
                    ip="127.0.0.1",
                    port=self.port,
                    auth_key=self.client_key_file.name,
                    verify_attestation=True,
                    enforce_official_roots=False,
                    check_online_vendor=False,
                    allow_mock_attestation=True,
                )
            self.assertIn("Nonce mismatch", str(ctx.exception))
        finally:
            MockCVMHandler.override_nonce = None


if __name__ == "__main__":
    unittest.main()
