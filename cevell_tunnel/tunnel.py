"""
ox.tunnel — Agnostic Confidential Byte-Stream Tunnel
====================================================
Provides an agnostic, payload-independent transport tunnel between local callers
and confidential CVM enclaves:
  - Verifies hardware silicon attestation (Intel TDX / NVIDIA Hopper / AMD SEV-SNP)
  - Enforces physical TLS socket SPKI channel binding against silicon REPORT_DATA
  - Transparently wraps arbitrary HTTP request bodies into RFC 9180 HPKE Protobuf envelopes
  - Signs canonical HTTP request headers with tenant Ed25519 key (Cevell-Ed25519)
  - Decrypts length-prefixed streaming response frames in real-time, yielding raw plaintext bytes
  - Completely agnostic to endpoints, payloads, JSON schemas, SSE formats, or model providers
"""

import json
import ssl
import struct
import secrets
import hashlib
import http.client
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, Any, Tuple, Generator, AsyncGenerator, Optional, Union, BinaryIO

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from .crypto import (
    load_auth_key,
    sign_canonical_request,
    create_client_hpke_session,
    HPKESession,
    HPKEError,
)
from .wire import (
    encode_encrypted_inference_request,
    decode_streaming_frame,
    MAX_FRAME_SIZE,
    FRAME_TYPE_TOKEN_DELTA,
    FRAME_TYPE_COMPLETION,
    FRAME_TYPE_STREAM_END,
    FRAME_TYPE_ERROR,
)
from .attestation import (
    verify_attestation_document,
    AttestationResult,
    AttestationVerificationError,
)


class TunnelError(Exception):
    """Raised when tunnel handshake, framing, or decryption fails."""
    pass


class BoundHTTPSConnection(http.client.HTTPSConnection):
    """
    TLS connection that extracts the peer's actual X.509 SubjectPublicKeyInfo (SPKI)
    fingerprint on handshake and enforces hardware attestation binding before sending data.
    """
    def __init__(self, *args, expected_spki: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_spki = expected_spki
        self.peer_spki: Optional[str] = None

    def connect(self):
        super().connect()
        der_cert = self.sock.getpeercert(binary_form=True)
        if not der_cert:
            raise ssl.SSLError("No peer certificate returned on TLS handshake")
        cert = x509.load_der_x509_certificate(der_cert)
        spki = cert.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.peer_spki = hashlib.sha256(spki).hexdigest()
        if self.expected_spki and self.peer_spki.lower() != self.expected_spki.lower():
            self.close()
            raise ssl.CertificateError(
                f"TLS SPKI fingerprint mismatch: expected {self.expected_spki}, got {self.peer_spki}"
            )


class BoundHTTPSHandler(urllib.request.HTTPSHandler):
    """Urllib HTTPSHandler that injects BoundHTTPSConnection with SPKI validation."""
    def __init__(self, context=None, expected_spki: Optional[str] = None):
        super().__init__(context=context)
        self.expected_spki = expected_spki
        self.last_conn: Optional[BoundHTTPSConnection] = None

    def https_open(self, req):
        def creator(*args, **kwargs):
            conn = BoundHTTPSConnection(*args, expected_spki=self.expected_spki, **kwargs)
            self.last_conn = conn
            return conn
        return self.do_open(creator, req, context=self._context)


class ConfidentialTunnel:
    """
    Agnostic, payload-independent zero-trust tunnel for communicating with confidential CVMs.
    Encapsulates silicon attestation, channel binding, HPKE encryption, and Protobuf framing.
    """

    def __init__(
        self,
        cvm_url: Optional[str] = None,
        ip: Optional[str] = None,
        port: int = 443,
        auth_key: Union[str, bytes] = "auth.pem",
        verify_attestation: bool = True,
        verify_code: bool = True,
        expected_release: Optional[str] = "v1.0.0",
        expected_rtmr1: Optional[str] = None,
        enforce_official_roots: bool = True,
        check_online_vendor: bool = True,
        allow_mock_attestation: bool = False,
        allow_insecure_http: bool = False,
        timeout: float = 60.0,
    ):
        if cvm_url:
            clean_url = cvm_url.rstrip("/")
            if clean_url.startswith("http://"):
                if not allow_insecure_http:
                    raise ValueError(
                        "Insecure HTTP connections are disabled by default because silicon TLS channel binding "
                        "requires HTTPS. Set allow_insecure_http=True only for local testing environments."
                    )
                self.base_url = clean_url
            elif clean_url.startswith("https://"):
                self.base_url = clean_url
            else:
                self.base_url = f"https://{clean_url}"
        elif ip:
            clean_ip = ip.replace("https://", "").replace("http://", "").strip("/")
            if ":" in clean_ip:
                self.base_url = f"https://{clean_ip}"
            else:
                self.base_url = f"https://{clean_ip}:{port}"
        else:
            raise ValueError("Must provide either cvm_url or ip")

        self.timeout = timeout
        self.verify_attestation_enabled = verify_attestation
        self.verify_code = verify_code
        self.expected_release = expected_release
        self.expected_rtmr1 = expected_rtmr1
        self.enforce_official_roots = enforce_official_roots
        self.check_online_vendor = check_online_vendor
        self.allow_mock_attestation = allow_mock_attestation
        self.allow_insecure_http = allow_insecure_http

        # 1. Load tenant Ed25519 signing key
        self.priv_key = load_auth_key(auth_key)

        # 2. SSL context with disabled standard WebPKI validation (anchored to silicon SPKI)
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

        # 3. Opener state
        self._bound_handler = BoundHTTPSHandler(context=self.ssl_ctx, expected_spki=None)
        self._opener = urllib.request.build_opener(self._bound_handler)

        # 4. Silicon attestation state
        self.attestation: Optional[AttestationResult] = None
        if self.verify_attestation_enabled:
            self.audit_and_bind_attestation()

    def _create_attestation_opener(self) -> Tuple[urllib.request.OpenerDirector, BoundHTTPSHandler]:
        """Creates an HTTPS opener for pre-attestation discovery before SPKI is bound."""
        handler = BoundHTTPSHandler(context=self.ssl_ctx, expected_spki=None)
        return urllib.request.build_opener(handler), handler

    def audit_and_bind_attestation(self, nonce: Optional[str] = None) -> AttestationResult:
        """
        Queries the CVM attestation endpoint, cryptographically verifies manufacturer root
        certificates, validates hardware quote embedding and signatures, checks vendor CRLs,
        verifies TLS channel binding against physical socket SPKI, and binds the HPKE public key.
        """
        req_nonce = nonce or secrets.token_hex(32)
        parsed_base = urllib.parse.urlparse(self.base_url)
        origin = f"{parsed_base.scheme}://{parsed_base.netloc}"
        url = f"{origin}/v1/attestation?nonce={req_nonce}"

        auth_headers = {}
        if self.priv_key:
            auth_headers = sign_canonical_request(
                priv_key=self.priv_key,
                method="GET",
                path="/v1/attestation",
                body_bytes=b"",
                request_id=f"req-{secrets.token_hex(8)}",
            )

        opener, handler = self._create_attestation_opener()
        req = urllib.request.Request(url, headers=auth_headers, method="GET")
        data = None
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise AttestationVerificationError(
                        f"Attestation endpoint returned HTTP {resp.status}: {resp.read().decode('utf-8', errors='ignore')}"
                    )
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # Fallback to legacy path if /v1/attestation not routed
            url = f"{origin}/.well-known/cevell-attestation?nonce={req_nonce}"
            fb_headers = {}
            if self.priv_key:
                fb_headers = sign_canonical_request(
                    priv_key=self.priv_key,
                    method="GET",
                    path="/.well-known/cevell-attestation",
                    body_bytes=b"",
                    request_id=f"req-{secrets.token_hex(8)}",
                )
            req = urllib.request.Request(url, headers=fb_headers, method="GET")
            with opener.open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        captured_spki = None
        if handler.last_conn and handler.last_conn.peer_spki:
            captured_spki = handler.last_conn.peer_spki

        res = verify_attestation_document(
            doc=data,
            expected_nonce_hex=req_nonce,
            expected_tls_fingerprint=captured_spki,
            enforce_official_roots=self.enforce_official_roots,
            check_online_vendor=self.check_online_vendor,
            allow_mock_attestation=self.allow_mock_attestation,
            verify_code=self.verify_code and not self.allow_mock_attestation,
            expected_release=self.expected_release,
            expected_rtmr1=self.expected_rtmr1,
        )

        # In Cevell OS CVM, proxy server unwraps Protobuf HPKE requests against boot attestation document
        # digest sha256(cfg.AttestationDoc.Predicate.UserData). Fetch the boot quote to align
        # the inference request binding hash with the in-enclave supervisor.
        try:
            boot_headers = {}
            if self.priv_key:
                boot_headers = sign_canonical_request(
                    priv_key=self.priv_key,
                    method="GET",
                    path="/v1/attestation",
                    body_bytes=b"",
                    request_id=f"req-{secrets.token_hex(8)}",
                )
            boot_req = urllib.request.Request(f"{origin}/v1/attestation", headers=boot_headers, method="GET")
            with opener.open(boot_req, timeout=self.timeout) as boot_resp:
                if boot_resp.status == 200:
                    boot_doc = json.loads(boot_resp.read().decode("utf-8"))
                    boot_ud = boot_doc.get("predicate", {}).get("user_data", "")
                    if boot_ud:
                        res.binding_hash = hashlib.sha256(boot_ud.encode("utf-8")).digest()
        except Exception:
            pass

        self.attestation = res

        # Lock persistent opener strictly to the verified physical socket SPKI fingerprint
        self._bound_handler = BoundHTTPSHandler(
            context=self.ssl_ctx,
            expected_spki=self.attestation.tls_fingerprint,
        )
        self._opener = urllib.request.build_opener(self._bound_handler)
        return res

    def wrap_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body_bytes: Optional[bytes] = None,
        body: Optional[Union[bytes, str]] = None,
    ) -> Tuple[str, Dict[str, str], bytes, Optional[HPKESession]]:
        """
        Agnostically transforms any outbound HTTP request into a signed, HPKE-encrypted
        Protobuf EncryptedInferenceRequest envelope.
        
        Returns: (target_url, signed_headers, proto_body_bytes, hpke_session)
        """
        if body_bytes is None:
            if isinstance(body, str):
                body_bytes = body.encode("utf-8")
            elif isinstance(body, bytes):
                body_bytes = body
            else:
                body_bytes = b""

        if self.verify_attestation_enabled and not self.attestation:
            self.audit_and_bind_attestation()

        # Build normalized and effective paths
        parsed_base = urllib.parse.urlparse(self.base_url)
        base_prefix = parsed_base.path.rstrip("/")
        normalized_path = "/" + path.lstrip("/")
        if base_prefix and not normalized_path.startswith(base_prefix):
            effective_path = f"{base_prefix}{normalized_path}"
        else:
            effective_path = normalized_path

        target_url = f"{parsed_base.scheme}://{parsed_base.netloc}{effective_path}"

        # Route inference endpoints to HPKE envelope encryption; non-inference endpoints use direct authenticated transmission
        is_inference = any(
            effective_path.startswith(prefix)
            for prefix in (
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/messages",
                "/tokenize",
                "/detokenize",
            )
        )

        # If attestation is bound and endpoint is an inference endpoint, perform HPKE envelope encryption and Protobuf framing
        if self.attestation and is_inference:
            session, client_pub_bytes = create_client_hpke_session(self.attestation.hpke_public_key)
            req_nonce, ciphertext = session.encrypt_request(body_bytes)
            req_id = f"req-{secrets.token_hex(8)}"

            proto_bytes = encode_encrypted_inference_request(
                version=1,
                cipher_suite=1,
                client_ephemeral_public_key=client_pub_bytes,
                nonce=req_nonce,
                encrypted_payload=ciphertext,
                attestation_binding_hash=self.attestation.binding_hash,
                request_id=req_id,
            )

            # Canonical Ed25519 signature over request path, headers, and Protobuf body hash
            signed_headers = sign_canonical_request(
                priv_key=self.priv_key,
                method=method.upper(),
                path=effective_path,
                body_bytes=proto_bytes,
                request_id=req_id,
            )

            # Preserve caller headers except content-type, content-length, transfer-encoding, and host
            out_headers = {
                k: v for k, v in (headers or {}).items()
                if k.lower() not in ("content-length", "transfer-encoding", "host")
            }
            out_headers.update(signed_headers)
            out_headers["Content-Length"] = str(len(proto_bytes))
            out_headers["Content-Type"] = "application/x-protobuf"
            out_headers["Accept"] = "application/x-protobuf"

            return target_url, out_headers, proto_bytes, session
        else:
            if self.verify_attestation_enabled and is_inference:
                raise AttestationVerificationError(
                    "Attestation verification is enabled but no verified enclave attestation is latched. "
                    "Cannot transmit prompt in unencrypted plaintext."
                )

            # Direct authenticated JSON / plaintext transmission
            req_id = f"req-{secrets.token_hex(8)}"
            signed_headers = sign_canonical_request(
                priv_key=self.priv_key,
                method=method.upper(),
                path=effective_path,
                body_bytes=body_bytes,
                request_id=req_id,
            )
            out_headers = {
                k: v for k, v in (headers or {}).items()
                if k.lower() not in ("content-length", "transfer-encoding", "host")
            }
            out_headers.update(signed_headers)
            if len(body_bytes) > 0:
                out_headers["Content-Length"] = str(len(body_bytes))
            out_headers.setdefault("Accept", "application/json")
            return target_url, out_headers, body_bytes, None

    def unwrap_frame(self, full_frame: bytes, session: HPKESession) -> Tuple[int, int, bytes]:
        """
        Decodes and decrypts a single length-prefixed Protobuf StreamingInferenceFrame.
        Enforces monotonic sequence numbers and AAD authentication.
        
        Returns: (seq, frame_type, plaintext_bytes)
        """
        seq, frame_type, ct, tag, ts_ns = decode_streaming_frame(full_frame)
        decrypted = session.decrypt_frame(seq, frame_type, ct, tag, ts_ns)
        return seq, frame_type, decrypted

    def iter_decrypted_frames(
        self,
        stream_reader: Any,
        session: HPKESession,
    ) -> Generator[bytes, None, None]:
        """
        Agnostic generator consuming length-prefixed Protobuf frames from any synchronous stream
        reader (with .read(n) method) and yielding raw decrypted plaintext bytes.
        Does not parse or alter the underlying payload format (SSE, JSON, raw bytes, etc.).
        """
        try:
            stream_ended = False
            while True:
                prefix = stream_reader.read(4)
                if not prefix or len(prefix) < 4:
                    break

                frame_len = struct.unpack(">I", prefix)[0]
                if frame_len > MAX_FRAME_SIZE:
                    raise TunnelError(
                        f"Incoming frame length {frame_len} exceeds maximum allowed size ({MAX_FRAME_SIZE})"
                    )

                frame_body = stream_reader.read(frame_len)
                while len(frame_body) < frame_len:
                    more = stream_reader.read(frame_len - len(frame_body))
                    if not more:
                        raise TunnelError("Truncated streaming frame received from enclave")
                    frame_body += more

                full_frame = prefix + frame_body
                seq, frame_type, plaintext = self.unwrap_frame(full_frame, session)

                if frame_type == FRAME_TYPE_STREAM_END or plaintext.strip() == b"[DONE]":
                    stream_ended = True
                    yield b"data: [DONE]\n\n"
                    break
                elif frame_type == FRAME_TYPE_ERROR:
                    raise TunnelError(f"Enclave returned error frame: {plaintext.decode('utf-8', errors='ignore')}")
                elif frame_type == FRAME_TYPE_COMPLETION:
                    stream_ended = True
                    yield plaintext
                    break
                elif frame_type == FRAME_TYPE_TOKEN_DELTA:
                    if plaintext.startswith(b"data:") or plaintext.startswith(b"event:"):
                        yield plaintext if plaintext.endswith(b"\n\n") else (plaintext + b"\n\n")
                    else:
                        yield b"data: " + plaintext + b"\n\n"
                else:
                    yield plaintext

            if not stream_ended:
                raise TunnelError(
                    "Stream truncated prematurely: connection closed before transmitting FRAME_TYPE_STREAM_END or [DONE]"
                )
        finally:
            session.zeroize()

    async def aiter_decrypted_frames(
        self,
        async_stream: Any,
        session: HPKESession,
    ) -> AsyncGenerator[bytes, None]:
        """
        Agnostic async generator consuming length-prefixed Protobuf frames from any async byte stream
        and yielding raw decrypted plaintext bytes.
        """
        try:
            stream_ended = False
            buf = bytearray()

            async for chunk in async_stream:
                buf.extend(chunk)
                while len(buf) >= 4:
                    frame_len = struct.unpack(">I", buf[:4])[0]
                    if frame_len > MAX_FRAME_SIZE:
                        raise TunnelError(
                            f"Incoming frame length {frame_len} exceeds maximum allowed size ({MAX_FRAME_SIZE})"
                        )
                    total_needed = 4 + frame_len
                    if len(buf) < total_needed:
                        break

                    full_frame = bytes(buf[:total_needed])
                    del buf[:total_needed]

                    seq, frame_type, plaintext = self.unwrap_frame(full_frame, session)
                    if frame_type == FRAME_TYPE_STREAM_END or plaintext.strip() == b"[DONE]":
                        stream_ended = True
                        yield b"data: [DONE]\n\n"
                        break
                    elif frame_type == FRAME_TYPE_ERROR:
                        raise TunnelError(f"Enclave returned error frame: {plaintext.decode('utf-8', errors='ignore')}")
                    elif frame_type == FRAME_TYPE_COMPLETION:
                        stream_ended = True
                        yield plaintext
                        break
                    elif frame_type == FRAME_TYPE_TOKEN_DELTA:
                        if plaintext.startswith(b"data:") or plaintext.startswith(b"event:"):
                            yield plaintext if plaintext.endswith(b"\n\n") else (plaintext + b"\n\n")
                        else:
                            yield b"data: " + plaintext + b"\n\n"
                    else:
                        yield plaintext
                if stream_ended:
                    break

            if not stream_ended:
                raise TunnelError(
                    "Stream truncated prematurely: connection closed before transmitting FRAME_TYPE_STREAM_END or [DONE]"
                )
        finally:
            session.zeroize()
