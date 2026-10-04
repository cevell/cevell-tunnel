"""
ox.client — High-Level Attestation-Verified Confidential CVM Client
===================================================================
Provides a seamless, OpenAI-compatible client for confidential inference:
  - Fetches and verifies hardware attestation (NVIDIA DICE + Intel TDX / AMD SEV-SNP)
  - Cryptographically verifies TDX quotes, DICE leaf signatures, and vendor CRL status
  - Enforces TLS socket SPKI channel binding against silicon REPORT_DATA
  - Binds the in-enclave HPKE public key
  - Signs canonical requests with the latched tenant Ed25519 key (aws.pem / auth.pem)
  - Encapsulates prompts in RFC 9180 HPKE Protobuf envelopes (cevell.wire.v1)
  - Decrypts streaming frames in real-time
"""

import json
import ssl
import time
import struct
import secrets
import hashlib
import http.client
import urllib.request
import urllib.error
from typing import Dict, Any, List, Generator, Optional, Union, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from .crypto import load_auth_key, sign_canonical_request, create_client_hpke_session, HPKESession
from .wire import (
    MAX_FRAME_SIZE,
    encode_encrypted_inference_request,
    decode_streaming_frame,
    FRAME_TYPE_TOKEN_DELTA,
    FRAME_TYPE_COMPLETION,
    FRAME_TYPE_STREAM_END,
    FRAME_TYPE_ERROR,
)
from .attestation import verify_attestation_document, AttestationResult, AttestationVerificationError


class CVMInferenceError(Exception):
    """Raised when an inference error frame or HTTP failure occurs."""
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


class _ChatCompletionsAdapter:
    """Provides client.chat.completions.create(...) syntax compatibility."""
    def __init__(self, client: "CVMClient"):
        self._client = client

    def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        return self._client.chat_completion(
            messages=messages,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )


class _ChatAdapter:
    def __init__(self, client: "CVMClient"):
        self.completions = _ChatCompletionsAdapter(client)


class CVMClient:
    """
    Authoritative client for interacting with Cevell OS Confidential Virtual Machines.
    """

    def __init__(
        self,
        cvm_url: Optional[str] = None,
        ip: Optional[str] = None,
        port: int = 443,
        auth_key: Union[str, bytes] = "auth.pem",
        default_model: Optional[str] = None,
        verify_attestation: bool = True,
        verify_code: bool = True,
        expected_release: Optional[str] = "latest",
        expected_rtmr1: Optional[str] = None,
        enforce_official_roots: bool = True,
        check_online_vendor: bool = True,
        timeout: float = 60.0,
        allow_insecure_http: bool = False,
        allow_mock_attestation: bool = False,
    ):
        if cvm_url:
            if cvm_url.startswith("http://"):
                if not allow_insecure_http:
                    raise ValueError(
                        "Insecure HTTP connections are disabled by default because silicon TLS channel binding "
                        "requires HTTPS. Set allow_insecure_http=True only for local testing environments."
                    )
                self.base_url = cvm_url.rstrip("/")
            elif cvm_url.startswith("https://"):
                self.base_url = cvm_url.rstrip("/")
            else:
                self.base_url = f"https://{cvm_url}"
        elif ip:
            clean_ip = ip.replace("https://", "").replace("http://", "").strip("/")
            if ":" in clean_ip:
                self.base_url = f"https://{clean_ip}"
            else:
                self.base_url = f"https://{clean_ip}:{port}"
        else:
            raise ValueError("Must provide either cvm_url or ip")

        self.default_model = default_model
        self.timeout = timeout
        self.verify_attestation_enabled = verify_attestation
        self.verify_code = verify_code
        self.expected_release = expected_release
        self.expected_rtmr1 = expected_rtmr1
        self.enforce_official_roots = enforce_official_roots
        self.check_online_vendor = check_online_vendor
        self.allow_mock_attestation = allow_mock_attestation

        # 1. Load the latched tenant Ed25519 authentication private key
        self.priv_key = load_auth_key(auth_key)

        # 2. SSL Context (permissive of self-signed TLS because channel identity is anchored to TDX/SEV silicon)
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

        # 3. Default opener before attestation binding
        self._bound_handler = BoundHTTPSHandler(context=self.ssl_ctx, expected_spki=None)
        self._opener = urllib.request.build_opener(self._bound_handler)

        # 4. Attestation state
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
        url = f"{self.base_url}/v1/attestation?nonce={req_nonce}"

        opener, handler = self._create_attestation_opener()

        req = urllib.request.Request(url, method="GET")
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
            url = f"{self.base_url}/.well-known/cevell-attestation?nonce={req_nonce}"
            req = urllib.request.Request(url, method="GET")
            with opener.open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        captured_spki = None
        if handler.last_conn and handler.last_conn.peer_spki:
            captured_spki = handler.last_conn.peer_spki

        # Enforce strict challenge-response nonce freshness to prevent quote replay attacks
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

        # In Cevell OS CVM, the proxy server unwraps Protobuf HPKE requests against the boot attestation document
        # digest sha256(cfg.AttestationDoc.Predicate.UserData). Fetch the boot quote to ensure
        # the inference request binding hash aligns with the in-enclave proxy supervisor.
        try:
            boot_req = urllib.request.Request(f"{self.base_url}/v1/attestation", method="GET")
            with opener.open(boot_req, timeout=self.timeout) as boot_resp:
                if boot_resp.status == 200:
                    boot_doc = json.loads(boot_resp.read().decode("utf-8"))
                    boot_ud = boot_doc.get("predicate", {}).get("user_data", "")
                    if boot_ud:
                        res.binding_hash = hashlib.sha256(boot_ud.encode("utf-8")).digest()
        except Exception:
            pass

        self.attestation = res

        # Reconfigure persistent opener locked strictly to the verified TLS SPKI fingerprint
        self._bound_handler = BoundHTTPSHandler(
            context=self.ssl_ctx,
            expected_spki=self.attestation.tls_fingerprint,
        )
        self._opener = urllib.request.build_opener(self._bound_handler)

        return res

    def load_model(
        self,
        model: str,
        expose_in_health: bool = True,
        bitsandbytes: bool = False,
        gguf: bool = False,
        quantization: Optional[str] = None,
        timeout: float = 600.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Dynamically orders model provisioning on the CVM via authenticated POST /v1/models/load.

        :param model: HuggingFace model repository ID (e.g. 'unsloth/Llama-3.2-3B-Instruct') or GGUF path/repo
        :param expose_in_health: Whether the loaded model name is visible on unauthenticated GET /v1/health
        :param bitsandbytes: Enable BitsAndBytes quantization plugin (--quantization bitsandbytes)
        :param gguf: Enable GGUF format plugin (--quantization gguf)
        :param quantization: Explicit quantization method override (e.g. "bitsandbytes", "gguf", "fp8", "awq")
        :param timeout: Maximum wait time in seconds for model download and initialization (default: 600s)
        :param kwargs: Additional model parameters passed to CVM inference engine
        :return: Response JSON containing load status and model metadata
        """
        payload: Dict[str, Any] = {
            "model": model,
            "expose_in_health": expose_in_health,
        }
        if bitsandbytes:
            payload["bitsandbytes"] = True
        if gguf:
            payload["gguf"] = True
        if quantization:
            payload["quantization"] = quantization
        payload.update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        req_id = f"req-{secrets.token_hex(8)}"
        headers = sign_canonical_request(
            priv_key=self.priv_key,
            method="POST",
            path="/v1/models/load",
            body_bytes=body,
            request_id=req_id,
        )
        headers["Content-Type"] = "application/json"
        url = f"{self.base_url}/v1/models/load"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with self._opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @property
    def chat(self) -> _ChatAdapter:
        """OpenAI-compatible client.chat.completions.create(...) accessor."""
        return _ChatAdapter(self)

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        High-level convenience generator yielding raw token delta strings.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream_gen = self.chat_completion(
            messages=messages,
            model=model,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        for chunk in stream_gen:
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta

    def ask(self, prompt: str, **kwargs) -> str:
        """
        Sends a single prompt and returns the full aggregated response text.
        """
        parts = []
        for delta in self.stream(prompt, **kwargs):
            parts.append(delta)
        return "".join(parts)

    def get_active_model(self) -> str:
        """Discovers the currently loaded model on the CVM from /v1/health."""
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/health", method="GET")
            with self._opener.open(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("model"):
                    self.default_model = data["model"]
                    return data["model"]
        except Exception:
            pass
        return self.default_model or "Qwen/Qwen2.5-7B-Instruct"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Executes an end-to-end encrypted inference request against the CVM.
        """
        if not self.attestation and self.verify_attestation_enabled:
            self.audit_and_bind_attestation()

        target_model = model or self.default_model or self.get_active_model()

        # 1. Build raw OpenAI-compatible JSON payload
        raw_payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        plaintext_bytes = json.dumps(raw_payload).encode("utf-8")

        if self.attestation:
            # 2. HPKE Encryption using the verified CVM public key
            session, client_pub_bytes = create_client_hpke_session(self.attestation.hpke_public_key)

            req_nonce, ciphertext = session.encrypt_request(plaintext_bytes)
            req_id = f"req-{secrets.token_hex(8)}"

            # 3. Serialize into Protobuf EncryptedInferenceRequest
            proto_req_bytes = encode_encrypted_inference_request(
                version=1,
                cipher_suite=1,
                client_ephemeral_public_key=client_pub_bytes,
                nonce=req_nonce,
                encrypted_payload=ciphertext,
                attestation_binding_hash=self.attestation.binding_hash,
                request_id=req_id,
            )

            # 4. Sign canonical HTTP request using the latched Ed25519 tenant key
            headers = sign_canonical_request(
                priv_key=self.priv_key,
                method="POST",
                path="/v1/chat/completions",
                body_bytes=proto_req_bytes,
                request_id=req_id,
            )
            headers["Content-Type"] = "application/x-protobuf"
            headers["Accept"] = "application/x-protobuf"

            url = f"{self.base_url}/v1/chat/completions"
            http_req = urllib.request.Request(url, data=proto_req_bytes, headers=headers, method="POST")

            if stream:
                return self._execute_streaming_request(http_req, session)
            else:
                return self._execute_unary_request(http_req, session)
        else:
            if self.verify_attestation_enabled:
                raise AttestationVerificationError(
                    "Attestation verification is enabled but no verified enclave attestation is latched. "
                    "Cannot transmit prompt in unencrypted plaintext."
                )
            # Standard authenticated JSON inference
            req_id = f"req-{secrets.token_hex(8)}"
            headers = sign_canonical_request(
                priv_key=self.priv_key,
                method="POST",
                path="/v1/chat/completions",
                body_bytes=plaintext_bytes,
                request_id=req_id,
            )
            headers["Content-Type"] = "application/json"

            url = f"{self.base_url}/v1/chat/completions"
            http_req = urllib.request.Request(url, data=plaintext_bytes, headers=headers, method="POST")

            if stream:
                return self._execute_plain_streaming_request(http_req)
            else:
                return self._execute_plain_unary_request(http_req)

    def _execute_plain_streaming_request(
        self,
        http_req: urllib.request.Request,
    ) -> Generator[Dict[str, Any], None, None]:
        """Reads plain Server-Sent Events (SSE) streaming chunks from CVM."""
        with self._opener.open(http_req, timeout=self.timeout) as resp:
            if resp.status >= 400:
                err_msg = resp.read().decode("utf-8", errors="ignore")
                raise CVMInferenceError(f"HTTP {resp.status}: {err_msg}")

            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    clean = line[5:].strip()
                    if clean == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(clean)
                        yield chunk_json
                    except Exception:
                        pass

    def _execute_plain_unary_request(
        self,
        http_req: urllib.request.Request,
    ) -> Dict[str, Any]:
        """Reads plain JSON completion from CVM."""
        with self._opener.open(http_req, timeout=self.timeout) as resp:
            if resp.status >= 400:
                err_msg = resp.read().decode("utf-8", errors="ignore")
                raise CVMInferenceError(f"HTTP {resp.status}: {err_msg}")
            data = resp.read().decode("utf-8")
            return json.loads(data)

    def _execute_streaming_request(
        self,
        http_req: urllib.request.Request,
        session: HPKESession
    ) -> Generator[Dict[str, Any], None, None]:
        """Reads length-prefixed streaming frames and yields parsed SSE chunk dicts."""
        try:
            with self._opener.open(http_req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    err_msg = resp.read().decode("utf-8", errors="ignore")
                    raise CVMInferenceError(f"HTTP {resp.status}: {err_msg}")

                stream_ended = False
                while True:
                    # Read 4-byte big-endian length prefix
                    prefix = resp.read(4)
                    if not prefix or len(prefix) < 4:
                        break

                    frame_len = struct.unpack(">I", prefix)[0]
                    if frame_len > MAX_FRAME_SIZE:
                        raise CVMInferenceError(
                            f"Streaming frame length {frame_len} exceeds maximum allowed size ({MAX_FRAME_SIZE})"
                        )

                    frame_body = resp.read(frame_len)
                    while len(frame_body) < frame_len:
                        more = resp.read(frame_len - len(frame_body))
                        if not more:
                            raise CVMInferenceError("Truncated streaming frame received")
                        frame_body += more

                    full_frame = prefix + frame_body
                    seq, frame_type, ct, tag, ts_ns = decode_streaming_frame(full_frame)

                    decrypted = session.decrypt_frame(seq, frame_type, ct, tag, ts_ns)
                    text = decrypted.decode("utf-8")

                    if frame_type == FRAME_TYPE_STREAM_END or text.strip() == "[DONE]":
                        stream_ended = True
                        break
                    elif frame_type == FRAME_TYPE_ERROR:
                        raise CVMInferenceError(f"Enclave returned error frame: {text}")

                    # Each decrypted frame contains the SSE data payload (JSON)
                    for line in text.split("\n"):
                        clean = line.strip()
                        if clean.startswith("data:"):
                            clean = clean[5:].strip()
                        if not clean or clean == "[DONE]":
                            if clean == "[DONE]":
                                stream_ended = True
                            continue
                        try:
                            chunk_json = json.loads(clean)
                            yield chunk_json
                        except Exception:
                            pass

                if not stream_ended:
                    raise CVMInferenceError(
                        "Stream truncated prematurely: server closed connection before transmitting FRAME_TYPE_STREAM_END or [DONE]"
                    )
        finally:
            session.zeroize()

    def _execute_unary_request(
        self,
        http_req: urllib.request.Request,
        session: HPKESession
    ) -> Dict[str, Any]:
        """Reads non-streaming completion frames and returns the parsed completion dict."""
        try:
            with self._opener.open(http_req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    err_msg = resp.read().decode("utf-8", errors="ignore")
                    raise CVMInferenceError(f"HTTP {resp.status}: {err_msg}")

                # Read first length-prefixed frame
                prefix = resp.read(4)
                if not prefix or len(prefix) < 4:
                    raise CVMInferenceError("Empty response body from CVM")

                frame_len = struct.unpack(">I", prefix)[0]
                if frame_len > MAX_FRAME_SIZE:
                    raise CVMInferenceError(
                        f"Response frame length {frame_len} exceeds maximum allowed size ({MAX_FRAME_SIZE})"
                    )

                frame_body = resp.read(frame_len)
                while len(frame_body) < frame_len:
                    more = resp.read(frame_len - len(frame_body))
                    if not more:
                        break
                    frame_body += more

                full_frame = prefix + frame_body
                seq, frame_type, ct, tag, ts_ns = decode_streaming_frame(full_frame)

                decrypted = session.decrypt_frame(seq, frame_type, ct, tag, ts_ns)
                text = decrypted.decode("utf-8")

                if frame_type == FRAME_TYPE_ERROR:
                    raise CVMInferenceError(f"Enclave returned error frame: {text}")

                return json.loads(text)
        finally:
            session.zeroize()
