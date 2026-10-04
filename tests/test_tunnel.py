"""
Unit & Integration Tests for Agnostic Confidential Tunnel, Transport, and Proxy
================================================================================
Tests:
  - ConfidentialTunnel request wrapping, frame unwrapping, and streaming iteration
  - ConfidentialTransport with httpx.Client and official OpenAI Python SDK
  - AsyncConfidentialTransport with httpx.AsyncClient and AsyncOpenAI
  - LocalCVMProxy daemon transparently proxying standard HTTP requests
"""

import os
import ssl
import json
import time
import struct
import socket
import hashlib
import tempfile
import threading
import unittest
import urllib.request
import asyncio

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from http.server import HTTPServer, BaseHTTPRequestHandler

import httpx
import openai

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cevell_tunnel import (
    ConfidentialTunnel,
    TunnelError,
    ConfidentialTransport,
    AsyncConfidentialTransport,
    create_http_client,
    create_async_http_client,
    LocalCVMProxy,
)
from cevell_tunnel.wire import (
    decode_protobuf,
    encode_field_varint,
    encode_field_bytes,
    FRAME_TYPE_TOKEN_DELTA,
    FRAME_TYPE_STREAM_END,
)
from cevell_tunnel.crypto import compute_frame_nonce, compute_frame_aad


def generate_test_cert():
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


class MockCVMServer(BaseHTTPRequestHandler):
    server_x25519_priv: x25519.X25519PrivateKey
    server_x25519_pub: bytes
    authorized_ed25519_pub: ed25519.Ed25519PublicKey

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/attestation"):
            query_nonce = ""
            if "nonce=" in self.path:
                query_nonce = self.path.split("nonce=")[1].split("&")[0]

            tls_fp = getattr(MockCVMServer, "server_tls_fp", b"\xbb" * 32)
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
                    "nonce": query_nonce,
                }
            }
            body = json.dumps(doc).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/health":
            body = b'{"status":"ok","model":"Qwen/Qwen2.5-7B-Instruct"}'
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

            # Check protobuf header
            fields = decode_protobuf(body_bytes)
            client_pub_bytes = fields[3]
            req_nonce = fields[4]
            ciphertext = fields[5]

            shared_secret = self.server_x25519_priv.exchange(
                x25519.X25519PublicKey.from_public_bytes(client_pub_bytes)
            )
            req_key = HKDF(hashes.SHA256(), 32, None, b"cevell-hpke-req-aes-gcm").derive(shared_secret)
            resp_key = HKDF(hashes.SHA256(), 32, req_nonce, b"cevell-hpke-resp-aes-gcm").derive(shared_secret)
            resp_iv = HKDF(hashes.SHA256(), 12, req_nonce, b"cevell-hpke-resp-base-iv").derive(shared_secret)

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
                f0 = make_frame(0, FRAME_TYPE_TOKEN_DELTA, b'data: {"id":"chat-1","choices":[{"delta":{"content":"Hello from "}}]}\n\n')
                f1 = make_frame(1, FRAME_TYPE_TOKEN_DELTA, b'data: {"id":"chat-1","choices":[{"delta":{"content":"Confidential CVM!"}}]}\n\n')
                f2 = make_frame(2, FRAME_TYPE_STREAM_END, b"data: [DONE]\n\n")
                self.wfile.write(f0 + f1 + f2)
            else:
                resp_payload = json.dumps({
                    "id": "chat-unary-1",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Confidential CVM!"},
                        "finish_reason": "stop"
                    }]
                }).encode("utf-8")
                f0 = make_frame(0, 2, resp_payload)
                self.wfile.write(f0)
        else:
            self.send_response(404)
            self.end_headers()


class TestTunnelAndTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cert_pem, key_pem = generate_test_cert()
        cls.cert_file = tempfile.NamedTemporaryFile(delete=False)
        cls.cert_file.write(cert_pem)
        cls.cert_file.flush()

        cls.key_file = tempfile.NamedTemporaryFile(delete=False)
        cls.key_file.write(key_pem)
        cls.key_file.flush()

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

        MockCVMServer.server_x25519_priv = cls.server_priv
        MockCVMServer.server_x25519_pub = cls.server_pub
        MockCVMServer.authorized_ed25519_pub = cls.client_ed_pub
        cert_obj = x509.load_pem_x509_certificate(cert_pem)
        spki_bytes = cert_obj.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        cls.real_tls_fp = hashlib.sha256(spki_bytes).digest()
        MockCVMServer.server_tls_fp = cls.real_tls_fp

        cls.httpd = HTTPServer(("127.0.0.1", 0), MockCVMServer)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cls.cert_file.name, keyfile=cls.key_file.name)
        cls.httpd.socket = ctx.wrap_socket(cls.httpd.socket, server_side=True)
        cls.port = cls.httpd.server_port

        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.cvm_url = f"https://127.0.0.1:{cls.port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        for f in (cls.cert_file, cls.key_file, cls.client_key_file):
            try:
                os.unlink(f.name)
            except OSError:
                pass

    def test_confidential_tunnel_wrap_and_unwrap(self):
        tunnel = ConfidentialTunnel(
            cvm_url=self.cvm_url,
            auth_key=self.client_key_file.name,
            allow_mock_attestation=True,
        )

        target_url, headers, body, session = tunnel.wrap_request(
            method="POST",
            path="/chat/completions",
            body=b'{"model":"test","messages":[{"role":"user","content":"hi"}]}'
        )

        self.assertIn("/v1/chat/completions", target_url)

        self.assertIn("Content-Type", headers)
        self.assertEqual(headers["Content-Type"], "application/x-protobuf")
        self.assertIn("x-signature", headers)
        self.assertIsNotNone(session)
        self.assertTrue(len(body) > 0)

    def test_confidential_transport_with_openai_client(self):
        # Instantiate official OpenAI client with our custom ConfidentialTransport
        client = openai.OpenAI(
            base_url=self.cvm_url,
            api_key="dummy-not-needed",
            http_client=httpx.Client(
                transport=ConfidentialTransport(
                    cvm_url=self.cvm_url,
                    auth_key=self.client_key_file.name,
                    allow_mock_attestation=True,
                )
            ),
        )

        # 1. Unary non-streaming completion
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        self.assertEqual(resp.choices[0].message.content, "Hello from Confidential CVM!")

        # 2. Streaming completion
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        self.assertEqual("".join(chunks), "Hello from Confidential CVM!")

    def test_async_confidential_transport(self):
        async def run_async_test():
            # Instantiate official AsyncOpenAI client with our AsyncConfidentialTransport
            client = openai.AsyncOpenAI(
                base_url=self.cvm_url,
                api_key="dummy-not-needed",
                http_client=httpx.AsyncClient(
                    transport=AsyncConfidentialTransport(
                        cvm_url=self.cvm_url,
                        auth_key=self.client_key_file.name,
                        allow_mock_attestation=True,
                    )
                ),
            )

            resp = await client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )
            self.assertEqual(resp.choices[0].message.content, "Hello from Confidential CVM!")

            stream = await client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            chunks = []
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
            self.assertEqual("".join(chunks), "Hello from Confidential CVM!")

        asyncio.run(run_async_test())

    def test_local_cvm_proxy_server(self):
        proxy = LocalCVMProxy(
            cvm_url=self.cvm_url,
            auth_key=self.client_key_file.name,
            listen_host="127.0.0.1",
            listen_port=0,
            allow_mock_attestation=True,
        )

        proxy_thread = threading.Thread(target=proxy.server.serve_forever, daemon=True)
        proxy_thread.start()
        proxy_port = proxy.server_port

        try:
            # Standard unmodified OpenAI client talking to local HTTP loopback proxy
            client = openai.OpenAI(
                base_url=f"http://127.0.0.1:{proxy_port}/v1",
                api_key="dummy-not-needed",
            )

            # Test unary completion through proxy
            resp = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )
            self.assertEqual(resp.choices[0].message.content, "Hello from Confidential CVM!")

            # Test streaming completion through proxy
            stream = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            collected = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
            self.assertEqual(collected, "Hello from Confidential CVM!")

        finally:
            proxy.shutdown()


if __name__ == "__main__":
    unittest.main()
