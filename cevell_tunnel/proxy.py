"""
ox.proxy — Agnostic Local Loopback Proxy Daemon
================================================
Spins up a lightweight local loopback HTTP server (127.0.0.1:8080) that transparently
intercepts standard unencrypted HTTP requests from any application, CLI tool, or SDK
(curl, LangChain, Cursor IDE, Continue.dev, Python, Go, Rust), performing hardware
attestation and HPKE envelope encryption under the hood.

Usage:
    export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
    export OPENAI_API_KEY="dummy"
    curl http://127.0.0.1:8080/v1/chat/completions -d '{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'
"""

import sys
import json
import socket
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Union
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler

from .tunnel import ConfidentialTunnel, TunnelError

logger = logging.getLogger("ox.proxy")


class TunnelProxyHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler that routes incoming HTTP requests through the ConfidentialTunnel.
    """
    tunnel: ConfidentialTunnel
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Standard silent logging unless configured
        logger.debug(format, *args)

    def do_OPTIONS(self) -> None:
        """CORS preflight support for local browser applications."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        self._handle_request("GET")

    def do_POST(self) -> None:
        self._handle_request("POST")

    def do_PUT(self) -> None:
        self._handle_request("PUT")

    def do_DELETE(self) -> None:
        self._handle_request("DELETE")

    def _handle_request(self, method: str) -> None:
        # 1. Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

        path = self.path

        # Handle direct discovery and health probes without encryption
        if path.startswith("/v1/attestation") or path.startswith("/.well-known/") or path == "/v1/health":
            target_url = f"{self.tunnel.base_url}{path}"
            fwd_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
            req = urllib.request.Request(target_url, data=body_bytes if body_bytes else None, headers=fwd_headers, method=method)
            try:
                with self.tunnel._opener.open(req, timeout=self.tunnel.timeout) as resp:
                    resp_body = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
            except urllib.error.HTTPError as e:
                err_body = e.read()
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                return

        # 2. Wrap through confidential tunnel
        fwd_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        try:
            target_url, signed_headers, proto_bytes, session = self.tunnel.wrap_request(
                method=method,
                path=path,
                headers=fwd_headers,
                body_bytes=body_bytes,
            )
        except Exception as e:
            err_msg = json.dumps({"error": f"Confidential tunnel encryption failure: {e}"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
            return

        # 3. Forward to CVM
        http_req = urllib.request.Request(
            target_url,
            data=proto_bytes,
            headers=signed_headers,
            method=method,
        )

        try:
            resp = self.tunnel._opener.open(http_req, timeout=self.tunnel.timeout)
        except urllib.error.HTTPError as e:
            err_bytes = e.read()
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)
            return
        except Exception as e:
            err_msg = json.dumps({"error": f"CVM upstream connection failure: {e}"}).encode("utf-8")
            self.send_response(504)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
            return

        # 4. Stream or return decrypted response
        if session:
            # Length-prefixed Protobuf frames -> HTTP Chunked Transfer Encoding
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for decrypted_chunk in self.tunnel.iter_decrypted_frames(resp, session):
                    if not decrypted_chunk:
                        continue
                    # Chunk format: <hex_size>\r\n<data>\r\n
                    chunk_header = f"{len(decrypted_chunk):X}\r\n".encode("ascii")
                    self.wfile.write(chunk_header + decrypted_chunk + b"\r\n")
                    self.wfile.flush()

                # Terminal chunk
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception as e:
                logger.error("Error while streaming decrypted frames: %s", e)
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
        else:
            resp_bytes = resp.read()
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp_bytes)


class LocalCVMProxy:
    """
    Manages the lifecycle of a local loopback daemon forwarding traffic through ConfidentialTunnel.
    """

    def __init__(
        self,
        cvm_url: Optional[str] = None,
        ip: Optional[str] = None,
        port: int = 443,
        auth_key: Union[str, bytes] = "auth.pem",
        listen_host: str = "127.0.0.1",
        listen_port: int = 8080,
        tunnel: Optional[ConfidentialTunnel] = None,
        **tunnel_kwargs
    ):
        self.listen_host = listen_host
        self.listen_port = listen_port

        if tunnel:
            self.tunnel = tunnel
        else:
            self.tunnel = ConfidentialTunnel(
                cvm_url=cvm_url,
                ip=ip,
                port=port,
                auth_key=auth_key,
                **tunnel_kwargs
            )

        class BoundHandler(TunnelProxyHandler):
            tunnel = self.tunnel

        self.server = ThreadingHTTPServer((self.listen_host, self.listen_port), BoundHandler)
        self.server_port = self.server.server_port

    def start(self) -> None:
        """Runs the proxy server on the current thread (blocking)."""
        print(f"[*] Cevell Confidential Tunnel Proxy listening on http://{self.listen_host}:{self.server_port}")
        print(f"[*] Target CVM Enclave: {self.tunnel.base_url}")
        print(f"[*] Configure your SDK or tools with:")
        print(f"    export OPENAI_BASE_URL=\"http://{self.listen_host}:{self.server_port}/v1\"")
        print(f"    export OPENAI_API_KEY=\"dummy\"\n")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
