"""
ox.transport — Agnostic Confidential HTTPX Transport for Mature SDKs
=====================================================================
Plugs the zero-trust hardware attestation and HPKE envelope encryption layer
directly into standard HTTP clients (such as the official `openai` or `anthropic` SDKs):
  - Subclasses `httpx.BaseTransport` (sync) and `httpx.AsyncBaseTransport` (async)
  - Intercepts raw HTTP requests across ANY endpoint (chat, embeddings, models, audio)
  - Agnostically transforms request bodies into signed RFC 9180 HPKE Protobuf envelopes
  - Yields decrypted raw plaintext bytes directly into standard HTTP response streams
  - Provides 100% feature parity with OpenAI tool calling, Pydantic parsing, retries, and streaming
"""

import urllib.request
import urllib.error
import secrets
from typing import Optional, Dict, Any, Union, Iterator, AsyncIterator

from .tunnel import ConfidentialTunnel, TunnelError
from .attestation import AttestationVerificationError
from .crypto import sign_canonical_request

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False
    _SyncByteStream = object
    _AsyncByteStream = object
else:
    _SyncByteStream = httpx.SyncByteStream
    _AsyncByteStream = httpx.AsyncByteStream


class _DecryptedSyncStream(_SyncByteStream):
    """Consumes length-prefixed Protobuf frames and yields raw decrypted plaintext bytes."""
    def __init__(self, tunnel: ConfidentialTunnel, resp: Any, session: Any):
        self.tunnel = tunnel
        self.resp = resp
        self.session = session
        self._gen = self.tunnel.iter_decrypted_frames(self.resp, self.session)

    def __iter__(self) -> Iterator[bytes]:
        yield from self._gen

    def close(self) -> None:
        try:
            self.resp.close()
        except Exception:
            pass


class _DecryptedAsyncStream(_AsyncByteStream):
    """Async variant yielding raw decrypted plaintext bytes from an async stream reader."""
    def __init__(self, tunnel: ConfidentialTunnel, async_stream: Any, session: Any):
        self.tunnel = tunnel
        self.async_stream = async_stream
        self.session = session
        self._gen = self.tunnel.aiter_decrypted_frames(self.async_stream, self.session)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._gen:
            yield chunk

    async def aclose(self) -> None:
        if hasattr(self.async_stream, "aclose"):
            await self.async_stream.aclose()


if HAS_HTTPX:
    class ConfidentialTransport(httpx.BaseTransport):
        """
        Custom sync HTTPX transport that automatically wraps requests into hardware-attested,
        HPKE-encrypted Protobuf envelopes and streams decrypted response bytes.
        """

        def __init__(
            self,
            cvm_url: Optional[str] = None,
            ip: Optional[str] = None,
            port: int = 443,
            auth_key: Union[str, bytes] = "auth.pem",
            tunnel: Optional[ConfidentialTunnel] = None,
            verify_attestation: bool = True,
            enforce_official_roots: bool = True,
            check_online_vendor: bool = True,
            allow_mock_attestation: bool = False,
            allow_insecure_http: bool = False,
            timeout: float = 60.0,
        ):
            super().__init__()
            if tunnel:
                self.tunnel = tunnel
            else:
                self.tunnel = ConfidentialTunnel(
                    cvm_url=cvm_url,
                    ip=ip,
                    port=port,
                    auth_key=auth_key,
                    verify_attestation=verify_attestation,
                    enforce_official_roots=enforce_official_roots,
                    check_online_vendor=check_online_vendor,
                    allow_mock_attestation=allow_mock_attestation,
                    allow_insecure_http=allow_insecure_http,
                    timeout=timeout,
                )

        @property
        def base_url(self) -> str:
            return f"{self.tunnel.base_url.rstrip('/')}/v1"

        @property
        def cvm_url(self) -> str:
            return self.tunnel.base_url

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            """
            Intercepts any HTTP request, performs agnostic envelope encryption, and
            returns a decrypted HTTPX response.
            """
            raw_body = request.read()
            path = request.url.raw_path.decode("ascii")

            # Bypass envelope encryption for unauthenticated/discovery routes
            if path.startswith("/v1/attestation") or path.startswith("/.well-known/"):
                req_headers = dict(request.headers)
                if self.tunnel.priv_key:
                    clean_path = path.split("?")[0]
                    signed = sign_canonical_request(
                        priv_key=self.tunnel.priv_key,
                        method=request.method,
                        path=clean_path,
                        body_bytes=raw_body or b"",
                        request_id=f"req-{secrets.token_hex(8)}",
                    )
                    req_headers.update(signed)
                req = urllib.request.Request(
                    f"{self.tunnel.base_url}{path}",
                    data=raw_body if raw_body else None,
                    headers=req_headers,
                    method=request.method,
                )
                try:
                    resp = self.tunnel._opener.open(req, timeout=self.tunnel.timeout)
                    content = resp.read()
                    return httpx.Response(
                        status_code=resp.status,
                        headers=dict(resp.headers),
                        content=content,
                    )
                except urllib.error.HTTPError as e:
                    return httpx.Response(
                        status_code=e.code,
                        headers=dict(e.headers),
                        content=e.read(),
                    )

            # Agnostic wrap: encrypts payload, wraps in Protobuf, signs canonical headers
            target_url, signed_headers, proto_bytes, session = self.tunnel.wrap_request(
                method=request.method,
                path=path,
                headers=dict(request.headers),
                body_bytes=raw_body,
            )

            req = urllib.request.Request(
                target_url,
                data=proto_bytes,
                headers=signed_headers,
                method=request.method,
            )

            try:
                resp = self.tunnel._opener.open(req, timeout=self.tunnel.timeout)
            except urllib.error.HTTPError as e:
                err_content = e.read()
                return httpx.Response(
                    status_code=e.code,
                    headers=dict(e.headers),
                    content=err_content,
                )

            # Check for error status returned directly
            if resp.status >= 400:
                err_body = resp.read()
                return httpx.Response(
                    status_code=resp.status,
                    headers=dict(resp.headers),
                    content=err_body,
                )

            resp_ct = resp.headers.get("content-type", "").lower()
            if session and ("application/x-protobuf" in resp_ct or "application/octet-stream" in resp_ct):
                stream = _DecryptedSyncStream(self.tunnel, resp, session)
                out_headers = dict(resp.headers)
                req_accept = request.headers.get("accept", "").lower()
                is_sse = "text/event-stream" in req_accept or b'"stream": true' in raw_body or b'"stream":true' in raw_body
                if "application/x-protobuf" in resp_ct:
                    out_headers["content-type"] = "text/event-stream" if is_sse else "application/json"
                return httpx.Response(
                    status_code=resp.status,
                    headers=out_headers,
                    stream=stream,
                )
            else:
                return httpx.Response(
                    status_code=resp.status,
                    headers=dict(resp.headers),
                    content=resp.read(),
                )

    class AsyncConfidentialTransport(httpx.AsyncBaseTransport):
        """
        Custom async HTTPX transport delegating to the agnostic confidential tunnel.
        """

        def __init__(
            self,
            cvm_url: Optional[str] = None,
            ip: Optional[str] = None,
            port: int = 443,
            auth_key: Union[str, bytes] = "auth.pem",
            tunnel: Optional[ConfidentialTunnel] = None,
            **kwargs
        ):
            super().__init__()
            self._sync_transport = ConfidentialTransport(
                cvm_url=cvm_url,
                ip=ip,
                port=port,
                auth_key=auth_key,
                tunnel=tunnel,
                **kwargs
            )

        @property
        def base_url(self) -> str:
            return self._sync_transport.base_url

        @property
        def cvm_url(self) -> str:
            return self._sync_transport.cvm_url

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            import asyncio
            loop = asyncio.get_running_loop()
            # Offload synchronous socket handshake to thread pool
            sync_resp = await loop.run_in_executor(None, self._sync_transport.handle_request, request)

            if isinstance(sync_resp.stream, httpx.SyncByteStream):
                sync_stream = sync_resp.stream

                class _AsyncAdaptedStream(httpx.AsyncByteStream):
                    async def __aiter__(self) -> AsyncIterator[bytes]:
                        for chunk in sync_stream:
                            yield chunk

                    async def aclose(self) -> None:
                        await loop.run_in_executor(None, sync_stream.close)

                return httpx.Response(
                    status_code=sync_resp.status_code,
                    headers=sync_resp.headers,
                    stream=_AsyncAdaptedStream(),
                    request=request,
                )
            return sync_resp

else:
    class ConfidentialTransport:
        def __init__(self, *args, **kwargs):
            raise ImportError("httpx is required to use ConfidentialTransport. Install with: pip install httpx")

    class AsyncConfidentialTransport:
        def __init__(self, *args, **kwargs):
            raise ImportError("httpx is required to use AsyncConfidentialTransport. Install with: pip install httpx")


def create_http_client(
    cvm_url: Optional[str] = None,
    ip: Optional[str] = None,
    port: int = 443,
    auth_key: Union[str, bytes] = "auth.pem",
    **kwargs
) -> Any:
    """
    Creates an official HTTPX Client pre-configured with the ConfidentialTransport adapter.
    Can be passed directly into OpenAI(http_client=...) or used standalone.
    """
    if not HAS_HTTPX:
        raise ImportError("httpx is required for create_http_client. Install with: pip install httpx")
    transport = ConfidentialTransport(cvm_url=cvm_url, ip=ip, port=port, auth_key=auth_key, **kwargs)
    return httpx.Client(transport=transport)


def create_async_http_client(
    cvm_url: Optional[str] = None,
    ip: Optional[str] = None,
    port: int = 443,
    auth_key: Union[str, bytes] = "auth.pem",
    **kwargs
) -> Any:
    """
    Creates an official HTTPX AsyncClient pre-configured with AsyncConfidentialTransport.
    Can be passed directly into AsyncOpenAI(http_client=...).
    """
    if not HAS_HTTPX:
        raise ImportError("httpx is required for create_async_http_client. Install with: pip install httpx")
    transport = AsyncConfidentialTransport(cvm_url=cvm_url, ip=ip, port=port, auth_key=auth_key, **kwargs)
    return httpx.AsyncClient(transport=transport)

