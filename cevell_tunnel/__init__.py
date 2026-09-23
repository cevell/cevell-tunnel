"""
cevell_tunnel — Agnostic Translation Layer & Transport Tunnel for Hardware-Attested CVMs
========================================================================================
"""

from .client import CVMClient, CVMInferenceError
from .attestation import verify_attestation_document, AttestationResult, AttestationVerificationError
from .crypto import load_auth_key, HPKESession, HPKEError, AuthKeyError
from .tunnel import ConfidentialTunnel, TunnelError
from .transport import (
    ConfidentialTransport,
    AsyncConfidentialTransport,
    create_http_client,
    create_async_http_client,
)
from .proxy import LocalCVMProxy

# Authoritative aliases
Tunnel = ConfidentialTunnel
Transport = ConfidentialTransport
AsyncTransport = AsyncConfidentialTransport
Client = CVMClient
Proxy = LocalCVMProxy

__all__ = [
    "Tunnel",
    "Transport",
    "AsyncTransport",
    "Client",
    "Proxy",
    "ConfidentialTunnel",
    "TunnelError",
    "ConfidentialTransport",
    "AsyncConfidentialTransport",
    "create_http_client",
    "create_async_http_client",
    "LocalCVMProxy",
    "CVMClient",
    "CVMInferenceError",
    "verify_attestation_document",
    "AttestationResult",
    "AttestationVerificationError",
    "load_auth_key",
    "HPKESession",
    "HPKEError",
    "AuthKeyError",
]

__version__ = "0.2.0"
