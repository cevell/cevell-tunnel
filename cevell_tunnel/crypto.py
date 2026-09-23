"""
ox.crypto — Cryptographic Primitives for Latched Auth and RFC 9180 HPKE
======================================================================
Handles:
  - Ed25519 key loading (OpenSSH and PEM format)
  - Canonical request signing with body SHA-256 (Anti-replay + Header-first validation)
  - Ephemeral X25519 Diffie-Hellman Key Exchange (RFC 7748 / RFC 9180)
  - HKDF-SHA256 Symmetric Key Derivation
  - AES-256-GCM Envelope Encryption and Monotonic Frame Decryption with AAD
"""

import os
import time
import struct
import base64
import hashlib
import secrets
from typing import Dict, Tuple, Union, Optional

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# RFC 7748 Table 6: Non-contributory and low-order Curve25519 points
LOW_ORDER_X25519_BYTES = {
    b"\x00" * 32,
    b"\x01" + b"\x00" * 31,
    bytes.fromhex("e0eb7a7c3b41b8ae1656e3faf19fc46ada098dec9c32675fc00f10b1b17b44cc"),
    bytes.fromhex("5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157"),
    bytes.fromhex("ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    bytes.fromhex("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    bytes.fromhex("eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    bytes.fromhex("cdffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    bytes.fromhex("ceffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
    bytes.fromhex("cfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"),
}


def zeroize_buffer(buf: Union[bytearray, memoryview]) -> None:
    """Overwrites mutable buffer contents with zeros to scrub secrets from memory."""
    if isinstance(buf, (bytearray, memoryview)):
        for i in range(len(buf)):
            buf[i] = 0


class HPKEError(Exception):
    """Raised when HPKE key exchange, derivation, or decryption fails."""
    pass


class AuthKeyError(Exception):
    """Raised when the latched client Ed25519 private key cannot be parsed."""
    pass


def load_auth_key(key_input: Union[str, bytes, os.PathLike]) -> ed25519.Ed25519PrivateKey:
    """
    Loads an Ed25519 private key from a filesystem path (str or Path), PEM bytes/string,
    OpenSSH string, raw 32-byte seed, 64-byte keypair, or 64/128-character hexadecimal string/file.
    """
    data: bytes = b""
    if isinstance(key_input, (str, os.PathLike)):
        key_str = str(key_input)
        expanded = os.path.expanduser(key_str)
        if os.path.isfile(expanded):
            with open(expanded, "rb") as f:
                data = f.read().strip()
        else:
            clean = key_str.strip()
            data = clean.encode("utf-8")
    elif isinstance(key_input, bytes):
        data = key_input.strip()
    else:
        raise AuthKeyError(f"Unsupported key input type: {type(key_input)}")

    # 1. Try OpenSSH format
    try:
        key = serialization.load_ssh_private_key(data, password=None)
        if isinstance(key, ed25519.Ed25519PrivateKey):
            return key
    except Exception:
        pass

    # 2. Try standard PEM format (PKCS#8)
    try:
        key = serialization.load_pem_private_key(data, password=None)
        if isinstance(key, ed25519.Ed25519PrivateKey):
            return key
    except Exception:
        pass

    # 3. Try hex decoding (handles 64-char hex seeds and 128-char hex keypairs in files or strings)
    clean_str = ""
    try:
        clean_str = data.decode("ascii").strip()
    except (UnicodeDecodeError, AttributeError):
        pass

    if len(clean_str) == 64:
        try:
            raw_seed = bytes.fromhex(clean_str)
            if len(raw_seed) == 32:
                return ed25519.Ed25519PrivateKey.from_private_bytes(raw_seed)
        except Exception:
            pass
    elif len(clean_str) == 128:
        try:
            raw_keypair = bytes.fromhex(clean_str)
            if len(raw_keypair) == 64:
                # Libsodium 64-byte secret key: first 32 bytes are private seed
                return ed25519.Ed25519PrivateKey.from_private_bytes(raw_keypair[:32])
        except Exception:
            pass

    # 4. Try raw 32-byte binary seed
    if len(data) == 32:
        try:
            return ed25519.Ed25519PrivateKey.from_private_bytes(data)
        except Exception:
            pass

    # 5. Try raw 64-byte binary keypair (libsodium format: seed || pubkey)
    if len(data) == 64:
        try:
            return ed25519.Ed25519PrivateKey.from_private_bytes(data[:32])
        except Exception:
            pass

    # 6. Try base64 encoded 32-byte seed
    if len(clean_str) in (43, 44):
        try:
            b64_decoded = base64.b64decode(clean_str)
            if len(b64_decoded) == 32:
                return ed25519.Ed25519PrivateKey.from_private_bytes(b64_decoded)
        except Exception:
            pass

    raise AuthKeyError(
        "Could not load Ed25519 private key. Ensure key is valid OpenSSH, PEM, "
        "raw 32-byte seed, or 64-character hexadecimal format."
    )


def sign_canonical_request(
    priv_key: ed25519.Ed25519PrivateKey,
    method: str,
    path: str,
    body_bytes: bytes,
    request_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
) -> Dict[str, str]:
    """
    Computes canonical Ed25519 signature over HTTP headers and body hash.
    Returns the required header dictionary matching Cevell CVM specifications.
    """
    req_id = request_id or f"req-{secrets.token_hex(8)}"
    ts = timestamp or str(int(time.time()))
    n = nonce or secrets.token_hex(16)
    body_hash_hex = hashlib.sha256(body_bytes).hexdigest()

    canonical_str = f"{method.upper()}:{path}:{req_id}:{ts}:{n}:{body_hash_hex}"
    signature_bytes = priv_key.sign(canonical_str.encode("utf-8"))
    sig_b64 = base64.b64encode(signature_bytes).decode("ascii")

    return {
        "Authorization": f"Cevell-Ed25519 signature={sig_b64}, timestamp={ts}, nonce={n}, request_id={req_id}, body_sha256={body_hash_hex}",
        "x-signature": sig_b64,
        "x-timestamp": ts,
        "x-nonce": n,
        "x-request-id": req_id,
        "x-cevell-body-sha256": body_hash_hex,
    }


def compute_frame_nonce(base_iv: bytes, seq: int) -> bytes:
    """Derives unique 12-byte AEAD nonce by XORing the 4-byte sequence number into the base IV."""
    seq_bytes = struct.pack(">I", seq)
    iv = bytearray(base_iv)
    iv[8] ^= seq_bytes[0]
    iv[9] ^= seq_bytes[1]
    iv[10] ^= seq_bytes[2]
    iv[11] ^= seq_bytes[3]
    return bytes(iv)


def compute_frame_aad(seq: int, frame_type: int, timestamp_ns: int) -> bytes:
    """
    Constructs 13-byte Additional Authenticated Data (AAD) for a StreamingInferenceFrame:
    [seq (4B BE)][frame_type (1B)][timestamp_ns (8B BE)]
    """
    return struct.pack(">IBq", seq, frame_type, timestamp_ns)


class HPKESession:
    """Manages symmetric request encryption and response frame decryption."""

    def __init__(
        self,
        client_pub_bytes: bytes,
        req_key: bytes,
        resp_key: bytes,
        resp_base_iv: bytes,
    ):
        self.client_pub_bytes = client_pub_bytes
        self._req_key = bytearray(req_key)
        self._resp_key = bytearray(resp_key)
        self._resp_base_iv = bytearray(resp_base_iv)
        self.req_key = bytes(self._req_key)
        self.resp_key = bytes(self._resp_key)
        self.resp_base_iv = bytes(self._resp_base_iv)
        self._aead_req: Optional[AESGCM] = AESGCM(self.req_key)
        self._aead_resp: Optional[AESGCM] = AESGCM(self.resp_key)
        self.expected_seq: int = 0

    def encrypt_request(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Encrypts request plaintext using AES-256-GCM.
        Returns: (12-byte nonce, ciphertext_with_tag)
        """
        if self._aead_req is None:
            raise HPKEError("HPKESession has already been zeroized")
        nonce = secrets.token_bytes(12)
        ciphertext = self._aead_req.encrypt(nonce, plaintext, associated_data=associated_data)
        return nonce, ciphertext

    def decrypt_frame(
        self,
        seq: int,
        frame_type: int,
        ciphertext: bytes,
        auth_tag: bytes,
        timestamp_ns: int,
    ) -> bytes:
        """
        Decrypts an authenticated streaming frame using sequence number and AAD.
        Enforces strictly monotonic frame sequence numbers to prevent replay and reordering.
        """
        if self._aead_resp is None:
            raise HPKEError("HPKESession has already been zeroized")
        if seq != self.expected_seq:
            raise HPKEError(
                f"Out-of-order frame sequence: expected {self.expected_seq}, got {seq}"
            )
        nonce = compute_frame_nonce(bytes(self._resp_base_iv), seq)
        aad = compute_frame_aad(seq, frame_type, timestamp_ns)
        ciphertext_with_tag = ciphertext + auth_tag
        try:
            decrypted = self._aead_resp.decrypt(nonce, ciphertext_with_tag, aad)
            self.expected_seq += 1
            return decrypted
        except Exception as e:
            raise HPKEError(f"AEAD authentication failed on frame seq={seq}: {e}")

    def zeroize(self) -> None:
        """Wipes keys and cryptographic context securely from memory."""
        zeroize_buffer(self._req_key)
        zeroize_buffer(self._resp_key)
        zeroize_buffer(self._resp_base_iv)
        self.req_key = b"\x00" * len(self.req_key)
        self.resp_key = b"\x00" * len(self.resp_key)
        self.resp_base_iv = b"\x00" * len(self.resp_base_iv)
        self._aead_req = None
        self._aead_resp = None
        self.expected_seq = 0


def create_client_hpke_session(server_pub_bytes: bytes) -> Tuple[HPKESession, bytes]:
    """
    Performs RFC 9180 DHKEM(X25519, HKDF-SHA256) key agreement with the CVM public key.
    Returns: (session, client_ephemeral_public_key_bytes)
    """
    if len(server_pub_bytes) != 32:
        raise HPKEError(f"Recipient public key must be exactly 32 bytes (got {len(server_pub_bytes)})")

    if server_pub_bytes in LOW_ORDER_X25519_BYTES:
        raise HPKEError("Recipient public key is a low-order Curve25519 point (RFC 7748)")

    server_pub = x25519.X25519PublicKey.from_public_bytes(server_pub_bytes)
    client_priv = x25519.X25519PrivateKey.generate()
    client_pub_bytes = client_priv.public_key().public_bytes_raw()

    raw_shared = client_priv.exchange(server_pub)
    if raw_shared == b"\x00" * 32:
        raise HPKEError("Computed shared secret is all-zero (low-order attack)")

    shared_secret_buf = bytearray(raw_shared)
    try:
        def hkdf_expand(info: bytes, length: int) -> bytes:
            return HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=None,
                info=info,
            ).derive(bytes(shared_secret_buf))

        req_key = hkdf_expand(b"cevell-hpke-req-aes-gcm", 32)
        resp_key = hkdf_expand(b"cevell-hpke-resp-aes-gcm", 32)
        resp_base_iv = hkdf_expand(b"cevell-hpke-resp-base-iv", 12)

        session = HPKESession(
            client_pub_bytes=client_pub_bytes,
            req_key=req_key,
            resp_key=resp_key,
            resp_base_iv=resp_base_iv,
        )
        return session, client_pub_bytes
    finally:
        zeroize_buffer(shared_secret_buf)
