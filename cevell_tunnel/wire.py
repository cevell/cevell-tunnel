"""
cevell.wire.v1 — Minimal Pure-Python Protobuf Wire Serialization
================================================================
Implements zero-dependency binary encoding and decoding for:
  - cevell.wire.v1.EncryptedInferenceRequest
  - cevell.wire.v1.StreamingInferenceFrame
Eliminates requirement for compiled protoc stubs in client SDK.
"""

import struct
from typing import Dict, Any, Tuple, Optional

CIPHER_SUITE_UNSPECIFIED = 0
CIPHER_SUITE_DHKEM_X25519_AES256_GCM = 1
CIPHER_SUITE_DHKEM_X25519_CHACHA20_POLY1305 = 2

FRAME_TYPE_UNSPECIFIED = 0
FRAME_TYPE_TOKEN_DELTA = 1
FRAME_TYPE_COMPLETION = 2
FRAME_TYPE_STREAM_END = 3
FRAME_TYPE_ERROR = 4
FRAME_TYPE_HEARTBEAT = 5


def encode_varint(val: int) -> bytes:
    """Encodes an integer into unsigned protobuf varint bytes."""
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decodes a protobuf varint starting at offset, returning (value, new_offset)."""
    res = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated varint in protobuf stream")
        b = data[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise ValueError("Varint shift overflow (shift >= 64)")
    return res, offset


def encode_field_bytes(field_num: int, val_bytes: bytes) -> bytes:
    """Encodes a length-delimited bytes field (wire type 2)."""
    if not val_bytes:
        return b""
    tag = (field_num << 3) | 2
    return encode_varint(tag) + encode_varint(len(val_bytes)) + val_bytes


def encode_field_varint(field_num: int, val: int) -> bytes:
    """Encodes a varint field (wire type 0). Omits default 0 per protobuf v3."""
    if val == 0:
        return b""
    tag = (field_num << 3) | 0
    return encode_varint(tag) + encode_varint(val)


def decode_protobuf(data: bytes) -> Dict[int, Any]:
    """Decodes raw protobuf bytes into a map of field_num -> value."""
    offset = 0
    fields: Dict[int, Any] = {}
    while offset < len(data):
        tag, offset = decode_varint(data, offset)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # Varint
            val, offset = decode_varint(data, offset)
            fields[field_num] = val
        elif wire_type == 2:  # Length-delimited
            length, offset = decode_varint(data, offset)
            if length < 0 or offset + length > len(data):
                raise ValueError("Length-delimited field truncated or negative length")
            val = data[offset : offset + length]
            offset += length
            fields[field_num] = val
        elif wire_type == 1:  # 64-bit fixed
            if offset + 8 > len(data):
                raise ValueError("64-bit fixed field truncated")
            val = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            fields[field_num] = val
        elif wire_type == 5:  # 32-bit fixed
            if offset + 4 > len(data):
                raise ValueError("32-bit fixed field truncated")
            val = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            fields[field_num] = val
        else:
            break
    return fields


def encode_encrypted_inference_request(
    version: int = 1,
    cipher_suite: int = CIPHER_SUITE_DHKEM_X25519_AES256_GCM,
    client_ephemeral_public_key: bytes = b"",
    nonce: bytes = b"",
    encrypted_payload: bytes = b"",
    attestation_binding_hash: Optional[bytes] = None,
    request_id: str = "",
    compression: int = 0,
) -> bytes:
    """
    Serializes cevell.wire.v1.EncryptedInferenceRequest into binary protobuf envelope.
    """
    parts = [
        encode_field_varint(1, version),
        encode_field_varint(2, cipher_suite),
        encode_field_bytes(3, client_ephemeral_public_key),
        encode_field_bytes(4, nonce),
        encode_field_bytes(5, encrypted_payload),
    ]
    if attestation_binding_hash:
        parts.append(encode_field_bytes(6, attestation_binding_hash))
    if request_id:
        parts.append(encode_field_bytes(7, request_id.encode("utf-8")))
    if compression:
        parts.append(encode_field_varint(8, compression))
    return b"".join(parts)


MAX_FRAME_SIZE = 64 * 1024 * 1024  # 64MB limit to prevent memory exhaustion


def decode_streaming_frame(frame_bytes: bytes) -> Tuple[int, int, bytes, bytes, int]:
    """
    Parses a 4-byte length-delimited StreamingInferenceFrame protobuf message.
    
    Wire format: [4B BE Length][StreamingInferenceFrame Protobuf]
    
    Returns:
        (sequence_number, frame_type, encrypted_payload, auth_tag, timestamp_ns)
    """
    if len(frame_bytes) < 4:
        raise ValueError("Frame buffer smaller than 4-byte length prefix")

    prefix_len = struct.unpack(">I", frame_bytes[:4])[0]
    if prefix_len > MAX_FRAME_SIZE:
        raise ValueError(f"Frame length {prefix_len} exceeds maximum allowed size ({MAX_FRAME_SIZE})")
    if len(frame_bytes) - 4 < prefix_len:
        raise ValueError(
            f"Incomplete frame: expected {prefix_len} bytes, got {len(frame_bytes) - 4}"
        )

    proto_data = frame_bytes[4 : 4 + prefix_len]
    fields = decode_protobuf(proto_data)

    seq = fields.get(1, 0)
    frame_type = fields.get(2, 0)
    payload = fields.get(3, b"")
    auth_tag = fields.get(4, b"")
    timestamp_ns = fields.get(5, 0)

    return seq, frame_type, payload, auth_tag, timestamp_ns
