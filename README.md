# Cevell Tunnel (`cevell-tunnel`)

[![PyPI version](https://img.shields.io/pypi/v/cevell-tunnel.svg)](https://pypi.org/project/cevell-tunnel/)
[![Python versions](https://img.shields.io/pypi/pyversions/cevell-tunnel.svg)](https://pypi.org/project/cevell-tunnel/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Agnostic Confidential Translation Layer, HTTP Transport, and Proxy for Hardware-Attested CVMs**

`cevell-tunnel` bridges standard unencrypted HTTP/REST client requests to hardware-attested, end-to-end encrypted Confidential Virtual Machines (CVMs) running on Cevell OS.

It translates arbitrary plaintext requests into silicon-attested, RFC 9180 HPKE-encrypted Protobuf envelopes (`cevell.wire.v1`), and decrypts streaming response frames (e.g. Server-Sent Events / SSE) back to plaintext in real time.

---

## Features

- **Hardware Attestation Verification**: Verifies hardware quotes (Intel TDX, NVIDIA Hopper DICE) against silicon vendor roots of trust.
- **RFC 9180 HPKE Payload Encryption**: End-to-end encryption (DHKEM(X25519, HKDF-SHA256) + HKDF-SHA256 + AES-GCM-256) straight to protected enclave memory.
- **TLS SPKI Channel Binding**: Cryptographically binds the HPKE session to the in-transit TLS channel to prevent Man-in-the-Middle (MITM) and relay attacks.
- **Canonical Ed25519 Request Authentication**: Deterministically signs HTTP request headers and bodies with replay protection.
- **Zero-Trust Drop-in Transport**: Plugs directly into the official `openai` Python SDK or any `httpx` client.
- **Local Proxy Daemon**: Exposes a local HTTP loopback endpoint (`127.0.0.1:8080`) allowing `curl`, Cursor, Continue.dev, or any language client to transparently query confidential CVMs.

---

## Installation

```bash
# Core package (cryptography only, zero unnecessary bloat)
pip install cevell-tunnel

# With HTTPX transport adapter (for use with OpenAI Python SDK)
pip install "cevell-tunnel[transport]"
```

---

## Quickstart

### 1. Official OpenAI Python SDK (Sync)

Plug `ConfidentialTransport` directly into `openai.OpenAI`:

```python
from openai import OpenAI
import httpx
from cevell_tunnel import ConfidentialTransport

# Initialize secure hardware transport
transport = ConfidentialTransport(
    ip="136.114.201.214",           # Your CVM's public IP
    auth_key="~/.cevell/tenant.pem", # Path to Ed25519 private key
    verify_attestation=True,        # Validates Intel TDX / Hopper DICE quotes
)

client = OpenAI(
    base_url=transport.base_url,
    api_key="EMPTY",                # Authentication is handled cryptographically via Ed25519
    http_client=httpx.Client(transport=transport),
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain zero-trust computing."}],
)
print(response.choices[0].message.content)
```

### 2. Official OpenAI Python SDK (Async & Streaming)

```python
import asyncio
from openai import AsyncOpenAI
import httpx
from cevell_tunnel import AsyncConfidentialTransport

async def main():
    transport = AsyncConfidentialTransport(
        ip="136.114.201.214",
        auth_key="~/.cevell/tenant.pem",
        verify_attestation=True,
    )

    client = AsyncOpenAI(
        base_url=transport.base_url,
        api_key="EMPTY",
        http_client=httpx.AsyncClient(transport=transport),
    )

    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Write a short haiku about enclaves."}],
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Local Loopback Proxy (CLI)

Run the local proxy daemon to tunnel any HTTP tool (e.g. `curl`, IDE extensions):

```bash
cevell-tunnel proxy \
  --ip 136.114.201.214 \
  --key ~/.cevell/tenant.pem \
  --port 8080 \
  --verify-attestation
```

Then query the local proxy using standard `curl`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello confidential AI!"}]
  }'
```

---

## Security Model

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      YOUR STANDARD APPLICATION / SDK                     │
│                                                                          │
│   • Official OpenAI Python SDK (`openai.OpenAI`)                         │
│   • Official OpenAI Async Python SDK (`openai.AsyncOpenAI`)              │
│   • LangChain, LlamaIndex, LiteLLM, curl, or Cursor                      │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ Plaintext HTTP / SSE
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  CEVELL TUNNEL (TRANSLATION LAYER)                       │
│                                                                          │
│   1. Silicon Hardware Quote Verification (Intel TDX, NVIDIA Hopper DICE) │
│   2. In-Transit TLS SPKI Channel Binding (Anti-MITM / Anti-Relay)        │
│   3. Canonical Ed25519 Request Signing (`Authorization: Cevell-Ed25519`) │
│   4. RFC 9180 HPKE Payload Encryption (X25519 + HKDF-SHA256 + AES-GCM)   │
│   5. Outer Wire Protocol Framing (`cevell.wire.v1.EncryptedInference...`)|
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ Protobuf HPKE Wire over HTTPS
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        CONFIDENTIAL CVM (ENCLAVE)                        │
│                                                                          │
│   1. cevell-node validates Ed25519 signature & channel binding           │
│   2. Hardware enclave decrypts HPKE ciphertext directly in protected RAM │
│   3. Forwarded to internal inference engine (vLLM / llama-server)        │
│   4. Streaming SSE tokens encrypted into monotonic length-prefixed frames│
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Repository & Links

- **Source Code**: [https://github.com/cevell/cevell-tunnel](https://github.com/cevell/cevell-tunnel)
- **Bug Tracker**: [https://github.com/cevell/cevell-tunnel/issues](https://github.com/cevell/cevell-tunnel/issues)
- **Website**: [https://cevell.com](https://cevell.com)

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
