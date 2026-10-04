"""
cevell_tunnel.cli — Command-Line Interface for Hardware-Attested CVM Tunnel
===========================================================================
Run interactive prompts, attestation audits, or launch the local confidential proxy:
  cevell-tunnel proxy --ip 136.114.201.214 --key auth.pem
"""

import sys
import json
import argparse
from typing import Optional

from .client import CVMClient
from .proxy import LocalCVMProxy


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "proxy":
        proxy_parser = argparse.ArgumentParser(
            prog="cevell-tunnel proxy",
            description="Start local confidential loopback proxy for standard LLM SDKs, curl, and tools",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        proxy_parser.add_argument(
            "--ip", "-i",
            required=True,
            help="Public IP address or hostname of the CVM",
        )
        proxy_parser.add_argument(
            "--key", "-k",
            default="auth.pem",
            help="Path to tenant Ed25519 private key (auth.pem / aws.pem)",
        )
        proxy_parser.add_argument(
            "--port", "-p",
            type=int,
            default=443,
            help="CVM HTTPS port",
        )
        proxy_parser.add_argument(
            "--listen-host",
            default="127.0.0.1",
            help="Loopback proxy listen host",
        )
        proxy_parser.add_argument(
            "--listen-port",
            type=int,
            default=8080,
            help="Loopback proxy listen port",
        )
        proxy_parser.add_argument(
            "--no-attestation",
            action="store_true",
            help="Bypass hardware attestation verification",
        )
        proxy_parser.add_argument(
            "--no-verify-code",
            action="store_true",
            help="Bypass guest OS/kernel code measurement verification against official releases",
        )
        proxy_parser.add_argument(
            "--release",
            default="latest",
            help="Expected official GitHub release tag for code measurement verification (default: latest)",
        )
        proxy_parser.add_argument(
            "--expected-rtmr1",
            default=None,
            help="Explicit expected RTMR1 measurement hash (overrides GitHub release lookup)",
        )
        args = proxy_parser.parse_args(sys.argv[2:])
        proxy = LocalCVMProxy(
            ip=args.ip,
            port=args.port,
            auth_key=args.key,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            verify_attestation=not args.no_attestation,
            verify_code=not args.no_verify_code,
            expected_release=args.release,
            expected_rtmr1=args.expected_rtmr1,
        )
        proxy.start()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "load":
        load_parser = argparse.ArgumentParser(
            prog="cevell-tunnel load",
            description="Dynamically load and provision a model on the confidential CVM",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        load_parser.add_argument(
            "--ip", "-i",
            required=True,
            help="Public IP address or hostname of the CVM",
        )
        load_parser.add_argument(
            "--key", "-k",
            default="auth.pem",
            help="Path to tenant Ed25519 private key (auth.pem / aws.pem)",
        )
        load_parser.add_argument(
            "--port", "-p",
            type=int,
            default=443,
            help="CVM HTTPS port",
        )
        load_parser.add_argument(
            "--model", "-m",
            required=True,
            help="Model repository name or path to load (e.g. meta-llama/Llama-3.1-8B-Instruct, unsloth/Llama-3.2-3B-Instruct)",
        )
        load_parser.add_argument(
            "--bitsandbytes",
            action="store_true",
            help="Enable BitsAndBytes quantization plugin (--quantization bitsandbytes --load-format bitsandbytes)",
        )
        load_parser.add_argument(
            "--gguf",
            action="store_true",
            help="Enable GGUF format plugin (--quantization gguf)",
        )
        load_parser.add_argument(
            "--quantization",
            default=None,
            help="Explicit quantization override (e.g. bitsandbytes, gguf, fp8, awq)",
        )
        load_parser.add_argument(
            "--timeout",
            type=float,
            default=600.0,
            help="Maximum timeout in seconds for downloading and loading model weights",
        )
        load_parser.add_argument(
            "--no-attestation",
            action="store_true",
            help="Bypass hardware attestation verification (e.g. for development CVMs)",
        )
        load_parser.add_argument(
            "--no-verify-code",
            action="store_true",
            help="Bypass guest OS/kernel code measurement verification against official releases",
        )
        load_parser.add_argument(
            "--release",
            default="latest",
            help="Expected official GitHub release tag for code measurement verification (default: latest)",
        )
        load_parser.add_argument(
            "--expected-rtmr1",
            default=None,
            help="Explicit expected RTMR1 measurement hash (overrides GitHub release lookup)",
        )
        args = load_parser.parse_args(sys.argv[2:])
        print(f"[*] Initializing connection to CVM at {args.ip}:{args.port}...")
        try:
            client = CVMClient(
                ip=args.ip,
                port=args.port,
                auth_key=args.key,
                verify_attestation=not args.no_attestation,
                verify_code=not args.no_verify_code,
                expected_release=args.release,
                expected_rtmr1=args.expected_rtmr1,
            )
        except Exception as e:
            print(f"[-] Initialization / Attestation failure: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"[*] Ordering model load for: {args.model}...")
        if args.bitsandbytes:
            print("[*] Activating BitsAndBytes 4-bit/8-bit quantization plugin...")
        if args.gguf:
            print("[*] Activating GGUF quantization format plugin...")
        try:
            res = client.load_model(
                model=args.model,
                bitsandbytes=args.bitsandbytes,
                gguf=args.gguf,
                quantization=args.quantization,
                timeout=args.timeout,
            )
            print(f"[+] Model loaded successfully:\n{json.dumps(res, indent=2)}")
        except Exception as e:
            print(f"[-] Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)
        return

    parser = argparse.ArgumentParser(
        description="Minimal Portable SDK CLI for Confidential CVM Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text to send to the confidential model",
    )
    parser.add_argument(
        "--ip", "-i",
        required=True,
        help="Public IP address or hostname of the CVM",
    )
    parser.add_argument(
        "--key", "-k",
        default="auth.pem",
        help="Path to tenant Ed25519 private key (auth.pem / aws.pem)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=443,
        help="CVM HTTPS port",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model repository name (default: auto-detected from CVM)",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        help="Dynamically provision and load model on the CVM before prompt execution or as standalone action",
    )
    parser.add_argument(
        "--bitsandbytes",
        action="store_true",
        help="Enable BitsAndBytes quantization plugin (--quantization bitsandbytes --load-format bitsandbytes)",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Enable GGUF format plugin (--quantization gguf)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Explicit quantization override (e.g. bitsandbytes, gguf, fp8, awq)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming token output",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit attestation and print verified hardware claims without sending prompt",
    )
    parser.add_argument(
        "--no-attestation",
        action="store_true",
        help="Bypass hardware attestation verification (e.g. for development or spot CVMs without quote devices)",
    )
    parser.add_argument(
        "--no-verify-code",
        action="store_true",
        help="Bypass guest OS/kernel code measurement verification against official releases",
    )
    parser.add_argument(
        "--release",
        default="latest",
        help="Expected official GitHub release tag for code measurement verification (default: latest)",
    )
    parser.add_argument(
        "--expected-rtmr1",
        default=None,
        help="Explicit expected RTMR1 measurement hash (overrides GitHub release lookup)",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Start local confidential loopback proxy server instead of executing a prompt",
    )
    parser.add_argument(
        "--listen-host",
        default="127.0.0.1",
        help="Loopback proxy listen host (when --proxy is specified)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=8080,
        help="Loopback proxy listen port (when --proxy is specified)",
    )

    args = parser.parse_args()

    if args.proxy:
        proxy = LocalCVMProxy(
            ip=args.ip,
            port=args.port,
            auth_key=args.key,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            verify_attestation=not args.no_attestation,
            verify_code=not args.no_verify_code,
            expected_release=args.release,
            expected_rtmr1=args.expected_rtmr1,
        )
        proxy.start()
        return

    print(f"[*] Initializing connection to CVM at {args.ip}:{args.port}...")
    try:
        client = CVMClient(
            ip=args.ip,
            port=args.port,
            auth_key=args.key,
            default_model=args.model,
            verify_attestation=not args.no_attestation,
            verify_code=not args.no_verify_code,
            expected_release=args.release,
            expected_rtmr1=args.expected_rtmr1,
        )
    except Exception as e:
        print(f"[-] Initialization / Attestation failure: {e}", file=sys.stderr)
        sys.exit(1)

    if client.attestation:
        att = client.attestation
        print("\n" + "=" * 60)
        print("  VERIFIED CONFIDENTIAL HARDWARE ATTESTATION")
        print("=" * 60)
        print(f"  Platform:         {att.platform.upper()}")
        print(f"  GPU Hardware:     {att.gpu_model or 'CPU Enclave'}")
        print(f"  GPU Architecture: {att.gpu_arch or 'N/A'}")
        if att.code_verified:
            print(f"  Code Integrity:   VERIFIED (Release {att.code_release})")
            if att.code_roothash:
                print(f"  dm-verity Root:   {att.code_roothash}")
            if att.rtmr1:
                print(f"  RTMR1 Digest:     {att.rtmr1[:32]}...")
        elif att.rtmr1:
            print(f"  RTMR1 Digest:     {att.rtmr1[:32]}... (unverified)")
        print(f"  TLS Fingerprint:  {att.tls_fingerprint[:32]}...")
        print(f"  HPKE Public Key:  {att.hpke_public_key_hex[:32]}...")
        print("  Root CA Trust:    Anchored to official Intel/NVIDIA Silicon Roots")
        print("=" * 60 + "\n")

    if args.verify_only:
        print("[+] Hardware attestation verification completed successfully.")
        sys.exit(0)

    target_load = args.load_model
    if target_load:
        print(f"[*] Ordering model load for: {target_load}...")
        if args.bitsandbytes:
            print("[*] Activating BitsAndBytes 4-bit/8-bit quantization plugin...")
        if args.gguf:
            print("[*] Activating GGUF quantization format plugin...")
        try:
            res = client.load_model(
                model=target_load,
                bitsandbytes=args.bitsandbytes,
                gguf=args.gguf,
                quantization=args.quantization,
            )
            print(f"[+] Model loaded successfully:\n{json.dumps(res, indent=2)}")
        except Exception as e:
            print(f"[-] Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)

        if not args.prompt:
            return

    if not args.prompt:
        print("[-] Please provide a prompt, specify --load-model, or use --verify-only", file=sys.stderr)
        sys.exit(1)

    resolved_model = args.load_model or args.model or client.get_active_model()
    print(f"[*] Dispatching encrypted prompt to {resolved_model}...")
    if args.no_stream:
        resp = client.chat_completion(
            messages=[{"role": "user", "content": args.prompt}],
            model=resolved_model,
            stream=False,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("\n" + content + "\n")
    else:
        for delta in client.stream(
            args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print("\n")


if __name__ == "__main__":
    main()
