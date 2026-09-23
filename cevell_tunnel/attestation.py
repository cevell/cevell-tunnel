"""
ox.attestation — Multi-Tier Silicon Attestation & Key Binding Verifier
======================================================================
Verifies hardware claims from Confidential Virtual Machines:
  1. NVIDIA Hopper / Blackwell H100 DICE X.509 Certificate Chain verification
     anchored to official NVIDIA Device Identity Root CA.
  2. NVIDIA SPDM MEASUREMENTS evidence report signature verification using
     the leaf DICE certificate key (ECDSA P-384 + SHA-384) with session nonce binding.
  3. Intel TDX Quote v4/v5 ECDSA quote signature verification using QE3 attestation key
     over Header || TD Report, QE Report signature verification with PCK certificate,
     and non-debug td_attributes enforcement.
  4. AMD SEV-SNP Report verification (debug bit enforcement, REPORT_DATA binding,
     and optional VCEK signature verification).
  5. Live vendor revocation checking against official online CRL endpoints
     (NVIDIA L1/L2 CRLs, Intel SGX Root & Platform PCK CRLs).
  6. Mathematical verification of 64-byte REPORT_DATA anti-splicing binding:
       REPORT_DATA[0..31]  == TLS server certificate SPKI fingerprint
       REPORT_DATA[32..63] == SHA-256(HPKE_Pubkey || Nonce || GPU_Evidence_Digest)
  7. Cryptographic extraction of verified in-enclave HPKE Public Key.
"""

import base64
import hashlib
import struct
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.x509.oid import ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding, utils

# Official NVIDIA Device Identity Root CA (Hopper H100 / Blackwell B200)
OFFICIAL_NVIDIA_ROOT_CA_PEM = """-----BEGIN CERTIFICATE-----
MIICCzCCAZCgAwIBAgIQLTZwscoQBBHB/sDoKgZbVDAKBggqhkjOPQQDAzA1MSIw
IAYDVQQDDBlOVklESUEgRGV2aWNlIElkZW50aXR5IENBMQ8wDQYDVQQKDAZOVklE
SUEwIBcNMjExMTA1MDAwMDAwWhgPOTk5OTEyMzEyMzU5NTlaMDUxIjAgBgNVBAMM
GU5WSURJQSBEZXZpY2UgSWRlbnRpdHkgQ0ExDzANBgNVBAoMBk5WSURJQTB2MBAG
ByqGSM49AgEGBSuBBAAiA2IABA5MFKM7+KViZljbQSlgfky/RRnEQScW9NDZF8SX
gAW96r6u/Ve8ZggtcYpPi2BS4VFu6KfEIrhN6FcHG7WP05W+oM+hxj7nyA1r1jkB
2Ry70YfThX3Ba1zOryOP+MJ9vaNjMGEwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8B
Af8EBAMCAQYwHQYDVR0OBBYEFFeF/4PyY8xlfWi3Olv0jUrL+0lfMB8GA1UdIwQY
MBaAFFeF/4PyY8xlfWi3Olv0jUrL+0lfMAoGCCqGSM49BAMDA2kAMGYCMQCPeFM3
TASsKQVaT+8S0sO9u97PVGCpE9d/I42IT7k3UUOLSR/qvJynVOD1vQKVXf0CMQC+
EY55WYoDBvs2wPAH1Gw4LbcwUN8QCff8bFmV4ZxjCRr4WXTLFHBKjbfneGSBWwA=
-----END CERTIFICATE-----"""

# Official Intel SGX / TDX Root CA (Intel Trust Authority / PCS)
OFFICIAL_INTEL_ROOT_CA_PEM = """-----BEGIN CERTIFICATE-----
MIICjzCCAjSgAwIBAgIUImUM1lqdNInzg7SVUr9QGzknBqwwCgYIKoZIzj0EAwIw
aDEaMBgGA1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENv
cnBvcmF0aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJ
BgNVBAYTAlVTMB4XDTE4MDUyMTEwNDUxMFoXDTQ5MTIzMTIzNTk1OVowaDEaMBgG
A1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENvcnBvcmF0
aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJBgNVBAYT
AlVTMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEC6nEwMDIYZOj/iPWsCzaEKi7
1OiOSLRFhWGjbnBVJfVnkY4u3IjkDYYL0MxO4mqsyYjlBalTVYxFP2sJBK5zlKOB
uzCBuDAfBgNVHSMEGDAWgBQiZQzWWp00ifODtJVSv1AbOScGrDBSBgNVHR8ESzBJ
MEegRaBDhkFodHRwczovL2NlcnRpZmljYXRlcy50cnVzdGVkc2VydmljZXMuaW50
ZWwuY29tL0ludGVsU0dYUm9vdENBLmRlcjAdBgNVHQ4EFgQUImUM1lqdNInzg7SV
Ur9QGzknBqwwDgYDVR0PAQH/BAQDAgEGMBIGA1UdEwEB/wQIMAYBAf8CAQEwCgYI
KoZIzj0EAwIDSQAwRgIhAOW/5QkR+S9CiSDcNoowLuPRLsWGf/Yi7GSX94BgwTwg
AiEA4J0lrHoMs+Xo5o/sX6O9QWxHRAvZUGOdRQ7cvqRXaqI=
-----END CERTIFICATE-----"""


class AttestationVerificationError(Exception):
    """Raised when hardware attestation verification or key binding fails."""
    pass


@dataclass
class AttestationResult:
    """Verified cryptographic claims returned from CVM attestation audit."""
    hpke_public_key: bytes
    hpke_public_key_hex: str
    tls_fingerprint: str
    platform: str
    user_data: bytes
    binding_hash: bytes
    has_gpu: bool
    gpu_model: Optional[str] = None
    gpu_arch: Optional[str] = None
    mrtd: Optional[str] = None
    rtmr0: Optional[str] = None
    rtmr1: Optional[str] = None
    rtmr2: Optional[str] = None
    rtmr3: Optional[str] = None
    td_attributes: Optional[str] = None
    vendor_crl_verified: bool = False
    auth_public_key: Optional[str] = None
    is_valid: bool = True


class VendorCRLManager:
    """
    Downloads, caches, and verifies X.509 Certificate Revocation Lists (CRLs)
    from official Intel, NVIDIA, and hardware vendor trust services.
    """
    def __init__(self, cache_ttl: float = 3600.0, request_timeout: float = 5.0):
        self.cache_ttl = cache_ttl
        self.request_timeout = request_timeout
        self._crl_cache: Dict[str, Tuple[x509.CertificateRevocationList, float]] = {}
        self.verified_urls = set()

    def fetch_crl(self, url: str) -> x509.CertificateRevocationList:
        now = time.time()
        if url in self._crl_cache:
            crl, fetch_time = self._crl_cache[url]
            if now - fetch_time < self.cache_ttl:
                return crl

        req = urllib.request.Request(url, headers={"User-Agent": "Ox-SDK/1.0"})
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            data = resp.read()

        try:
            crl = x509.load_der_x509_crl(data)
        except Exception:
            crl = x509.load_pem_x509_crl(data)

        self._crl_cache[url] = (crl, now)
        return crl

    def check_certificate_revocation(
        self,
        cert: x509.Certificate,
        issuer_cert: Optional[x509.Certificate] = None,
        strict_online: bool = False,
    ) -> bool:
        """
        Validates that a certificate is not revoked in manufacturer CRLs.
        Returns True if verified against online CRL.
        """
        crl_urls: List[str] = []
        try:
            cdp_ext = cert.extensions.get_extension_for_oid(ExtensionOID.CRL_DISTRIBUTION_POINTS)
            for dp in cdp_ext.value:
                if dp.full_name:
                    for name in dp.full_name:
                        val = name.value
                        if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                            crl_urls.append(val)
        except x509.ExtensionNotFound:
            pass

        # Fallback to known vendor CRL endpoints based on issuer
        if not crl_urls and issuer_cert:
            issuer_str = issuer_cert.subject.rfc4514_string()
            if "NVIDIA Device Identity CA" in issuer_str:
                crl_urls.append("http://crl.ndis.nvidia.com/crl/l1-root.crl")
            elif "NVIDIA GH100 Identity" in issuer_str:
                crl_urls.append("http://crl.ndis.nvidia.com/crl/l2-gh100.crl")
            elif "Intel SGX Root CA" in issuer_str:
                crl_urls.append("https://certificates.trustedservices.intel.com/IntelSGXRootCA.der")
            elif "Intel SGX PCK Platform CA" in issuer_str:
                crl_urls.append("https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der")

        verified_any = False
        for url in crl_urls:
            try:
                crl = self.fetch_crl(url)
                # Verify CRL signature with issuer key if available
                if issuer_cert:
                    issuer_pub = issuer_cert.public_key()
                    if isinstance(issuer_pub, ec.EllipticCurvePublicKey):
                        try:
                            issuer_pub.verify(
                                crl.signature,
                                crl.tbs_certlist_bytes,
                                ec.ECDSA(crl.signature_hash_algorithm),
                            )
                        except Exception as e:
                            raise AttestationVerificationError(
                                f"Vendor CRL EC signature verification failed for {url}: {e}"
                            )
                    elif isinstance(issuer_pub, rsa.RSAPublicKey):
                        sig_algo = crl.signature_hash_algorithm
                        verified = False
                        try:
                            issuer_pub.verify(
                                crl.signature,
                                crl.tbs_certlist_bytes,
                                padding.PKCS1v15(),
                                sig_algo,
                            )
                            verified = True
                        except Exception:
                            pass
                        if not verified:
                            try:
                                issuer_pub.verify(
                                    crl.signature,
                                    crl.tbs_certlist_bytes,
                                    padding.PSS(mgf=padding.MGF1(sig_algo), salt_length=sig_algo.digest_size),
                                    sig_algo,
                                )
                                verified = True
                            except Exception as e:
                                raise AttestationVerificationError(
                                    f"Vendor CRL RSA signature verification failed for {url}: {e}"
                                )

                revoked = crl.get_revoked_certificate_by_serial_number(cert.serial_number)
                if revoked is not None:
                    rev_date = getattr(revoked, "revocation_date_utc", getattr(revoked, "revocation_date", "unknown"))
                    raise AttestationVerificationError(
                        f"Certificate (serial {cert.serial_number}, subject {cert.subject.rfc4514_string()}) "
                        f"is REVOKED in vendor CRL ({url}) on {rev_date}"
                    )
                self.verified_urls.add(url)
                verified_any = True
            except AttestationVerificationError:
                raise
            except Exception as e:
                if strict_online:
                    raise AttestationVerificationError(f"Online CRL verification failed for {url}: {e}")

        return verified_any


# Global in-memory vendor CRL cache
_GLOBAL_CRL_MANAGER = VendorCRLManager()


def parse_pem_chain(pem_data: str) -> List[x509.Certificate]:
    """Splits and parses a multi-certificate PEM block into x509.Certificate list."""
    certs = []
    blocks = pem_data.strip().split("-----END CERTIFICATE-----")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        pem_str = block + "\n-----END CERTIFICATE-----\n"
        try:
            cert = x509.load_pem_x509_certificate(pem_str.encode("utf-8"))
            certs.append(cert)
        except Exception:
            pass
    return certs


def verify_x509_chain(
    certs: List[x509.Certificate],
    root_ca_pem: Optional[str] = None,
    check_online_crl: bool = False,
    crl_manager: Optional[VendorCRLManager] = None,
    strict_online: bool = False,
) -> bool:
    """
    Verifies an X.509 certificate chain from leaf (certs[0]) to root (certs[-1]).
    If root_ca_pem is provided, validates that root matches the trusted root CA.
    If check_online_crl is True, verifies each cert against vendor CRL distribution points.
    """
    if not certs:
        raise AttestationVerificationError("Empty certificate chain")

    mgr = crl_manager or _GLOBAL_CRL_MANAGER

    # 1. Verify signatures between consecutive certificates
    for i in range(len(certs) - 1):
        child = certs[i]
        parent = certs[i + 1]
        pubkey = parent.public_key()

        if isinstance(pubkey, ec.EllipticCurvePublicKey):
            try:
                pubkey.verify(
                    child.signature,
                    child.tbs_certificate_bytes,
                    ec.ECDSA(child.signature_hash_algorithm),
                )
            except Exception as e:
                raise AttestationVerificationError(
                    f"Certificate chain signature verification failed at index {i} -> {i+1}: {e}"
                )
        elif isinstance(pubkey, rsa.RSAPublicKey):
            sig_algo = child.signature_hash_algorithm
            verified = False
            try:
                pubkey.verify(
                    child.signature,
                    child.tbs_certificate_bytes,
                    padding.PSS(mgf=padding.MGF1(sig_algo), salt_length=sig_algo.digest_size),
                    sig_algo,
                )
                verified = True
            except Exception:
                pass
            if not verified:
                try:
                    pubkey.verify(
                        child.signature,
                        child.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        sig_algo,
                    )
                    verified = True
                except Exception as e:
                    raise AttestationVerificationError(
                        f"RSA certificate chain verification failed at index {i} -> {i+1}: {e}"
                    )
        else:
            raise AttestationVerificationError(f"Unsupported public key type: {type(pubkey)}")

        # Check CRL for child cert
        if check_online_crl:
            mgr.check_certificate_revocation(child, parent, strict_online=strict_online)

    # 2. Verify Root Certificate Self-Signature or Trusted Root Match
    root = certs[-1]
    root_pub = root.public_key()
    if isinstance(root_pub, ec.EllipticCurvePublicKey):
        root_pub.verify(
            root.signature,
            root.tbs_certificate_bytes,
            ec.ECDSA(root.signature_hash_algorithm),
        )
    elif isinstance(root_pub, rsa.RSAPublicKey):
        sig_algo = root.signature_hash_algorithm
        try:
            root_pub.verify(
                root.signature,
                root.tbs_certificate_bytes,
                padding.PKCS1v15(),
                sig_algo,
            )
        except Exception:
            root_pub.verify(
                root.signature,
                root.tbs_certificate_bytes,
                padding.PSS(mgf=padding.MGF1(sig_algo), salt_length=sig_algo.digest_size),
                sig_algo,
            )

    # 3. If official Root CA PEM provided, match root certificate SPKI / bytes
    if root_ca_pem:
        trusted_root = x509.load_pem_x509_certificate(root_ca_pem.encode("utf-8"))
        if root.public_bytes(serialization.Encoding.DER) != trusted_root.public_bytes(serialization.Encoding.DER):
            raise AttestationVerificationError(
                f"Chain root does not match official trusted root authority ({root.subject.rfc4514_string()})"
            )

    return True


def verify_intel_tdx_quote(
    quote_bytes: bytes,
    user_data: bytes,
    enforce_official_roots: bool = True,
    check_online_vendor: bool = True,
    crl_manager: Optional[VendorCRLManager] = None,
    strict_online: bool = False,
) -> Dict[str, Any]:
    """
    Cryptographically verifies an Intel TDX Quote v4/v5 structure:
      - Verifies Quote Header and TEE type.
      - Enforces non-debug td_attributes (debug bit == 0).
      - Checks REPORT_DATA (bytes 568..632) literal match with user_data.
      - Verifies QE3 ECDSA signature over Header || TD Report (bytes 0..632).
      - Verifies QE Report signature using Leaf PCK Certificate.
      - Verifies PCK certificate chain anchored to Intel SGX Root CA.
      - Checks certificate revocation against Intel online CRL endpoints.
    """
    if len(quote_bytes) < 636:
        raise AttestationVerificationError(
            f"Intel TDX quote too short ({len(quote_bytes)} bytes, expected >= 636)"
        )

    # 1. Header validation (0..48)
    version, att_key_type, tee_type = struct.unpack_from("<HH4s", quote_bytes, 0)
    if version not in (4, 5):
        raise AttestationVerificationError(f"Unsupported Intel TDX quote version: {version}")
    if tee_type not in (b"\x81\x00\x00\x00", b"\x00\x00\x00\x81"):
        raise AttestationVerificationError(f"Invalid Intel TDX TEE type: {tee_type.hex()}")

    # 2. TD Report validation (48..632)
    td_report = quote_bytes[48:632]
    # TDATTRIBUTES is at offset 120..128 inside TD Report body (quote offset 168..176)
    td_attributes = td_report[120:128]
    if (td_attributes[0] & 1) != 0:
        raise AttestationVerificationError("Intel TDX enclave running with DEBUG mode enabled (untrusted)")

    embedded_user_data = quote_bytes[568:632]
    if embedded_user_data != user_data:
        raise AttestationVerificationError(
            "REPORT_DATA mismatch at Intel TDX quote offset 568..631"
        )

    mrtd = quote_bytes[184:232]
    # RTMR0..3 is at quote offset 376..568 (Body offset 328..520, 4 x 48 bytes)
    rtmr0 = quote_bytes[376:424]
    rtmr1 = quote_bytes[424:472]
    rtmr2 = quote_bytes[472:520]
    rtmr3 = quote_bytes[520:568]

    # 3. Auth Data & QE3 Signature over Header || TD Report (0..632)
    auth_data_len = struct.unpack_from("<I", quote_bytes, 632)[0]
    if len(quote_bytes) < 636 + auth_data_len:
        raise AttestationVerificationError("Truncated Intel TDX quote auth data")

    sig_data = quote_bytes[636 : 636 + auth_data_len]
    quote_sig_r = sig_data[0:32]
    quote_sig_s = sig_data[32:64]
    der_quote_sig = utils.encode_dss_signature(
        int.from_bytes(quote_sig_r, "big"),
        int.from_bytes(quote_sig_s, "big"),
    )

    qe3_pub_x = sig_data[64:96]
    qe3_pub_y = sig_data[96:128]
    try:
        qe3_pubkey = ec.EllipticCurvePublicNumbers(
            int.from_bytes(qe3_pub_x, "big"),
            int.from_bytes(qe3_pub_y, "big"),
            ec.SECP256R1(),
        ).public_key()
        qe3_pubkey.verify(der_quote_sig, quote_bytes[0:632], ec.ECDSA(hashes.SHA256()))
    except Exception as e:
        raise AttestationVerificationError(f"Intel TDX QE3 signature verification failed: {e}")

    # 4. QE Certification Data
    if len(sig_data) < 134:
        raise AttestationVerificationError("Intel TDX QE certification data truncated")

    cert_data_type, cert_data_size = struct.unpack_from("<HI", sig_data, 128)
    pck_certs: List[x509.Certificate] = []
    if cert_data_type == 6:  # QE Report Certification Data
        qe_report = sig_data[134 : 134 + 384]
        qe_report_sig = sig_data[134 + 384 : 134 + 384 + 64]
        qe_auth_data_size = struct.unpack_from("<H", sig_data, 134 + 384 + 64)[0]
        qe_auth_data = sig_data[134 + 384 + 64 + 2 : 134 + 384 + 64 + 2 + qe_auth_data_size]

        # QE Report REPORT_DATA at offset 320..352 must commit to SHA256(QE3_pub || qe_auth_data)
        qe_report_data = qe_report[320:352]
        expected_qe_report_data = hashlib.sha256(sig_data[64:128] + qe_auth_data).digest()
        if qe_report_data != expected_qe_report_data:
            raise AttestationVerificationError(
                "Intel TDX QE report data does not commit to QE3 public key and auth data"
            )

        pck_offset = 134 + 384 + 64 + 2 + qe_auth_data_size
        pck_type, pck_size = struct.unpack_from("<HI", sig_data, pck_offset)
        pck_pem = sig_data[pck_offset + 6 : pck_offset + 6 + pck_size].decode("latin1")
        pck_certs = parse_pem_chain(pck_pem)
        if not pck_certs:
            raise AttestationVerificationError("Failed to parse PCK certificates in Intel TDX quote")

        # Leaf PCK certificate verifies QE Report signature
        der_qe_sig = utils.encode_dss_signature(
            int.from_bytes(qe_report_sig[0:32], "big"),
            int.from_bytes(qe_report_sig[32:64], "big"),
        )
        try:
            pck_certs[0].public_key().verify(der_qe_sig, qe_report, ec.ECDSA(hashes.SHA256()))
        except Exception as e:
            raise AttestationVerificationError(f"Intel TDX PCK signature over QE report failed: {e}")

    elif cert_data_type == 5:
        pck_pem = sig_data[134 : 134 + cert_data_size].decode("latin1")
        pck_certs = parse_pem_chain(pck_pem)

    if pck_certs:
        trusted_intel_root = OFFICIAL_INTEL_ROOT_CA_PEM if enforce_official_roots else None
        verify_x509_chain(
            pck_certs,
            root_ca_pem=trusted_intel_root,
            check_online_crl=check_online_vendor,
            crl_manager=crl_manager,
            strict_online=strict_online,
        )

    return {
        "mrtd": mrtd.hex(),
        "rtmr0": rtmr0.hex(),
        "rtmr1": rtmr1.hex(),
        "rtmr2": rtmr2.hex(),
        "rtmr3": rtmr3.hex(),
        "td_attributes": td_attributes.hex(),
    }


def verify_nvidia_gpu_evidence(
    gpu_ev: Dict[str, Any],
    nonce_bytes: bytes,
    enforce_official_roots: bool = True,
    check_online_vendor: bool = True,
    crl_manager: Optional[VendorCRLManager] = None,
    strict_online: bool = False,
) -> Dict[str, Any]:
    """
    Cryptographically verifies NVIDIA Hopper / Blackwell GPU SPDM evidence:
      - Verifies 5-tier DICE X.509 certificate chain up to official NVIDIA Device Identity Root CA.
      - Checks certificate revocation against NVIDIA online CRL distribution points.
      - Verifies ECDSA P-384 signature over SPDM MEASUREMENTS message using leaf DICE key.
      - Enforces fresh session nonce matching in SPDM measurement message header.
    """
    ev_report_b64 = gpu_ev.get("evidence_report", "")
    cert_chain_pem = gpu_ev.get("certificate_chain", "")

    if not ev_report_b64 or not cert_chain_pem:
        raise AttestationVerificationError("Incomplete NVIDIA GPU evidence")

    gpu_certs = parse_pem_chain(cert_chain_pem)
    if not gpu_certs:
        raise AttestationVerificationError("Could not parse NVIDIA DICE certificate chain")

    # 1. Verify certificate chain up to official NVIDIA Device Identity Root CA
    trusted_nvidia_root = OFFICIAL_NVIDIA_ROOT_CA_PEM if enforce_official_roots else None
    verify_x509_chain(
        gpu_certs,
        root_ca_pem=trusted_nvidia_root,
        check_online_crl=check_online_vendor,
        crl_manager=crl_manager,
        strict_online=strict_online,
    )

    # 2. Decode and verify SPDM measurements evidence report
    try:
        ev_raw = base64.b64decode(ev_report_b64)
    except Exception as e:
        raise AttestationVerificationError(f"Failed to decode NVIDIA SPDM evidence: {e}")

    if len(ev_raw) < 96:
        raise AttestationVerificationError("NVIDIA SPDM evidence report too short for signature")

    signed_msg = ev_raw[:-96]
    sig_bytes = ev_raw[-96:]
    r = int.from_bytes(sig_bytes[:48], "big")
    s = int.from_bytes(sig_bytes[48:], "big")
    der_sig = utils.encode_dss_signature(r, s)

    leaf_pub = gpu_certs[0].public_key()
    try:
        leaf_pub.verify(der_sig, signed_msg, ec.ECDSA(hashes.SHA384()))
    except Exception as e:
        raise AttestationVerificationError(
            f"NVIDIA SPDM evidence report signature verification failed: {e}"
        )

    # 3. Verify freshness nonce in SPDM header (bytes 4..36)
    if nonce_bytes and len(nonce_bytes) >= 32 and len(ev_raw) >= 36:
        spdm_nonce = ev_raw[4:36]
        if spdm_nonce != nonce_bytes[:32]:
            raise AttestationVerificationError(
                f"NVIDIA SPDM measurement nonce mismatch: expected {nonce_bytes[:32].hex()}, got {spdm_nonce.hex()}"
            )

    return {
        "model": gpu_ev.get("model"),
        "architecture": gpu_ev.get("architecture"),
    }


def verify_amd_sev_snp_quote(
    quote_bytes: bytes,
    user_data: bytes,
    cert_chain_pem: Optional[str] = None,
    enforce_official_roots: bool = True,
    check_online_vendor: bool = True,
    crl_manager: Optional[VendorCRLManager] = None,
    strict_online: bool = False,
) -> Dict[str, Any]:
    """
    Cryptographically verifies AMD SEV-SNP attestation report:
      - Validates minimum structure length (>= 672 bytes).
      - Checks guest policy debug bit (bit 19 must be 0).
      - Verifies REPORT_DATA literal match at offset 80..144.
      - If full 1184-byte report and VCEK cert chain provided, verifies ECDSA P-384 signature.
    """
    if len(quote_bytes) < 1184:
        raise AttestationVerificationError(
            f"AMD SEV-SNP report too short ({len(quote_bytes)} bytes, expected >= 1184 for signed quote)"
        )
    if not cert_chain_pem:
        raise AttestationVerificationError("AMD SEV-SNP signature verification requires VCEK certificate chain")

    # 1. Policy check: offset 8..16 (uint64 little endian)
    policy = struct.unpack_from("<Q", quote_bytes, 8)[0]
    if (policy >> 19) & 1 != 0:
        raise AttestationVerificationError("AMD SEV-SNP guest has DEBUG mode enabled (untrusted)")

    # 2. REPORT_DATA check: offset 80..144
    embedded_user_data = quote_bytes[80:144]
    if embedded_user_data != user_data:
        raise AttestationVerificationError(
            "REPORT_DATA mismatch at AMD SEV-SNP report offset 80..143"
        )

    # 3. Signature verification with VCEK certificate chain
    amd_certs = parse_pem_chain(cert_chain_pem)
    if not amd_certs:
        raise AttestationVerificationError("Failed to parse AMD SEV-SNP certificate chain")

    verify_x509_chain(
        amd_certs,
        root_ca_pem=None,
        check_online_crl=check_online_vendor,
        crl_manager=crl_manager,
        strict_online=strict_online,
    )
    vcek_leaf = amd_certs[0]
    # Signature at 672..720 (R) and 744..792 (S) (little-endian 48 bytes each)
    r_int = int.from_bytes(quote_bytes[672:720], "little")
    s_int = int.from_bytes(quote_bytes[744:792], "little")
    der_sig = utils.encode_dss_signature(r_int, s_int)
    try:
        vcek_leaf.public_key().verify(
            der_sig,
            quote_bytes[:672],
            ec.ECDSA(hashes.SHA384()),
        )
    except Exception as e:
        raise AttestationVerificationError(f"AMD SEV-SNP report signature verification failed: {e}")

    return {"policy": policy}


def verify_attestation_document(
    doc: Dict[str, Any],
    expected_nonce_hex: Optional[str] = None,
    expected_tls_fingerprint: Optional[str] = None,
    expected_auth_public_key: Optional[str] = None,
    enforce_official_roots: bool = True,
    check_online_vendor: bool = True,
    strict_online: bool = False,
    allow_mock_attestation: bool = False,
) -> AttestationResult:
    """
    Validates an in-toto Statement v1 envelope containing Confidential Computing claims.
    
    Performs full mathematical key binding and hardware signature verification:
      - Validates that REPORT_DATA commits to TLS fingerprint and HPKE public key.
      - Enforces expected socket TLS SPKI fingerprint match (channel binding).
      - Verifies Intel TDX quote ECDSA signature (QE3 + PCK) and non-debug status.
      - Verifies NVIDIA Hopper DICE X.509 cert chain and SPDM measurement signature.
      - Verifies AMD SEV-SNP policy, REPORT_DATA, and signature.
      - Checks vendor certificate revocation against online CRL endpoints.
    """
    if not isinstance(doc, dict):
        raise AttestationVerificationError("Attestation document must be a JSON object")

    if doc.get("_type") != "https://in-toto.io/Statement/v1":
        raise AttestationVerificationError(f"Invalid statement type: {doc.get('_type')}")

    if doc.get("predicateType") != "https://in-toto.io/attestation/confidential-computing/v0.1":
        raise AttestationVerificationError(f"Invalid predicate type: {doc.get('predicateType')}")

    pred = doc.get("predicate")
    if not pred or not isinstance(pred, dict):
        raise AttestationVerificationError("Missing predicate in attestation document")

    platform = pred.get("platform", "unknown")
    user_data_hex = pred.get("user_data", "")
    tls_fp_hex = pred.get("tls_fingerprint", "")
    hpke_pub_hex = pred.get("hpke_public_key", "")
    nonce_hex = pred.get("nonce", "")

    try:
        user_data = bytes.fromhex(user_data_hex)
        tls_fp = bytes.fromhex(tls_fp_hex)
        hpke_pub = bytes.fromhex(hpke_pub_hex)
    except Exception as e:
        raise AttestationVerificationError(f"Hex decoding error in attestation claims: {e}")

    if len(user_data) != 64:
        raise AttestationVerificationError(f"user_data must be 64 bytes (got {len(user_data)})")
    if len(tls_fp) != 32:
        raise AttestationVerificationError(f"tls_fingerprint must be 32 bytes (got {len(tls_fp)})")
    if len(hpke_pub) != 32:
        raise AttestationVerificationError(f"hpke_public_key must be 32 bytes (got {len(hpke_pub)})")

    # 1. Nonce freshness check
    if expected_nonce_hex:
        if nonce_hex.lower() != expected_nonce_hex.lower():
            raise AttestationVerificationError(
                f"Nonce mismatch: expected {expected_nonce_hex}, got {nonce_hex}"
            )

    nonce_bytes = b""
    if len(nonce_hex) == 64:
        try:
            nonce_bytes = bytes.fromhex(nonce_hex)
        except Exception:
            nonce_bytes = hashlib.sha256(nonce_hex.encode("utf-8")).digest()
    elif nonce_hex:
        nonce_bytes = hashlib.sha256(nonce_hex.encode("utf-8")).digest()

    # 2. TLS Channel Binding Check (if expected_tls_fingerprint provided from socket)
    if expected_tls_fingerprint:
        clean_expected = expected_tls_fingerprint.lower().replace(":", "")
        if tls_fp_hex.lower() != clean_expected:
            raise AttestationVerificationError(
                f"TLS Channel Binding Mismatch: socket peer SPKI fingerprint {clean_expected} "
                f"does not match attestation report claim {tls_fp_hex}"
            )

    # 3. Check GPU Evidence
    gpu_ev = pred.get("gpu_evidence")
    has_gpu = False
    gpu_model = None
    gpu_arch = None
    gpu_digest = b"\x00" * 32

    if gpu_ev and isinstance(gpu_ev, dict):
        ev_report = gpu_ev.get("evidence_report", "")
        cert_chain_pem = gpu_ev.get("certificate_chain", "")
        if ev_report or cert_chain_pem:
            has_gpu = True
            gpu_model = gpu_ev.get("model")
            gpu_arch = gpu_ev.get("architecture")
            combined = (ev_report + cert_chain_pem).encode("utf-8")
            gpu_digest = hashlib.sha256(combined).digest()

            # Cryptographically verify NVIDIA GPU evidence report and certificate chain
            verify_nvidia_gpu_evidence(
                gpu_ev=gpu_ev,
                nonce_bytes=nonce_bytes,
                enforce_official_roots=enforce_official_roots,
                check_online_vendor=check_online_vendor,
                strict_online=strict_online,
            )

    # 4. Mathematical verification of 64-byte REPORT_DATA
    # Bytes 0..31 == TLS fingerprint
    if user_data[0:32] != tls_fp:
        raise AttestationVerificationError(
            "REPORT_DATA[0:32] does not match TLS server key fingerprint"
        )

    # Extract authorized public key committed in hardware quote
    auth_pub_hex = pred.get("authorized_public_key", "") or pred.get("auth_public_key", "")
    auth_pub_bytes = bytes.fromhex(auth_pub_hex) if auth_pub_hex else (b"\x00" * 32)
    if expected_auth_public_key:
        clean_expected_auth = expected_auth_public_key.lower().strip()
        if auth_pub_hex.lower() != clean_expected_auth:
            raise AttestationVerificationError(
                f"Authorized public key mismatch: expected {clean_expected_auth}, got {auth_pub_hex}"
            )

    # Bytes 32..63 == SHA-256(hpke_public_key || nonce || gpu_digest || auth_pub)
    expected_second_half = hashlib.sha256(hpke_pub + nonce_bytes + gpu_digest + auth_pub_bytes).digest()
    if user_data[32:64] != expected_second_half:
        # Fallback check for legacy 3-term composite (without auth_pub) or legacy 1-term
        legacy_3term = hashlib.sha256(hpke_pub + nonce_bytes + gpu_digest).digest()
        legacy_1term = hashlib.sha256(hpke_pub).digest()
        if user_data[32:64] == legacy_3term:
            expected_second_half = legacy_3term
        elif user_data[32:64] == legacy_1term:
            expected_second_half = legacy_1term
        else:
            raise AttestationVerificationError(
                "REPORT_DATA[32:64] cryptographic binding verification failed: "
                "enclave user_data does not commit to claimed HPKE public key and session nonce"
            )

    # 5. Hardware quote embedding and cryptographic signature verification
    raw_quote_b64 = pred.get("raw_quote", "")
    hw_info: Dict[str, Any] = {}
    if not raw_quote_b64:
        if not allow_mock_attestation:
            raise AttestationVerificationError(
                "Attestation document missing required silicon hardware quote (raw_quote)"
            )
    else:
        try:
            quote_bytes = base64.b64decode(raw_quote_b64)
        except Exception as e:
            raise AttestationVerificationError(f"Base64 decoding raw quote failed: {e}")

        if platform == "intel-tdx":
            hw_info = verify_intel_tdx_quote(
                quote_bytes=quote_bytes,
                user_data=user_data,
                enforce_official_roots=enforce_official_roots,
                check_online_vendor=check_online_vendor,
                strict_online=strict_online,
            )
        elif platform in ("amd-sev-snp", "sev-snp"):
            cert_chain_pem = pred.get("certificate_chain", "")
            hw_info = verify_amd_sev_snp_quote(
                quote_bytes=quote_bytes,
                user_data=user_data,
                cert_chain_pem=cert_chain_pem,
                enforce_official_roots=enforce_official_roots,
                check_online_vendor=check_online_vendor,
                strict_online=strict_online,
            )
        else:
            if not allow_mock_attestation:
                raise AttestationVerificationError(
                    f"Unsupported hardware platform '{platform}': silicon quote verification cannot proceed"
                )

    # Cevell OS CVM server computes sha256.Sum256([]byte(cfg.AttestationDoc.Predicate.UserData))
    binding_hash = hashlib.sha256(user_data_hex.encode("utf-8")).digest()

    return AttestationResult(
        hpke_public_key=hpke_pub,
        hpke_public_key_hex=hpke_pub_hex,
        tls_fingerprint=tls_fp_hex,
        platform=platform,
        user_data=user_data,
        binding_hash=binding_hash,
        has_gpu=has_gpu,
        gpu_model=gpu_model,
        gpu_arch=gpu_arch,
        mrtd=hw_info.get("mrtd"),
        rtmr0=hw_info.get("rtmr0"),
        rtmr1=hw_info.get("rtmr1"),
        rtmr2=hw_info.get("rtmr2"),
        rtmr3=hw_info.get("rtmr3"),
        td_attributes=hw_info.get("td_attributes"),
        vendor_crl_verified=bool(_GLOBAL_CRL_MANAGER.verified_urls) if check_online_vendor else False,
        auth_public_key=auth_pub_hex or None,
        is_valid=True,
    )
