"""
Import this before making any HTTPS request in this project.

Some networks (school/corporate) do TLS interception, which breaks Python's
bundled CA verification. `truststore` routes verification through the OS
trust store instead, which has the interception CA installed -- the correct
fix, not verify=False. Falls back to default verification if truststore
isn't installed or injection fails, so this is a no-op on a normal network.
"""

from __future__ import annotations

import base64
import os
import ssl
import tempfile
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE_ACTIVE = True
except Exception:  # noqa: BLE001
    TRUSTSTORE_ACTIVE = False


def ensure_curl_cffi_ca_bundle() -> None:
    """curl_cffi (yfinance's HTTP backend) ships its own vendored TLS stack,
    entirely separate from Python's ssl module -- truststore.inject_into_ssl()
    above never reaches it. On a network doing TLS interception, curl_cffi
    then rejects the interception cert outright ("unable to get local issuer
    certificate"), because certifi's public-CA-only bundle doesn't include
    it either. Fix: build a combined bundle (certifi + whatever the OS
    already trusts, via ssl.enum_certificates) and point curl_cffi at it via
    CURL_CA_BUNDLE. No-op (falls back to certifi alone) on non-Windows or if
    this fails, since it's not needed on a network without interception.
    Call this before importing yfinance or any other curl_cffi-based
    library."""
    import certifi

    bundle_path = Path(tempfile.gettempdir()) / "10q_equity_agent_ca_bundle.pem"
    try:
        with open(certifi.where(), "rb") as f:
            pem_bytes = f.read()
        for store in ("ROOT", "CA"):
            for der, encoding, _trust in ssl.enum_certificates(store):
                if encoding != "x509_asn":
                    continue
                b64 = base64.b64encode(der).decode("ascii")
                lines = "\n".join(b64[i:i + 64] for i in range(0, len(b64), 64))
                pem_bytes += f"-----BEGIN CERTIFICATE-----\n{lines}\n-----END CERTIFICATE-----\n".encode()
        bundle_path.write_bytes(pem_bytes)
        os.environ["CURL_CA_BUNDLE"] = str(bundle_path)
    except Exception:  # noqa: BLE001 -- e.g. non-Windows, no enum_certificates
        os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
