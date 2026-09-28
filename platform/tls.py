# Apache-2.0
"""A self-signed certificate, so the platform can also be served over HTTPS.

Browsers hand a page the camera only on https:// or localhost. Opened from
another machine by IP, the webcam preview needs HTTPS; a certificate made here
does that, at the cost of one "not private" warning the first time.
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path


def ensure_certificate(folder: Path) -> tuple[Path, Path]:
    """``(cert.pem, key.pem)`` in ``folder``, made once and then reused.

    Reusing it matters: a browser remembers the exception for a certificate,
    so a new one on every start would bring the warning back every time.
    """
    folder = Path(folder)
    cert, key = folder / "cert.pem", folder / "key.pem"
    if cert.exists() and key.exists():
        return cert, key

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "rtdetr platform")])
    now = datetime.datetime.now(datetime.timezone.utc)
    names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    host = socket.gethostname()
    if host and host != "localhost":
        names.append(x509.DNSName(host))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(private, hashes.SHA256())
    )
    folder.mkdir(parents=True, exist_ok=True)
    key.write_bytes(private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    key.chmod(0o600)
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert, key
