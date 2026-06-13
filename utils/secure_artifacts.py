from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

from cryptography.fernet import Fernet


def derive_fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def build_local_artifact_url(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def artifact_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported artifact URL scheme: {parsed.scheme or 'none'}")
    raw_path = unquote(parsed.path)
    if raw_path.startswith("/") and len(raw_path) > 3 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    return Path(raw_path).resolve()


def encrypt_artifact_url(url: str, secret: str) -> str:
    cipher = Fernet(derive_fernet_key(secret))
    return cipher.encrypt(url.encode("utf-8")).decode("utf-8")


def decrypt_artifact_url(encrypted_url: str, secret: str) -> str:
    cipher = Fernet(derive_fernet_key(secret))
    return cipher.decrypt(encrypted_url.encode("utf-8")).decode("utf-8")
