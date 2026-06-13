from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


def load_api_keys(
    encrypted_file: str | Path, master_key: str | bytes
) -> dict[str, Any]:
    path = Path(encrypted_file).expanduser().resolve()
    if not path.exists():
        return {}

    payload = path.read_bytes()
    if not payload:
        return {}

    fernet = Fernet(_normalize_master_key(master_key))
    try:
        decrypted = fernet.decrypt(payload)
    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt API key vault with the supplied master key."
        ) from exc
    return json.loads(decrypted.decode("utf-8"))


def save_api_keys(
    encrypted_file: str | Path, master_key: str | bytes, payload: dict[str, Any]
) -> None:
    path = Path(encrypted_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fernet = Fernet(_normalize_master_key(master_key))
    serialized = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    path.write_bytes(fernet.encrypt(serialized))


def _normalize_master_key(master_key: str | bytes) -> bytes:
    if isinstance(master_key, bytes):
        return master_key
    encoded = master_key.encode("utf-8")
    if len(encoded) != 44:
        raise ValueError("Fernet master keys must be 44 characters long.")
    return encoded
