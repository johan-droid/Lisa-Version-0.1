from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from cryptography.fernet import Fernet

from utils.config_loader import load_config
from utils.encryption import load_api_keys, save_api_keys
from utils.snapshot import get_hmac_key, get_snapshot_key_path


def test_async_config_loader_supports_plan_style_and_current_layout(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
app_name: LISA
workspace_root: .
model_path: models/tinyllama.gguf
max_concurrent_arms: 12
constitution_restricted: Safe mode.
constitution_unrestricted: Lab mode.
mcp_servers:
  - name: filesystem
    command: ["python", "-c", "print('mcp')"]
    args: ["--verbose"]
    methods: ["filesystem.read"]
""",
        encoding="utf-8",
    )

    async def run() -> dict[str, object]:
        return await load_config(config_path)

    config = asyncio.run(run())

    assert config["app_name"] == "LISA"
    assert config["model_path"].name == "tinyllama.gguf"
    assert config["constitution_restricted"] == "Safe mode."
    assert config["max_concurrent_arms"] == 12
    assert config["mcp_servers"][0]["name"] == "filesystem"


def test_encrypted_api_key_round_trip(tmp_path: Path) -> None:
    master_key = Fernet.generate_key()
    vault_path = tmp_path / "keys.enc"

    save_api_keys(
        vault_path, master_key, {"providers": {"openai": {"api_key": "secret"}}}
    )
    payload = load_api_keys(vault_path, master_key)

    assert payload["providers"]["openai"]["api_key"] == "secret"


def test_snapshot_key_is_generated_persistently_when_not_configured(tmp_path: Path) -> None:
    settings = SimpleNamespace(workspace_root=tmp_path, bot_security_key=None)

    first = get_hmac_key(settings)
    second = get_hmac_key(settings)
    key_path = get_snapshot_key_path(settings)

    assert key_path.exists()
    assert first == second
    assert len(first) >= 32
