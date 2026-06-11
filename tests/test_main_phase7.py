from __future__ import annotations

import json
from pathlib import Path

from main import build_app, load_bootstrap_config


def test_bootstrap_config_loads_yaml_and_attaches_runtime_state(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
workspace_root: {workspace_root}
settings:
  db_path: data/custom.db
  skills_dir: skills
  persona_vectors_path: data/persona_vectors.npz
  gating_model_path: data/gating_model.pkl
  message_hub_enabled: false
  evolution_enabled: false
constitutions:
  restricted: Safe mode.
  unrestricted: Lab mode.
mcp_servers:
  filesystem:
    command: ["python", "-c", "print('mcp')"]
    methods: ["filesystem.read"]
""".format(workspace_root=workspace_root.as_posix()),
        encoding="utf-8",
    )

    settings, bootstrap = load_bootstrap_config(config_path)
    app = build_app(settings, bootstrap)

    assert settings.db_path.name == "custom.db"
    assert bootstrap.constitution_texts["restricted"] == "Safe mode."
    assert app.state.bootstrap_config.path == config_path
    assert app.state.constitution_texts["unrestricted"] == "Lab mode."
    assert settings.workspace_root == workspace_root.resolve()
    assert (workspace_root / "mcp_servers.json").exists()
    payload = json.loads((workspace_root / "mcp_servers.json").read_text(encoding="utf-8"))
    assert payload["servers"]["filesystem"]["methods"] == ["filesystem.read"]
