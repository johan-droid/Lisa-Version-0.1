from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker


@dataclass(slots=True)
class EnvironmentCheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if self.errors:
            raise EnvironmentError("\n".join(self.errors))


def check_environment(settings: Any, bootstrap: Any | None = None) -> EnvironmentCheckResult:
    result = EnvironmentCheckResult(ok=True)
    workspace_root = Path(getattr(settings, "workspace_root", Path.cwd())).resolve()

    model_path = getattr(settings, "local_model_path", None) or getattr(settings, "model_path", None)
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            message = f"Missing model file: {model_path}"
            if _is_truthy(os.environ.get("LISA_REQUIRE_LOCAL_MODEL")):
                result.errors.append(message)
            else:
                result.warnings.append(f"{message} (continuing without local inference)")

    docker_error = _check_docker()
    if docker_error:
        if _is_truthy(os.environ.get("LISA_REQUIRE_DOCKER")):
            result.errors.append(docker_error)
        else:
            result.warnings.append(docker_error)

    if not os.access(workspace_root, os.W_OK):
        result.errors.append(f"Workspace is not writable: {workspace_root}")

    backup_dir = Path(getattr(settings, "backup_dir", workspace_root / "backups"))
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        result.errors.append(f"Unable to create backup directory {backup_dir}: {exc}")

    if bootstrap is not None and not getattr(bootstrap, "constitution_texts", None):
        result.warnings.append("No constitution texts were provided; defaults will be used.")

    result.ok = not result.errors
    return result


def _check_docker() -> str | None:
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        return f"Docker is unavailable or not running: {exc}"
    return None


def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
