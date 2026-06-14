from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml


async def load_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize the bootstrap configuration from YAML.

    The loader accepts both the current repository layout and the plan-style
    layout so the runtime can evolve without breaking older config files.
    """

    resolved = Path(path).expanduser().resolve()
    return await asyncio.to_thread(_load_config_sync, resolved)


def _load_config_sync(path: Path) -> dict[str, Any]:
    raw = _read_yaml(path)
    settings = dict(raw.get("settings") or {})
    constitutions = dict(raw.get("constitutions") or {})

    normalized: dict[str, Any] = {
        "path": path,
        "raw": raw,
        "app_name": raw.get("app_name") or settings.get("app_name") or "LISA",
        "agent_id": raw.get("agent_id") or settings.get("agent_id") or "lisa",
        "workspace_root": _path_or_default(
            raw.get("workspace_root") or settings.get("workspace_root") or ".",
            base=path.parent,
        ),
        "model_path": _path_or_default(
            raw.get("model_path")
            or settings.get("local_model_path")
            or settings.get("model_path")
            or "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            base=path.parent,
        ),
        "local_model_path": _path_or_default(
            raw.get("model_path")
            or settings.get("local_model_path")
            or settings.get("model_path")
            or "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            base=path.parent,
        ),
        "db_path": _path_or_default(
            settings.get("db_path") or raw.get("db_path") or "data/lisa_notepad.db",
            base=path.parent,
        ),
        "redis_url": raw.get("redis_url") or settings.get("redis_url"),
        "postgres_dsn": raw.get("postgres_dsn") or settings.get("postgres_dsn"),
        "chroma_persist_dir": _path_or_default(
            settings.get("chroma_persist_dir")
            or raw.get("chroma_persist_dir")
            or "data/chroma",
            base=path.parent,
        ),
        "working_memory_ttl_seconds": int(
            raw.get("working_memory_ttl_seconds")
            if raw.get("working_memory_ttl_seconds") is not None
            else settings.get("working_memory_ttl_seconds", 7200)
        ),
        "skills_dir": _path_or_default(
            settings.get("skills_dir") or raw.get("skills_dir") or "skills",
            base=path.parent,
        ),
        "evolution_artifacts_dir": _path_or_default(
            settings.get("evolution_artifacts_dir")
            or raw.get("evolution_artifacts_dir")
            or "data/evolution_artifacts",
            base=path.parent,
        ),
        "persona_vectors_path": _path_or_default(
            settings.get("persona_vectors_path")
            or raw.get("persona_vectors_path")
            or "data/persona_vectors.npz",
            base=path.parent,
        ),
        "gating_model_path": _path_or_default(
            settings.get("gating_model_path")
            or raw.get("gating_model_path")
            or "data/gating_model.pkl",
            base=path.parent,
        ),
        "constitution_restricted": str(
            raw.get("constitution_restricted")
            or constitutions.get("restricted")
            or "You are LISA in restricted mode."
        ),
        "constitution_unrestricted": str(
            raw.get("constitution_unrestricted")
            or constitutions.get("unrestricted")
            or "You are LISA in unrestricted lab mode."
        ),
        "mcp_servers": _normalize_mcp_servers(raw.get("mcp_servers")),
        "interface_keys": dict(raw.get("interface_keys") or {}),
        "freellmapi": list(raw.get("freellmapi") or []),
        "evolution_time_range": _normalize_time_range(
            raw.get("evolution_time_range"), settings
        ),
        "max_concurrent_arms": int(
            raw.get("max_concurrent_arms") or settings.get("max_concurrent_arms") or 10
        ),
        "message_hub_start_listener": bool(
            raw.get("message_hub_start_listener")
            if raw.get("message_hub_start_listener") is not None
            else settings.get("message_hub_start_listener", False)
        ),
        "allow_local_terminal_fallback": bool(
            raw.get("allow_local_terminal_fallback")
            if raw.get("allow_local_terminal_fallback") is not None
            else settings.get("allow_local_terminal_fallback", False)
        ),
        "hybrid_brain_enabled": bool(
            raw.get("hybrid_brain_enabled")
            if raw.get("hybrid_brain_enabled") is not None
            else settings.get("hybrid_brain_enabled", False)
        ),
        "hybrid_brain_stress_threshold": int(
            raw.get("hybrid_brain_stress_threshold")
            if raw.get("hybrid_brain_stress_threshold") is not None
            else settings.get("hybrid_brain_stress_threshold", 4)
        ),
        "hybrid_brain_prompt_chars_threshold": int(
            raw.get("hybrid_brain_prompt_chars_threshold")
            if raw.get("hybrid_brain_prompt_chars_threshold") is not None
            else settings.get("hybrid_brain_prompt_chars_threshold", 600)
        ),
        "hybrid_brain_race_window_ms": int(
            raw.get("hybrid_brain_race_window_ms")
            if raw.get("hybrid_brain_race_window_ms") is not None
            else settings.get("hybrid_brain_race_window_ms", 250)
        ),
        "evolution_skill_autoload_limit": int(
            raw.get("evolution_skill_autoload_limit")
            if raw.get("evolution_skill_autoload_limit") is not None
            else settings.get("evolution_skill_autoload_limit", 2)
        ),
        "admin_api_token": raw.get("admin_api_token")
        or settings.get("admin_api_token"),
        "session_token_ttl_seconds": int(
            raw.get("session_token_ttl_seconds")
            if raw.get("session_token_ttl_seconds") is not None
            else settings.get("session_token_ttl_seconds", 300)
        ),
        "allow_remote_bind": bool(
            raw.get("allow_remote_bind")
            if raw.get("allow_remote_bind") is not None
            else settings.get("allow_remote_bind", False)
        ),
        "autonomous_enabled": bool(
            raw.get("autonomous_enabled")
            if raw.get("autonomous_enabled") is not None
            else settings.get("autonomous_enabled", False)
        ),
        "enable_unsafe_admin_endpoints": bool(
            raw.get("enable_unsafe_admin_endpoints")
            if raw.get("enable_unsafe_admin_endpoints") is not None
            else settings.get("enable_unsafe_admin_endpoints", False)
        ),
        "telegram_allowed_user_ids": list(
            raw.get("telegram_allowed_user_ids")
            or settings.get("telegram_allowed_user_ids")
            or []
        ),
        "slack_allowed_user_ids": list(
            raw.get("slack_allowed_user_ids")
            or settings.get("slack_allowed_user_ids")
            or []
        ),
        "whatsapp_allowed_user_ids": list(
            raw.get("whatsapp_allowed_user_ids")
            or settings.get("whatsapp_allowed_user_ids")
            or []
        ),
        "sentry_dsn": raw.get("sentry_dsn") or settings.get("sentry_dsn"),
        "sentry_environment": raw.get("sentry_environment")
        or settings.get("sentry_environment"),
        "sentry_release": raw.get("sentry_release") or settings.get("sentry_release"),
        "sentry_traces_sample_rate": float(
            raw.get("sentry_traces_sample_rate")
            if raw.get("sentry_traces_sample_rate") is not None
            else settings.get("sentry_traces_sample_rate", 0.0)
        ),
        "sentry_profiles_sample_rate": float(
            raw.get("sentry_profiles_sample_rate")
            if raw.get("sentry_profiles_sample_rate") is not None
            else settings.get("sentry_profiles_sample_rate", 0.0)
        ),
        "sentry_send_default_pii": bool(
            raw.get("sentry_send_default_pii")
            if raw.get("sentry_send_default_pii") is not None
            else settings.get("sentry_send_default_pii", False)
        ),
    }

    return normalized


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a YAML mapping.")
    return dict(data)


def _path_or_default(value: Any, *, base: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def _normalize_mcp_servers(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                normalized.append(
                    {
                        "name": str(item["name"]),
                        "command": [str(part) for part in item.get("command", [])],
                        "args": [str(part) for part in item.get("args", [])],
                        "methods": [str(method) for method in item.get("methods", [])],
                        "env": dict(item.get("env") or {}),
                        "cwd": item.get("cwd"),
                    }
                )
        return normalized

    if isinstance(value, dict):
        normalized = []
        for name, item in value.items():
            raw = item if isinstance(item, dict) else {}
            normalized.append(
                {
                    "name": str(name),
                    "command": [str(part) for part in raw.get("command", [])],
                    "args": [str(part) for part in raw.get("args", [])],
                    "methods": [str(method) for method in raw.get("methods", [])],
                    "env": dict(raw.get("env") or {}),
                    "cwd": raw.get("cwd"),
                }
            )
        return normalized

    return []


def _normalize_time_range(value: Any, settings: dict[str, Any]) -> dict[str, int]:
    if isinstance(value, dict):
        start = int(value.get("start", 3))
        end = int(value.get("end", 5))
        return {"start": start, "end": end}

    start_hour = int(settings.get("evolution_window_start_hour", 3))
    duration = int(settings.get("evolution_window_duration_hours", 2))
    return {"start": start_hour, "end": (start_hour + duration) % 24}
