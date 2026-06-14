from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from lisa.api import create_app
from lisa.config import Settings
from utils.encryption import load_api_keys
from utils.env_check import check_environment
from utils.logger import configure_logging
from utils.observability import configure_sentry
from utils.process_lock import ProcessLock, ProcessLockHeldError

LOGGER = logging.getLogger("lisa.startup")

try:
    import resource
except ImportError:  # pragma: no cover - Windows and some embedded runtimes
    resource = None  # type: ignore[assignment]


@dataclass(slots=True)
class BootstrapConfig:
    path: Path
    raw: dict[str, Any] = field(default_factory=dict)
    constitution_texts: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)


def apply_memory_limit(limit_bytes: int = 1_000_000_000) -> None:
    """Best-effort address-space cap for Unix-like runtimes."""

    if resource is None:
        LOGGER.info(
            "Skipping RLIMIT_AS cap because the resource module is unavailable."
        )
        return

    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        LOGGER.info("Applied RLIMIT_AS soft=%s hard=%s", soft_limit, hard_limit)
    except (AttributeError, ValueError, OSError) as exc:
        LOGGER.warning("Unable to apply RLIMIT_AS memory cap: %s", exc)


def load_bootstrap_config(path: Path) -> tuple[Settings, BootstrapConfig]:
    raw = _load_yaml_mapping(path)
    settings_payload = _collect_settings_payload(raw)
    settings = Settings(**settings_payload)

    constitution_texts = _collect_constitution_texts(raw)
    mcp_servers = _normalize_mcp_servers(raw.get("mcp_servers"))
    bootstrap = BootstrapConfig(
        path=path,
        raw=raw,
        constitution_texts=constitution_texts,
        mcp_servers=mcp_servers,
    )
    return settings, bootstrap


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a YAML mapping.")
    return dict(data)


def _collect_settings_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("settings", "model", "freellmapi", "message_hub", "evolution", "paths"):
        section = _mapping(raw.get(key))
        payload.update(section)

    if "workspace_root" in raw:
        payload["workspace_root"] = raw["workspace_root"]
    if "model_path" in raw:
        payload["local_model_path"] = raw["model_path"]
    for key in (
        "constitution_restricted",
        "constitution_unrestricted",
        "interface_keys",
        "max_concurrent_arms",
    ):
        if key in raw:
            payload[key] = raw[key]

    for key in (
        "app_name",
        "agent_id",
        "workspace_root",
        "model_path",
        "local_model_path",
        "db_path",
        "redis_url",
        "postgres_dsn",
        "chroma_persist_dir",
        "working_memory_ttl_seconds",
        "skills_dir",
        "evolution_artifacts_dir",
        "persona_vectors_path",
        "gating_model_path",
        "local_model_path",
        "local_model_context_size",
        "local_model_n_threads",
        "local_model_n_gpu_layers",
        "docker_image",
        "allow_local_terminal_fallback",
        "freellmapi_base_url",
        "freellmapi_api_key",
        "freellmapi_default_provider",
        "freellmapi_timeout_seconds",
        "message_hub_enabled",
        "message_hub_host",
        "message_hub_port",
        "message_hub_start_listener",
        "evolution_enabled",
        "autonomous_enabled",
        "evolution_check_interval_seconds",
        "evolution_window_start_hour",
        "evolution_window_duration_hours",
        "evolution_idle_after_seconds",
        "evolution_candidate_limit",
        "evolution_browser_queries",
        "evolution_min_reward",
        "evolution_staging_dir",
        "incoming_queue_size",
        "tool_timeout_seconds",
        "hybrid_brain_enabled",
        "hybrid_brain_stress_threshold",
        "hybrid_brain_prompt_chars_threshold",
        "hybrid_brain_race_window_ms",
        "evolution_skill_autoload_limit",
        "model_provider",
        "model_name",
        "model_base_url",
        "model_api_key",
        "external_timeout_seconds",
        "enable_browser_tools",
        "admin_api_token",
        "session_token_ttl_seconds",
        "allow_remote_bind",
        "enable_unsafe_admin_endpoints",
        "enable_legacy_bot_pairing",
        "telegram_allowed_user_ids",
        "slack_allowed_user_ids",
        "whatsapp_allowed_user_ids",
        "max_concurrent_arms",
        "freellmapi_requests_per_minute",
        "freellmapi_tokens_per_minute",
        "sentry_dsn",
        "sentry_environment",
        "sentry_release",
        "sentry_traces_sample_rate",
        "sentry_profiles_sample_rate",
        "sentry_send_default_pii",
        "log_file",
        "backup_dir",
        "notepad_retention_days",
        "notepad_backup_keep",
        "personal_db_path",
        "enable_personal_features",
        "proactive_checkin_minutes",
    ):
        if key in raw:
            payload[key] = raw[key]

    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def _collect_constitution_texts(raw: dict[str, Any]) -> dict[str, str]:
    constitutions = _string_map(raw.get("constitutions"))
    if "constitution_restricted" in raw:
        constitutions["restricted"] = str(raw["constitution_restricted"])
    if "constitution_unrestricted" in raw:
        constitutions["unrestricted"] = str(raw["constitution_unrestricted"])
    return constitutions


def _normalize_mcp_servers(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        normalized: dict[str, Any] = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            payload = dict(entry)
            payload.pop("name", None)
            normalized[str(name)] = payload
        return normalized
    return {}


def _write_mcp_config(settings: Settings, bootstrap: BootstrapConfig) -> Path:
    path = settings.workspace_root / "mcp_servers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": bootstrap.mcp_servers,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_app(settings: Settings, bootstrap: BootstrapConfig):
    LOGGER.info("Loading configuration from %s", bootstrap.path)
    if bootstrap.mcp_servers:
        mcp_config_path = _write_mcp_config(settings, bootstrap)
        LOGGER.info("Wrote MCP server config to %s", mcp_config_path)
    app = create_app(settings)
    app.state.bootstrap_config = bootstrap
    app.state.constitution_texts = bootstrap.constitution_texts
    app.state.bootstrap_path = bootstrap.path
    return app


def parse_args() -> argparse.Namespace:
    default_host = _default_bind_host()
    default_port = _default_bind_port()
    parser = argparse.ArgumentParser(description="Start the LISA agent stack.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML bootstrap config."
    )
    parser.add_argument(
        "--host", default=default_host, help="Host for the FastAPI control plane."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port for the FastAPI control plane.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload for local development.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level.",
    )
    return parser.parse_args()


def _default_bind_host() -> str:
    env_host = os.environ.get("HOST") or os.environ.get("LISA_HOST")
    if env_host:
        return env_host
    return "127.0.0.1"


def _default_bind_port() -> int:
    raw_port = os.environ.get("PORT") or os.environ.get("LISA_PORT")
    if raw_port:
        try:
            return int(raw_port)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid PORT value %r; falling back to 8000", raw_port
            )
    return 8000


def _print_banner(settings: Settings, bootstrap: BootstrapConfig) -> None:
    hub = (
        f"{settings.message_hub_host}:{settings.message_hub_port}"
        if settings.message_hub_enabled
        else "disabled"
    )
    banner_lines = [
        "LISA boot sequence complete.",
        f"  config: {bootstrap.path}",
        f"  workspace: {settings.workspace_root}",
        f"  database: {settings.db_path}",
        f"  message hub: {hub}",
        f"  evolution: {'enabled' if settings.evolution_enabled else 'disabled'}",
    ]
    banner_lines.append(
        f"  admin api: {'configured' if settings.admin_api_token else 'not configured'}"
    )
    print("\n".join(banner_lines))


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _session_bootstrap_available(settings: Settings) -> bool:
    return bool(
        str(getattr(settings, "admin_api_token", "") or "").strip()
        or str(getattr(settings, "bot_security_key", "") or "").strip()
    )


def _warn_remote_binding(host: str, role: str) -> None:
    for _ in range(3):
        LOGGER.warning(
            "%s is binding to non-loopback host %s. Session auth bootstrap must remain enabled.",
            role,
            host,
        )


def _validate_network_exposure(settings: Settings, host: str) -> None:
    if not _is_loopback_host(host):
        if not settings.allow_remote_bind:
            raise RuntimeError(
                f"Refusing to bind the control plane to {host} without LISA_ALLOW_REMOTE_BIND=true."
            )
        if not _session_bootstrap_available(settings):
            raise RuntimeError(
                "Refusing remote control-plane bind because no admin token or bot security key is configured."
            )
        _warn_remote_binding(host, "Control plane")

    if settings.message_hub_enabled and settings.message_hub_start_listener:
        hub_host = str(settings.message_hub_host or "").strip() or "127.0.0.1"
        if not _is_loopback_host(hub_host):
            if not settings.allow_remote_bind:
                raise RuntimeError(
                    f"Refusing to bind the message hub listener to {hub_host} without LISA_ALLOW_REMOTE_BIND=true."
                )
            if not _session_bootstrap_available(settings):
                raise RuntimeError(
                    "Refusing remote message hub bind because no admin token or bot security key is configured."
                )
            _warn_remote_binding(hub_host, "Message hub")


def main() -> None:
    args = parse_args()

    apply_memory_limit()
    config_path = Path(args.config).resolve()
    settings, bootstrap = load_bootstrap_config(config_path)
    _validate_network_exposure(settings, args.host)
    process_lock = ProcessLock(
        settings.workspace_root / "data" / "locks" / "main.lock",
        role="main",
    )
    try:
        process_lock.acquire()
    except ProcessLockHeldError as exc:
        holder = exc.holder
        if holder is not None and holder.pid is not None:
            LOGGER.error(
                "Main process lock is already held by PID %s. Refusing to start a second instance.",
                holder.pid,
            )
        else:
            LOGGER.error("Main process lock is already held. Refusing to start.")
        sys.exit(1)
    key_vault_path = settings.workspace_root / "keys.enc"
    master_key = os.environ.get("LISA_MASTER_KEY") or os.environ.get(
        "LISA_KEYS_MASTER_KEY"
    )
    if master_key and key_vault_path.exists():
        try:
            encrypted_payload = load_api_keys(key_vault_path, master_key)
            interface_keys = dict(encrypted_payload.get("interface_keys") or {})
            if interface_keys:
                settings.interface_keys.update(
                    {str(key): str(value) for key, value in interface_keys.items()}
                )
            LOGGER.info("Loaded encrypted key vault from %s", key_vault_path)
        except Exception as exc:
            LOGGER.warning(
                "Unable to load encrypted key vault %s: %s", key_vault_path, exc
            )
    logger = configure_logging(settings.log_file, args.log_level)
    logger.info("Bootstrap config loaded from %s", config_path)
    if settings.bot_security_key:
        logger.warning(
            "LISA_BOT_SECURITY_KEY is deprecated for channel pairing but is still accepted for short-lived dashboard session bootstrap."
        )
    configure_sentry(settings, logger=logger)
    env_result = check_environment(settings, bootstrap)
    for warning in env_result.warnings:
        logger.warning(warning)
    if env_result.errors:
        for error in env_result.errors:
            logger.error(error)
        env_result.raise_if_failed()
    try:
        app = build_app(settings, bootstrap)
        _print_banner(settings, bootstrap)
        logger.info("Starting FastAPI control plane on %s:%s", args.host, args.port)
        asyncio.run(_serve(app, args.host, args.port, args.reload, args.log_level))
    finally:
        process_lock.release()


async def _serve(app, host: str, port: int, reload: bool, log_level: str) -> None:
    config = uvicorn.Config(
        app, host=host, port=port, reload=reload, log_level=log_level
    )
    server = uvicorn.Server(config)

    def _request_shutdown() -> None:
        server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: _request_shutdown())
        except Exception:
            pass

    try:
        await server.serve()
    finally:
        server.should_exit = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
