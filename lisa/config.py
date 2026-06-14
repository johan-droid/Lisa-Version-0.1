from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import sys
import json
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import NoDecode
from pydantic_settings import BaseSettings, SettingsConfigDict

UserIdList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    app_name: str = "LISA"
    agent_id: str = "lisa"
    workspace_root: Path = Field(default_factory=Path.cwd)
    db_path: Path = Path("data/lisa_notepad.db")
    redis_url: str | None = None
    postgres_dsn: str | None = None
    chroma_persist_dir: Path = Path("data/chroma")
    working_memory_ttl_seconds: int = 7200
    skills_dir: Path = Path("skills")
    evolution_artifacts_dir: Path = Path("data/evolution_artifacts")
    persona_vectors_path: Path = Path("data/persona_vectors.npz")
    gating_model_path: Path = Path("data/gating_model.pkl")
    session_token_ttl_seconds: int = 300
    local_model_path: Path | None = None
    local_model_context_size: int = 2048
    local_model_n_threads: int | None = None
    local_model_n_gpu_layers: int = 0
    docker_image: str = "python:3.12-slim"
    allow_local_terminal_fallback: bool = False
    freellmapi_base_url: str | None = None
    freellmapi_api_key: str | None = None
    freellmapi_default_provider: str | None = None
    freellmapi_chat_url: str | None = None
    freellmapi_embeddings_url: str | None = None
    freellmapi_timeout_seconds: int = 120
    message_hub_enabled: bool = True
    message_hub_host: str = "localhost"
    message_hub_port: int = 8800
    message_hub_start_listener: bool = False
    evolution_enabled: bool = True
    autonomous_enabled: bool = False
    evolution_check_interval_seconds: int = 1800
    evolution_window_start_hour: int = 3
    evolution_window_duration_hours: int = 2
    evolution_idle_after_seconds: int = 3600
    evolution_candidate_limit: int = 6
    evolution_browser_queries: int = 2
    evolution_min_reward: float = 0.35
    evolution_staging_dir: Path = Path("data/evolution")
    incoming_queue_size: int = 256
    tool_timeout_seconds: int = 30
    brain_timeout_seconds: int = 180
    max_request_body_bytes: int = 262_144
    max_message_chars: int = 16_000
    hybrid_brain_enabled: bool = False
    hybrid_brain_stress_threshold: int = 4
    hybrid_brain_prompt_chars_threshold: int = 600
    hybrid_brain_race_window_ms: int = 250
    evolution_skill_autoload_limit: int = 2
    model_provider: str | None = None
    model_name: str | None = None
    model_base_url: str | None = None
    model_api_key: str | None = None
    external_timeout_seconds: int = 120
    enable_browser_tools: bool = True
    admin_api_token: str | None = None
    allow_remote_bind: bool = False
    enable_unsafe_admin_endpoints: bool = False
    enable_legacy_bot_pairing: bool = False
    bot_security_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_webhook_secret: str | None = None
    whatsapp_bot_token: str | None = None
    slack_bot_token: str | None = None
    telegram_allowed_user_ids: UserIdList = Field(default_factory=list)
    slack_allowed_user_ids: UserIdList = Field(default_factory=list)
    whatsapp_allowed_user_ids: UserIdList = Field(default_factory=list)
    interface_keys: dict[str, str] = Field(default_factory=dict)
    freellmapi_requests_per_minute: int = 60
    freellmapi_tokens_per_minute: int = 60_000
    sentry_dsn: str | None = None
    sentry_environment: str | None = None
    sentry_release: str | None = None
    sentry_traces_sample_rate: float = 0.0
    sentry_profiles_sample_rate: float = 0.0
    sentry_send_default_pii: bool = False
    log_file: Path = Path("logs/lisa.log")
    backup_dir: Path = Path("backups")
    notepad_retention_days: int = 30
    notepad_backup_keep: int = 3
    personal_db_path: Path = Path("data/personal.db")
    enable_personal_features: bool = True
    proactive_checkin_minutes: int = 120

    model_config = SettingsConfigDict(
        env_prefix="LISA_",
        env_file=(
            None
            if ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules)
            else (".env", ".env.local")
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "local_model_path",
        "local_model_n_threads",
        "freellmapi_base_url",
        "freellmapi_api_key",
        "freellmapi_default_provider",
        "freellmapi_chat_url",
        "freellmapi_embeddings_url",
        "admin_api_token",
        "redis_url",
        "postgres_dsn",
        "sentry_dsn",
        "sentry_environment",
        "sentry_release",
        mode="before",
    )
    @classmethod
    def _blank_optional_fields(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator(
        "telegram_allowed_user_ids",
        "slack_allowed_user_ids",
        "whatsapp_allowed_user_ids",
        mode="before",
    )
    @classmethod
    def _normalize_allowed_ids(cls, value: object) -> object:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(value, (tuple, set, list)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        self.workspace_root = self.workspace_root.resolve()
        if not self.db_path.is_absolute():
            self.db_path = (self.workspace_root / self.db_path).resolve()
        if not self.skills_dir.is_absolute():
            self.skills_dir = (self.workspace_root / self.skills_dir).resolve()
        if not self.chroma_persist_dir.is_absolute():
            self.chroma_persist_dir = (
                self.workspace_root / self.chroma_persist_dir
            ).resolve()
        if not self.evolution_artifacts_dir.is_absolute():
            self.evolution_artifacts_dir = (
                self.workspace_root / self.evolution_artifacts_dir
            ).resolve()
        if not self.persona_vectors_path.is_absolute():
            self.persona_vectors_path = (
                self.workspace_root / self.persona_vectors_path
            ).resolve()
        if not self.gating_model_path.is_absolute():
            self.gating_model_path = (
                self.workspace_root / self.gating_model_path
            ).resolve()
        if not self.evolution_staging_dir.is_absolute():
            self.evolution_staging_dir = (
                self.workspace_root / self.evolution_staging_dir
            ).resolve()
        if not self.log_file.is_absolute():
            self.log_file = (self.workspace_root / self.log_file).resolve()
        if not self.backup_dir.is_absolute():
            self.backup_dir = (self.workspace_root / self.backup_dir).resolve()
        if not self.personal_db_path.is_absolute():
            self.personal_db_path = (
                self.workspace_root / self.personal_db_path
            ).resolve()
        if (
            self.local_model_path is not None
            and not self.local_model_path.is_absolute()
        ):
            self.local_model_path = (
                self.workspace_root / self.local_model_path
            ).resolve()
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
