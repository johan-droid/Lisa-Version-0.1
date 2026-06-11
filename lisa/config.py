from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LISA"
    workspace_root: Path = Field(default_factory=Path.cwd)
    db_path: Path = Path("data/lisa_notepad.db")
    skills_dir: Path = Path("skills")
    persona_vectors_path: Path = Path("data/persona_vectors.npz")
    gating_model_path: Path = Path("data/gating_model.pkl")
    local_model_path: Path | None = None
    local_model_context_size: int = 2048
    local_model_n_threads: int | None = None
    local_model_n_gpu_layers: int = 0
    docker_image: str = "python:3.12-slim"
    freellmapi_base_url: str | None = None
    freellmapi_api_key: str | None = None
    freellmapi_default_provider: str | None = None
    freellmapi_timeout_seconds: int = 30
    message_hub_enabled: bool = True
    message_hub_host: str = "localhost"
    message_hub_port: int = 8800
    evolution_enabled: bool = True
    autonomous_enabled: bool = True
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
    model_provider: str | None = None
    model_name: str | None = None
    model_base_url: str | None = None
    model_api_key: str | None = None
    external_timeout_seconds: int = 30
    enable_browser_tools: bool = True
    bot_security_key: str | None = None
    telegram_bot_token: str | None = None
    whatsapp_bot_token: str | None = None
    slack_bot_token: str | None = None
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
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "local_model_path",
        "local_model_n_threads",
        "freellmapi_base_url",
        "freellmapi_api_key",
        "freellmapi_default_provider",
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

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        self.workspace_root = self.workspace_root.resolve()
        if not self.db_path.is_absolute():
            self.db_path = (self.workspace_root / self.db_path).resolve()
        if not self.skills_dir.is_absolute():
            self.skills_dir = (self.workspace_root / self.skills_dir).resolve()
        if not self.persona_vectors_path.is_absolute():
            self.persona_vectors_path = (self.workspace_root / self.persona_vectors_path).resolve()
        if not self.gating_model_path.is_absolute():
            self.gating_model_path = (self.workspace_root / self.gating_model_path).resolve()
        if not self.evolution_staging_dir.is_absolute():
            self.evolution_staging_dir = (self.workspace_root / self.evolution_staging_dir).resolve()
        if not self.log_file.is_absolute():
            self.log_file = (self.workspace_root / self.log_file).resolve()
        if not self.backup_dir.is_absolute():
            self.backup_dir = (self.workspace_root / self.backup_dir).resolve()
        if not self.personal_db_path.is_absolute():
            self.personal_db_path = (self.workspace_root / self.personal_db_path).resolve()
        if self.local_model_path is not None and not self.local_model_path.is_absolute():
            self.local_model_path = (self.workspace_root / self.local_model_path).resolve()
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
