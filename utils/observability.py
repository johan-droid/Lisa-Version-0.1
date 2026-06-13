from __future__ import annotations

import logging
from typing import Any

from safety.input_sanitizer import sanitize_structure

LOGGER = logging.getLogger("lisa.observability")
_SENTRY_ENABLED = False

_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-admin-token",
    "x-slack-signature",
    "x-twilio-signature",
    "x-whatsapp-token",
    "x-telegram-bot-api-secret-token",
    "telegram_bot_token",
    "slack_bot_token",
    "whatsapp_auth_token",
    "freellmapi_api_key",
    "model_api_key",
}


def configure_sentry(settings: Any, logger: logging.Logger | None = None) -> bool:
    logger = logger or LOGGER
    dsn = getattr(settings, "sentry_dsn", None)
    if not dsn:
        logger.info("Sentry DSN not set; observability tracing disabled.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.aiohttp import AioHttpIntegration
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        logger.warning(
            "sentry-sdk or one of its integrations is unavailable. Skipping sentry initialization."
        )
        return False

    logging_integration = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )

    sentry_sdk.init(
        dsn=dsn,
        environment=getattr(settings, "sentry_environment", "production"),
        release=getattr(settings, "sentry_release", None),
        traces_sample_rate=float(getattr(settings, "sentry_traces_sample_rate", 0.0)),
        profiles_sample_rate=float(
            getattr(settings, "sentry_profiles_sample_rate", 0.0)
        ),
        send_default_pii=bool(getattr(settings, "sentry_send_default_pii", False)),
        integrations=[
            logging_integration,
            FastApiIntegration(),
            AioHttpIntegration(),
            AsyncioIntegration(),
        ],
        before_send=_before_send,
    )

    global _SENTRY_ENABLED
    _SENTRY_ENABLED = True
    logger.info(
        "Sentry initialized with FastAPI, aiohttp, asyncio, and logging integrations."
    )
    return True


def sentry_enabled() -> bool:
    return _SENTRY_ENABLED


def bind_runtime_context(**fields: Any) -> None:
    if not _SENTRY_ENABLED:
        return
    try:
        import sentry_sdk
    except ImportError:
        return

    with sentry_sdk.configure_scope() as scope:
        tags = {}
        context = {}
        for key, value in fields.items():
            if value is None:
                continue
            if key in {"request_id", "session_id", "source", "channel", "job_id"}:
                tags[key] = str(value)
            else:
                context[key] = _scrub_value(value)
        for key, value in tags.items():
            scope.set_tag(key, value)
        if context:
            scope.set_context("lisa", context)


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    return _scrub_value(event)


def _scrub_value(value: Any) -> Any:
    cleaned = sanitize_structure(value)
    if isinstance(cleaned, dict):
        for key in list(cleaned):
            key_text = str(key).lower()
            if (
                key_text in _SENSITIVE_KEYS
                or key_text.endswith("_token")
                or key_text.endswith("_key")
            ):
                cleaned[key] = "[redacted]"
    return cleaned
