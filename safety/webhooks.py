from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlparse


@dataclass(slots=True)
class WebhookSecrets:
    telegram_secret: str | None = None
    slack_signing_secret: str | None = None
    whatsapp_auth_token: str | None = None


def verify_webhook(
    source: str,
    headers: dict[str, str],
    body: bytes,
    *,
    url: str,
    secrets: WebhookSecrets,
) -> None:
    if source == "telegram" and secrets.telegram_secret:
        provided = headers.get("X-Telegram-Bot-Api-Secret-Token") or headers.get("x-telegram-bot-api-secret-token")
        if provided != secrets.telegram_secret:
            raise PermissionError("Invalid Telegram webhook secret token.")

    if source == "slack" and secrets.slack_signing_secret:
        timestamp = headers.get("X-Slack-Request-Timestamp") or headers.get("x-slack-request-timestamp")
        signature = headers.get("X-Slack-Signature") or headers.get("x-slack-signature")
        if not timestamp or not signature:
            raise PermissionError("Slack signature headers are missing.")
        base = f"v0:{timestamp}:{body.decode('utf-8', errors='replace')}".encode("utf-8")
        expected = "v0=" + hmac.new(
            secrets.slack_signing_secret.encode("utf-8"),
            base,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise PermissionError("Invalid Slack signature.")

    if source == "whatsapp" and secrets.whatsapp_auth_token:
        signature = headers.get("X-Twilio-Signature") or headers.get("x-twilio-signature")
        if signature:
            parsed = urlparse(url)
            params = sorted(parse_qsl(parsed.query, keep_blank_values=True))
            payload = parsed.scheme + "://" + parsed.netloc + parsed.path + "".join(f"{k}{v}" for k, v in params)
            digest = hmac.new(
                secrets.whatsapp_auth_token.encode("utf-8"),
                payload.encode("utf-8"),
                hashlib.sha1,
            ).digest()
            expected = base64.b64encode(digest).decode("utf-8")
            if not hmac.compare_digest(expected, signature):
                raise PermissionError("Invalid WhatsApp signature.")
        else:
            token = headers.get("X-WhatsApp-Token") or headers.get("x-whatsapp-token")
            if token != secrets.whatsapp_auth_token:
                raise PermissionError("Invalid WhatsApp token.")


def secrets_from_mapping(mapping: dict[str, Any] | None) -> WebhookSecrets:
    mapping = mapping or {}
    return WebhookSecrets(
        telegram_secret=str(mapping.get("telegram_secret") or mapping.get("telegram_bot_token") or "") or None,
        slack_signing_secret=str(mapping.get("slack_signing_secret") or "") or None,
        whatsapp_auth_token=str(mapping.get("whatsapp_auth_token") or mapping.get("whatsapp_token") or "") or None,
    )
