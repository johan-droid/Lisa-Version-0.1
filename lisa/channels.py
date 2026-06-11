from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx


ChannelName = Literal["direct", "telegram", "slack", "whatsapp"]


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


@dataclass(slots=True)
class ChannelCredentials:
    telegram_bot_token: str | None = None
    telegram_default_chat_id: str | None = None
    slack_bot_token: str | None = None
    slack_default_channel: str | None = None
    whatsapp_account_sid: str | None = None
    whatsapp_auth_token: str | None = None
    whatsapp_from_number: str | None = None
    whatsapp_default_to: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "ChannelCredentials":
        mapping = mapping or {}
        return cls(
            telegram_bot_token=_clean(mapping.get("telegram_bot_token") or mapping.get("telegram_token")),
            telegram_default_chat_id=_clean(
                mapping.get("telegram_default_chat_id") or mapping.get("telegram_chat_id")
            ),
            slack_bot_token=_clean(mapping.get("slack_bot_token") or mapping.get("slack_token")),
            slack_default_channel=_clean(
                mapping.get("slack_default_channel") or mapping.get("slack_channel")
            ),
            whatsapp_account_sid=_clean(mapping.get("whatsapp_account_sid") or mapping.get("twilio_account_sid")),
            whatsapp_auth_token=_clean(mapping.get("whatsapp_auth_token") or mapping.get("twilio_auth_token")),
            whatsapp_from_number=_clean(mapping.get("whatsapp_from_number") or mapping.get("twilio_from_number")),
            whatsapp_default_to=_clean(mapping.get("whatsapp_default_to") or mapping.get("whatsapp_to")),
        )

    def configured_channels(self) -> list[str]:
        configured: list[str] = []
        if self.telegram_bot_token:
            configured.append("telegram")
        if self.slack_bot_token:
            configured.append("slack")
        if self.whatsapp_account_sid and self.whatsapp_auth_token and self.whatsapp_from_number:
            configured.append("whatsapp")
        configured.append("direct")
        return configured


@dataclass(slots=True)
class ChannelDispatchResult:
    channel: str
    delivered: bool
    message: str
    payload: dict[str, Any] = None  # type: ignore[assignment]
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "delivered": self.delivered,
            "message": self.message,
            "payload": self.payload or {},
            "detail": self.detail,
        }


class ChannelGateway:
    def __init__(self, credentials: ChannelCredentials | None = None, timeout_seconds: int = 30):
        self.credentials = credentials or ChannelCredentials()
        self.timeout_seconds = timeout_seconds

    def capabilities(self) -> dict[str, Any]:
        return {
            "configured_channels": self.credentials.configured_channels(),
            "supports": {
                "telegram": bool(self.credentials.telegram_bot_token),
                "slack": bool(self.credentials.slack_bot_token),
                "whatsapp": bool(
                    self.credentials.whatsapp_account_sid
                    and self.credentials.whatsapp_auth_token
                    and self.credentials.whatsapp_from_number
                ),
            },
            "features": [
                "direct_response_api",
                "telegram_inline_keyboards",
                "telegram_callback_queries",
                "telegram_typing_indicators",
                "slack_post_message",
                "whatsapp_twilio_send",
            ],
        }

    async def send_typing(self, channel: ChannelName, user_id: str | None = None) -> dict[str, Any]:
        if channel != "telegram":
            return {"channel": channel, "delivered": False, "detail": "typing indicators are only supported for Telegram"}
        if not self.credentials.telegram_bot_token:
            return {"channel": channel, "delivered": False, "detail": "telegram bot token not configured"}
        chat_id = user_id or self.credentials.telegram_default_chat_id
        if not chat_id:
            return {"channel": channel, "delivered": False, "detail": "telegram chat id missing"}
        payload = {"chat_id": chat_id, "action": "typing"}
        return await self._telegram_api("sendChatAction", payload)

    async def answer_callback(self, callback_query_id: str, text: str | None = None, show_alert: bool = False) -> dict[str, Any]:
        if not self.credentials.telegram_bot_token:
            return {"channel": "telegram", "delivered": False, "detail": "telegram bot token not configured"}
        payload: dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text:
            payload["text"] = text
        return await self._telegram_api("answerCallbackQuery", payload)

    async def send_response(
        self,
        channel: ChannelName,
        *,
        user_id: str | None,
        text: str,
        parse_mode: str | None = None,
        reply_to_message_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        delivery_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        delivery_hints = delivery_hints or {}

        if channel == "direct":
            return {
                "channel": channel,
                "delivered": False,
                "message": text,
                "detail": "direct channel returns the response to the caller",
            }

        if channel == "telegram":
            return await self._send_telegram(
                text=text,
                chat_id=user_id,
                parse_mode=parse_mode or delivery_hints.get("parse_mode"),
                reply_to_message_id=reply_to_message_id or metadata.get("reply_to_message_id"),
                reply_markup=delivery_hints.get("reply_markup"),
                callback_query_id=metadata.get("telegram_callback_query_id"),
            )

        if channel == "slack":
            return await self._send_slack(
                text=text,
                channel_id=user_id,
                metadata=metadata,
                delivery_hints=delivery_hints,
            )

        if channel == "whatsapp":
            return await self._send_whatsapp(
                text=text,
                to=user_id,
                metadata=metadata,
            )

        return {
            "channel": channel,
            "delivered": False,
            "message": text,
            "detail": f"unsupported channel: {channel}",
        }

    async def _send_telegram(
        self,
        *,
        text: str,
        chat_id: str | None,
        parse_mode: str | None,
        reply_to_message_id: int | None,
        reply_markup: dict[str, Any] | None,
        callback_query_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.credentials.telegram_bot_token:
            return {"channel": "telegram", "delivered": False, "message": text, "detail": "telegram bot token not configured"}

        target_chat_id = chat_id or self.credentials.telegram_default_chat_id
        if not target_chat_id:
            return {"channel": "telegram", "delivered": False, "message": text, "detail": "telegram chat id missing"}

        if callback_query_id:
            await self.answer_callback(callback_query_id, text=None)

        payload: dict[str, Any] = {
            "chat_id": target_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if len(text) > 3500:
            payload["text"] = text[:3500] + "..."

        return await self._telegram_api("sendMessage", payload)

    async def _send_slack(
        self,
        *,
        text: str,
        channel_id: str | None,
        metadata: dict[str, Any],
        delivery_hints: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.credentials.slack_bot_token:
            return {"channel": "slack", "delivered": False, "message": text, "detail": "slack bot token not configured"}

        target_channel = channel_id or self.credentials.slack_default_channel
        if not target_channel:
            return {"channel": "slack", "delivered": False, "message": text, "detail": "slack channel missing"}

        payload: dict[str, Any] = {
            "channel": target_channel,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        thread_ts = metadata.get("thread_ts") or delivery_hints.get("thread_ts")
        if thread_ts:
            payload["thread_ts"] = thread_ts
        blocks = delivery_hints.get("blocks")
        if blocks:
            payload["blocks"] = blocks

        headers = {
            "Authorization": f"Bearer {self.credentials.slack_bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        delivered = bool(body.get("ok"))
        return {
            "channel": "slack",
            "delivered": delivered,
            "message": text,
            "payload": body,
            "detail": None if delivered else body.get("error", "slack delivery failed"),
        }

    async def _send_whatsapp(
        self,
        *,
        text: str,
        to: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not (
            self.credentials.whatsapp_account_sid
            and self.credentials.whatsapp_auth_token
            and self.credentials.whatsapp_from_number
        ):
            return {
                "channel": "whatsapp",
                "delivered": False,
                "message": text,
                "detail": "twilio credentials not configured",
            }

        target_to = to or self.credentials.whatsapp_default_to
        if not target_to:
            return {"channel": "whatsapp", "delivered": False, "message": text, "detail": "whatsapp recipient missing"}

        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/{self.credentials.whatsapp_account_sid}/Messages.json"
        )
        payload = {
            "From": self.credentials.whatsapp_from_number,
            "To": target_to,
            "Body": text[:1500],
        }
        if metadata.get("media_url"):
            payload["MediaUrl"] = metadata["media_url"]

        auth = (self.credentials.whatsapp_account_sid, self.credentials.whatsapp_auth_token)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, auth=auth) as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            body = response.json()

        return {
            "channel": "whatsapp",
            "delivered": True,
            "message": text,
            "payload": body,
            "detail": None,
        }

    async def _telegram_api(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.credentials.telegram_bot_token}/{method}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
        return {
            "channel": "telegram",
            "delivered": bool(body.get("ok")),
            "message": payload.get("text", ""),
            "payload": body,
            "detail": None if body.get("ok") else body.get("description", "telegram delivery failed"),
        }

