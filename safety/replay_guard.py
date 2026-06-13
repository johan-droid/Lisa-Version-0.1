from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


class ReplayAttackDetected(PermissionError):
    """Raised when a webhook or provider event is replayed within the cache TTL."""


@dataclass(slots=True)
class ReplayGuard:
    ttl_seconds: int = 600
    max_entries: int = 4096
    _seen: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def check_webhook(
        self,
        *,
        source: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        key = self._build_webhook_key(
            source=source, payload=payload, headers=headers, body=body
        )
        if key is None:
            return
        await self.register_or_raise(key)

    async def check_message(
        self, *, source: str, user_id: str, message_id: str | None
    ) -> None:
        if not message_id:
            return
        await self.register_or_raise(f"message:{source}:{user_id}:{message_id}")

    async def register_or_raise(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            if key in self._seen:
                raise ReplayAttackDetected(
                    "Replay attack detected for a previously processed event."
                )
            self._seen[key] = now + float(self.ttl_seconds)

    def _prune(self, now: float) -> None:
        expired = [key for key, expiry in self._seen.items() if expiry <= now]
        for key in expired:
            self._seen.pop(key, None)
        if len(self._seen) <= self.max_entries:
            return
        overflow = len(self._seen) - self.max_entries
        for key, _ in sorted(self._seen.items(), key=lambda item: item[1])[:overflow]:
            self._seen.pop(key, None)

    @staticmethod
    def _build_webhook_key(
        *,
        source: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        lowered_headers = {
            str(key).lower(): str(value) for key, value in headers.items()
        }
        if source == "telegram":
            update_id = payload.get("update_id")
            if update_id is not None:
                return f"webhook:telegram:update:{update_id}"
            callback = (
                payload.get("callback_query")
                if isinstance(payload.get("callback_query"), dict)
                else {}
            )
            callback_id = callback.get("id")
            if callback_id:
                return f"webhook:telegram:callback:{callback_id}"
        if source == "slack":
            event_id = payload.get("event_id") or lowered_headers.get(
                "x-slack-request-timestamp"
            )
            signature = lowered_headers.get("x-slack-signature")
            if event_id and signature:
                return f"webhook:slack:{event_id}:{signature}"
        if source == "whatsapp":
            for key in ("MessageSid", "SmsSid", "message_sid", "message_id"):
                value = payload.get(key)
                if value:
                    return f"webhook:whatsapp:{value}"
        event_id = (
            payload.get("message_id")
            or payload.get("id")
            or lowered_headers.get("x-idempotency-key")
        )
        if event_id:
            return f"webhook:{source}:{event_id}"
        if body and any(
            key in lowered_headers
            for key in ("x-slack-signature", "x-twilio-signature")
        ):
            digest = hashlib.sha256(body).hexdigest()
            return f"webhook:{source}:body:{digest}"
        return None
