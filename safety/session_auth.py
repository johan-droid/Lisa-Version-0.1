from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request, Response, WebSocket, status
from starlette.websockets import WebSocketState

from utils.snapshot import get_hmac_key

SESSION_COOKIE_NAME = "lisa_session"
SESSION_HEADER_NAME = "x-lisa-session"
DEFAULT_SESSION_TTL_SECONDS = 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    token_hash: str
    scope: str
    expires_at: datetime
    client_host: str | None
    user_agent: str | None


class SessionAuthManager:
    def __init__(self, settings: Any, ttl_seconds: int | None = None) -> None:
        self.settings = settings
        self.ttl_seconds = max(
            30,
            int(
                ttl_seconds
                or getattr(settings, "session_token_ttl_seconds", 0)
                or DEFAULT_SESSION_TTL_SECONDS
            ),
        )
        self._key = get_hmac_key(settings)
        self._records: dict[str, SessionRecord] = {}

    def issue_session(
        self,
        *,
        scope: str,
        client_host: str | None,
        user_agent: str | None,
        ttl_seconds: int | None = None,
    ) -> tuple[str, SessionRecord]:
        self._purge_expired()
        session_id = secrets.token_urlsafe(18)
        nonce = secrets.token_urlsafe(24)
        expires_at = _utcnow() + timedelta(
            seconds=max(30, int(ttl_seconds or self.ttl_seconds))
        )
        signature = self._sign(session_id, nonce, expires_at)
        token = f"{session_id}.{nonce}.{signature}"
        record = SessionRecord(
            session_id=session_id,
            token_hash=self._token_hash(token),
            scope=scope,
            expires_at=expires_at,
            client_host=client_host,
            user_agent=user_agent,
        )
        self._records[session_id] = record
        return token, record

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        session_id = self._session_id(token)
        if session_id:
            self._records.pop(session_id, None)

    def verify_request(self, request: Request, *, scope: str) -> SessionRecord:
        return self._verify_token(
            _extract_token(request.headers, request.cookies),
            scope=scope,
            client_host=_request_client_host(request),
            user_agent=_request_user_agent(request),
        )

    def verify_websocket(self, websocket: WebSocket, *, scope: str) -> SessionRecord:
        token = _extract_token(websocket.headers, websocket.cookies, websocket.query_params)
        return self._verify_token(
            token,
            scope=scope,
            client_host=_websocket_client_host(websocket),
            user_agent=_request_user_agent(websocket),
        )

    def verify_aiohttp_request(self, request: Any, *, scope: str) -> SessionRecord:
        token = _extract_aiohttp_token(request)
        client_host = None
        remote = getattr(request, "remote", None)
        if isinstance(remote, str) and remote.strip():
            client_host = remote.strip()
        user_agent = None
        headers = getattr(request, "headers", {})
        raw_user_agent = headers.get("User-Agent") or headers.get("user-agent")
        if isinstance(raw_user_agent, str) and raw_user_agent.strip():
            user_agent = raw_user_agent.strip()
        return self._verify_token(
            token,
            scope=scope,
            client_host=client_host,
            user_agent=user_agent,
        )

    def attach_cookie(self, response: Response, token: str, record: SessionRecord) -> None:
        max_age = max(1, int((record.expires_at - _utcnow()).total_seconds()))
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            max_age=max_age,
            expires=max_age,
            httponly=True,
            secure=False,
            samesite="strict",
            path="/",
        )

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")

    def is_credential_valid(self, credential: str | None) -> bool:
        if not credential:
            return False
        expected_values = [
            str(value).strip()
            for value in (
                getattr(self.settings, "admin_api_token", None),
                getattr(self.settings, "bot_security_key", None),
            )
            if isinstance(value, str) and value.strip()
        ]
        return any(
            secrets.compare_digest(credential, expected) for expected in expected_values
        )

    def has_bootstrap_credential(self) -> bool:
        return bool(
            str(getattr(self.settings, "admin_api_token", "") or "").strip()
            or str(getattr(self.settings, "bot_security_key", "") or "").strip()
        )

    def _verify_token(
        self,
        token: str | None,
        *,
        scope: str,
        client_host: str | None,
        user_agent: str | None,
    ) -> SessionRecord:
        self._purge_expired()
        if not token:
            raise HTTPException(status_code=401, detail="Session token is required.")

        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Session token is invalid.")
        session_id, nonce, signature = parts
        record = self._records.get(session_id)
        if record is None:
            raise HTTPException(status_code=401, detail="Session token is unknown.")
        if record.scope != scope:
            raise HTTPException(status_code=403, detail="Session scope is invalid.")
        if record.expires_at <= _utcnow():
            self._records.pop(session_id, None)
            raise HTTPException(status_code=401, detail="Session token has expired.")

        expected_signature = self._sign(session_id, nonce, record.expires_at)
        if not secrets.compare_digest(signature, expected_signature):
            raise HTTPException(status_code=401, detail="Session token is invalid.")
        if not secrets.compare_digest(self._token_hash(token), record.token_hash):
            raise HTTPException(status_code=401, detail="Session token is invalid.")

        if record.client_host and client_host and record.client_host != client_host:
            raise HTTPException(status_code=401, detail="Session client mismatch.")
        if record.user_agent and user_agent and record.user_agent != user_agent:
            raise HTTPException(status_code=401, detail="Session client mismatch.")
        return record

    def _purge_expired(self) -> None:
        now = _utcnow()
        stale = [
            session_id
            for session_id, record in self._records.items()
            if record.expires_at <= now
        ]
        for session_id in stale:
            self._records.pop(session_id, None)

    def _sign(self, session_id: str, nonce: str, expires_at: datetime) -> str:
        payload = f"{session_id}:{nonce}:{_isoformat(expires_at)}".encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _session_id(token: str) -> str | None:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return None
        return parts[0]


def require_session_request(
    request: Request, session_auth: SessionAuthManager, *, scope: str = "dashboard"
) -> SessionRecord:
    return session_auth.verify_request(request, scope=scope)


async def reject_websocket(
    websocket: WebSocket,
    detail: str,
    code: int = status.WS_1008_POLICY_VIOLATION,
) -> None:
    if websocket.application_state == WebSocketState.CONNECTING:
        await websocket.close(code=code, reason=detail)
    else:
        await websocket.close(code=code)


def _request_client_host(request: Any) -> str | None:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if isinstance(host, str) and host.strip():
        return host.strip()
    return None


def _websocket_client_host(websocket: WebSocket) -> str | None:
    client = getattr(websocket, "client", None)
    host = getattr(client, "host", None)
    if isinstance(host, str) and host.strip():
        return host.strip()
    return None


def _request_user_agent(request: Any) -> str | None:
    headers = getattr(request, "headers", {})
    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    if isinstance(user_agent, str) and user_agent.strip():
        return user_agent.strip()
    return None


def _extract_token(
    headers: Any,
    cookies: Any,
    query_params: Any | None = None,
) -> str | None:
    authorization = headers.get("authorization") or headers.get("Authorization")
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return token

    header_token = headers.get(SESSION_HEADER_NAME) or headers.get(
        SESSION_HEADER_NAME.title()
    )
    if isinstance(header_token, str) and header_token.strip():
        return header_token.strip()

    cookie_token = cookies.get(SESSION_COOKIE_NAME) if cookies is not None else None
    if isinstance(cookie_token, str) and cookie_token.strip():
        return cookie_token.strip()

    if query_params is not None:
        query_token = query_params.get("session")
        if isinstance(query_token, str) and query_token.strip():
            return query_token.strip()
    return None


def _extract_aiohttp_token(request: Any) -> str | None:
    headers = getattr(request, "headers", {})
    rel_url = getattr(request, "rel_url", None)
    query = getattr(rel_url, "query", None)
    cookies = getattr(request, "cookies", {})
    return _extract_token(headers, cookies, query)
