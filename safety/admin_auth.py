from __future__ import annotations

import secrets
from typing import Any

from fastapi import HTTPException, Request

ADMIN_AUTH_HEADERS = ("authorization", "x-admin-token")


def extract_admin_token(headers: Any) -> str | None:
    authorization = headers.get("authorization") or headers.get("Authorization")
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return token

    token = headers.get("x-admin-token") or headers.get("X-Admin-Token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def require_admin_request(
    request: Request, settings: Any, unsafe_only: bool = False
) -> None:
    expected = getattr(settings, "admin_api_token", None)
    if not expected:
        raise HTTPException(
            status_code=503, detail="Admin API token is not configured."
        )

    provided = extract_admin_token(request.headers)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Admin authorization failed.")

    if unsafe_only and not getattr(settings, "enable_unsafe_admin_endpoints", False):
        raise HTTPException(
            status_code=403, detail="Unsafe admin endpoints are disabled."
        )
