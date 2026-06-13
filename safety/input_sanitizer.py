from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
SQL_NOSQL_PATTERNS = {
    r"(?:^|[\s(])union\s+select(?:[\s)])": "sql_union_select",
    r"(?:^|[\s(])or\s+1=1(?:[\s)])": "sql_boolean_bypass",
    r"(?:^|[\s(])drop\s+table(?:[\s)])": "sql_drop_table",
    r";\s*--": "sql_comment_terminator",
    r"\$where\b": "nosql_where_operator",
    r"\$regex\b": "nosql_regex_operator",
    r"\$ne\b": "nosql_ne_operator",
    r"\$gt\b|\$gte\b|\$lt\b|\$lte\b": "nosql_comparison_operator",
}
SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(sk-[a-z0-9]{12,})\b"),
    re.compile(r"(?i)\b(xox[baprs]-[a-z0-9-]{10,})\b"),
    re.compile(r"(?i)\b(gh[pousr]_[a-z0-9]{20,})\b"),
    re.compile(r"(?i)\b(bearer\s+)([a-z0-9._-]{16,})\b"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|secret|password)\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9._-]{10,}\.[a-zA-Z0-9._-]{10,}\b"),
)
INVISIBLE_TRANSLATION = {
    ord("\u200b"): None,  # zero width space
    ord("\u200c"): None,  # zero width non-joiner
    ord("\u200d"): None,  # zero width joiner
    ord("\ufeff"): None,  # byte order mark
    ord("\u2060"): None,  # word joiner
    ord("\u202a"): None,  # bidi controls
    ord("\u202b"): None,
    ord("\u202c"): None,
    ord("\u202d"): None,
    ord("\u202e"): None,
    ord("\u2066"): None,
    ord("\u2067"): None,
    ord("\u2068"): None,
    ord("\u2069"): None,
}
MAX_TEXT_LENGTH = 16_000
MAX_VISIBLE_TEXT_LENGTH = 4_000
MAX_BODY_BYTES = 262_144
MAX_LOG_STRING_LENGTH = 4_000
MAX_LOG_COLLECTION_ITEMS = 50
MAX_LOG_DEPTH = 5

HIGH_RISK_PATTERNS = {
    r"ignore\s+previous": "override_previous_instructions",
    r"disregard\s+all": "override_all_instructions",
    r"forget\s+(?:previous|all)": "forget_prior_context",
    r"reveal\s+your\s+system": "reveal_system_prompt",
    r"developer\s+(?:mode|message)": "developer_message_exfiltration",
    r"jailbreak": "jailbreak_request",
    r"system\s+prompt": "system_prompt_reference",
    r"print\s+initial\s+prompt": "initial_prompt_exfiltration",
    r"override\s+instructions": "instruction_override",
}

MEDIUM_RISK_PATTERNS = {
    r"prompt\s+injection": "mentions_prompt_injection",
    r"you\s+are\s+now": "role_reassignment",
    r"base64": "encoded_payload_reference",
    r"hex\s+decode": "decode_instruction_reference",
}


@dataclass(slots=True)
class SanitizationResult:
    text: str
    suspicious: bool
    risk_score: int
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryInspectionResult:
    text: str
    suspicious: bool
    reasons: list[str] = field(default_factory=list)


def sanitize_text(text: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = CONTROL_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(INVISIBLE_TRANSLATION)
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text


def sanitize_user_visible_text(
    text: str, *, max_length: int = MAX_VISIBLE_TEXT_LENGTH
) -> str:
    cleaned = sanitize_text(text, max_length=max_length * 2)
    redacted = redact_sensitive_text(cleaned)
    if len(redacted) > max_length:
        redacted = redacted[: max_length - 3].rstrip() + "..."
    return redacted


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_TEXT_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(
                lambda match: f"{match.group(1)}[redacted]", redacted
            )
        else:
            redacted = pattern.sub("[redacted]", redacted)
    return redacted


def sanitize_structure(
    value: Any,
    *,
    max_depth: int = MAX_LOG_DEPTH,
    max_items: int = MAX_LOG_COLLECTION_ITEMS,
    max_string_length: int = MAX_LOG_STRING_LENGTH,
) -> Any:
    if max_depth <= 0:
        return "[truncated]"

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                cleaned["__truncated__"] = f"{len(value) - max_items} more keys omitted"
                break
            key_text = sanitize_text(str(key), max_length=128)
            lowered = key_text.lower()
            if (
                lowered.endswith("_token")
                or lowered.endswith("_key")
                or lowered
                in {
                    "authorization",
                    "cookie",
                    "set-cookie",
                    "password",
                    "secret",
                    "api_key",
                    "token",
                }
            ):
                cleaned[key_text] = "[redacted]"
            else:
                cleaned[key_text] = sanitize_structure(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string_length=max_string_length,
                )
        return cleaned

    if isinstance(value, list):
        items = [
            sanitize_structure(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"[{len(value) - max_items} more items omitted]")
        return items

    if isinstance(value, tuple):
        return tuple(
            sanitize_structure(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in value[:max_items]
        )

    if isinstance(value, str):
        return sanitize_user_visible_text(value, max_length=max_string_length)

    return value


def ensure_body_size(body: bytes, *, max_bytes: int = MAX_BODY_BYTES) -> None:
    if len(body) > max_bytes:
        raise ValueError(f"Request body exceeded the {max_bytes} byte limit.")


def inspect_query_text(text: str) -> QueryInspectionResult:
    normalized = sanitize_text(text).lower()
    reasons: list[str] = []
    for pattern, reason in SQL_NOSQL_PATTERNS.items():
        if re.search(pattern, normalized):
            reasons.append(reason)
    return QueryInspectionResult(
        text=normalized, suspicious=bool(reasons), reasons=reasons
    )


def inspect_text(text: str) -> SanitizationResult:
    lowered = sanitize_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", lowered)
    reasons: list[str] = []
    risk_score = 0

    for pattern, reason in HIGH_RISK_PATTERNS.items():
        if re.search(pattern, cleaned) or re.search(pattern, lowered):
            reasons.append(reason)
            risk_score += 4

    for pattern, reason in MEDIUM_RISK_PATTERNS.items():
        if re.search(pattern, cleaned) or re.search(pattern, lowered):
            reasons.append(reason)
            risk_score += 1

    suspicious = risk_score >= 4 or (
        any(
            reason.startswith("override") or reason.endswith("exfiltration")
            for reason in reasons
        )
        and risk_score >= 3
    )
    return SanitizationResult(
        text=sanitize_text(text),
        suspicious=suspicious,
        risk_score=risk_score,
        reasons=reasons,
    )


def is_prompt_injection_suspicious(text: str) -> bool:
    return inspect_text(text).suspicious
