from __future__ import annotations

import re


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SUSPICIOUS_PATTERNS = (
    r"ignore\s+previous\s+instructions",
    r"reveal\s+your\s+system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"prompt\s+injection",
    r"system\s+prompt",
)


def sanitize_text(text: str) -> str:
    text = CONTROL_CHARS.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_prompt_injection_suspicious(text: str) -> bool:
    lowered = sanitize_text(text).lower()
    return any(re.search(pattern, lowered) for pattern in SUSPICIOUS_PATTERNS)
