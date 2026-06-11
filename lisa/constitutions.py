from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConstitutionMode(StrEnum):
    RESTRICTED = "restricted"
    UNRESTRICTED = "unrestricted"


ENABLE_UNRESTRICTED_PREFIX = "ENABLE UNRESTRICTED MODE"
DISABLE_UNRESTRICTED_PREFIX = "DISABLE UNRESTRICTED MODE"


@dataclass(slots=True)
class ConstitutionCommand:
    target_mode: ConstitutionMode
    reason: str | None = None


def parse_constitution_command(message: str) -> ConstitutionCommand | None:
    stripped = message.strip()
    upper = stripped.upper()
    if upper.startswith(ENABLE_UNRESTRICTED_PREFIX):
        reason = stripped[len(ENABLE_UNRESTRICTED_PREFIX) :].strip(" []:-")
        return ConstitutionCommand(
            target_mode=ConstitutionMode.UNRESTRICTED,
            reason=reason or None,
        )
    if upper.startswith(DISABLE_UNRESTRICTED_PREFIX):
        return ConstitutionCommand(target_mode=ConstitutionMode.RESTRICTED)
    return None
