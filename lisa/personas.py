from __future__ import annotations

from enum import StrEnum


class Persona(StrEnum):
    ARCHITECT = "architect"
    ORACLE = "oracle"
    GUARDIAN = "guardian"
    EVOLUTION_ENGINE = "evolution_engine"
    DISTRIBUTED_MIND = "distributed_mind"


PERSONA_KEYWORDS: dict[Persona, tuple[str, ...]] = {
    Persona.ARCHITECT: (
        "build",
        "implement",
        "design",
        "api",
        "endpoint",
        "scaffold",
        "plan",
        "refactor",
    ),
    Persona.ORACLE: (
        "security",
        "audit",
        "bug",
        "vulnerability",
        "analyze",
        "profile",
        "review",
    ),
    Persona.GUARDIAN: (
        "monitor",
        "backup",
        "health",
        "infra",
        "uptime",
        "alert",
        "recover",
    ),
    Persona.EVOLUTION_ENGINE: (
        "improve",
        "learn",
        "optimize",
        "evolve",
        "skill",
        "adapt",
    ),
    Persona.DISTRIBUTED_MIND: (
        "team",
        "coordinate",
        "sync",
        "communicate",
        "handoff",
        "consensus",
    ),
}


def infer_persona_blend(message: str) -> dict[str, float]:
    lowered = message.lower()
    scores: dict[Persona, float] = {persona: 1.0 for persona in Persona}

    for persona, keywords in PERSONA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[persona] += 0.8

    total = sum(scores.values())
    normalized = {
        persona.value: round(score / total, 3) for persona, score in scores.items()
    }

    remainder = 1.0 - sum(normalized.values())
    normalized[Persona.ARCHITECT.value] = round(
        normalized[Persona.ARCHITECT.value] + remainder, 3
    )
    return normalized
