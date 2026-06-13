from __future__ import annotations

import hashlib
import math
from typing import Iterable

EMBEDDING_DIMS = 1536


def deterministic_embedding(text: str, dims: int = EMBEDDING_DIMS) -> list[float]:
    seed = (text or "").encode("utf-8", errors="replace")
    if not seed:
        seed = b"lisa"
    values: list[float] = []
    counter = 0
    while len(values) < dims:
        digest = hashlib.sha256(
            seed + counter.to_bytes(4, "big", signed=False)
        ).digest()
        for index in range(0, len(digest), 4):
            chunk = digest[index : index + 4]
            if len(chunk) < 4:
                chunk = chunk.ljust(4, b"\x00")
            integer = int.from_bytes(chunk, "big", signed=False)
            values.append(((integer / 0xFFFFFFFF) * 2.0) - 1.0)
            if len(values) >= dims:
                break
        counter += 1

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right_values)) or 1.0
    return dot / (left_norm * right_norm)
