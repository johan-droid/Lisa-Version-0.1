from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StyleLearner:
    def infer_profile(self, interactions: list[dict[str, Any]]) -> dict[str, Any]:
        lines = [str(item.get("output") or "") for item in interactions if item.get("output")]
        return {
            "line_count": len(lines),
            "prefers_type_hints": any("->" in line for line in lines),
            "prefers_async": any("async def" in line for line in lines),
        }
