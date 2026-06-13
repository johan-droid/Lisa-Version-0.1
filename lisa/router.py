from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lisa.llm import LLMClient


@dataclass(slots=True)
class RouteDecision:
    task_type: str
    brain: str
    reason: str


class LLMRouter:
    """Routes tasks to the appropriate reasoning backend."""

    ROUTING_RULES = {
        "simple_qa": "tinyllama",
        "tool_use": "freellm_external",
        "code_gen": "freellm_external",
        "reflection": "freellm_external",
        "embedding": "local_embed_model",
        "evolution": "freellm_external",
    }

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def classify_task(
        self, task: str, context: dict[str, Any] | None = None
    ) -> str:
        lowered = task.lower().strip()
        context = context or {}
        if context.get("task_type") in self.ROUTING_RULES:
            return str(context["task_type"])
        if context.get("mode") == "embedding":
            return "embedding"
        if lowered.startswith(
            ("what is ", "who is ", "when is ", "where is ", "how many ", "explain ")
        ):
            return "simple_qa"
        if any(
            token in lowered
            for token in ("evolve", "improve skill", "rewrite skill", "nightly cycle")
        ):
            return "evolution"
        if any(
            token in lowered
            for token in (
                "reflect",
                "critique",
                "why did this fail",
                "replan",
                "analyze failure",
            )
        ):
            return "reflection"
        if any(
            token in lowered
            for token in (
                "write code",
                "generate code",
                "refactor",
                "function",
                "class ",
                "python",
                "fastapi",
                "sql",
                "regex",
            )
        ):
            return "code_gen"
        if any(
            token in lowered
            for token in (
                "read file",
                "write file",
                "search web",
                "browser",
                "tool",
                "terminal",
                "docker",
                "call api",
                "send message",
            )
        ):
            return "tool_use"
        if len(lowered.split()) <= 24 and not re.search(r"[{}[\]();]", lowered):
            return "simple_qa"

        local_guess = await self.llm_client.classify_with_local(
            labels=list(self.ROUTING_RULES.keys()),
            task=task,
            context=context,
        )
        if local_guess is not None:
            return local_guess
        return "tool_use"

    async def route(
        self, task: str, context: dict[str, Any] | None = None
    ) -> RouteDecision:
        task_type = await self.classify_task(task, context=context)
        brain = self.ROUTING_RULES.get(task_type, "freellm_external")
        return RouteDecision(
            task_type=task_type,
            brain=brain,
            reason=f"classified_as_{task_type}",
        )
