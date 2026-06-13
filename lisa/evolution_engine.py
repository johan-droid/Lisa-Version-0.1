from __future__ import annotations

import json
import time
from typing import Any

from lisa.events import EventBus, LisaEvent


class EvolutionEngine:
    def __init__(
        self,
        *,
        memory: Any,
        llm_client: Any,
        tool_executor: Any,
        event_bus: EventBus,
        agent_id: str,
    ) -> None:
        self.memory = memory
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.agent_id = agent_id
        self._failure_counts: dict[str, int] = {}

    async def run_nightly_cycle(self) -> dict[str, Any]:
        weak_skills = await self.get_low_performing_skills(threshold=0.6)
        results: list[dict[str, Any]] = []
        for skill in weak_skills:
            failure_episodes = await self.get_failure_episodes(skill["skill_name"])
            success_episodes = await self.get_success_episodes(skill["skill_name"])
            candidate = await self.evolve_skill(
                skill=skill,
                failure_episodes=failure_episodes,
                success_episodes=success_episodes,
            )
            test_result = await self.test_skill_candidate(
                skill["skill_name"], candidate
            )
            score = self.score_skill(test_result)
            accepted = score > float(skill.get("success_rate", 0.0))
            if accepted:
                metadata = dict(skill.get("metadata") or {})
                metadata["success_rate"] = score
                await self.memory.replace_skill(
                    skill["skill_name"], candidate["code"], metadata
                )
                await self.log_evolution(skill, candidate, score)
            else:
                await self.log_regression_avoided(skill, candidate, score)
            results.append(
                {
                    "skill_name": skill["skill_name"],
                    "score": score,
                    "accepted": accepted,
                }
            )
        return {
            "status": "completed",
            "results": results,
            "candidate_count": len(results),
        }

    async def get_low_performing_skills(
        self, threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        return await self.memory.low_performing_skills(threshold)

    async def get_failure_episodes(self, skill_id: str) -> list[dict[str, Any]]:
        return await self.memory.recent_failure_episodes(task_type=skill_id, limit=8)

    async def get_success_episodes(self, skill_id: str) -> list[dict[str, Any]]:
        return await self.memory.recent_success_episodes(task_type=skill_id, limit=8)

    async def evolve_skill(
        self,
        *,
        skill: dict[str, Any],
        failure_episodes: list[dict[str, Any]],
        success_episodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = json.dumps(
            {
                "skill": skill,
                "failure_episodes": failure_episodes,
                "success_episodes": success_episodes,
                "instruction": "Return JSON with keys code, notes, test_command.",
            },
            ensure_ascii=False,
        )
        generation = await self.llm_client.generate_with_backend(
            (
                "freellm_external"
                if self.llm_client.external_backend_configured
                else "tinyllama"
            ),
            system_prompt="You improve autonomous agent skills. Return strict JSON.",
            user_prompt=prompt,
            max_tokens=1200,
            persona_weights={"evolution": 1.0},
        )
        parsed = self._parse_json(generation.text)
        if parsed is not None:
            return parsed
        return {
            "code": skill.get("content") or "",
            "notes": ["Fallback evolution candidate reused the current skill content."],
            "test_command": "python -m py_compile {path}",
        }

    async def test_skill_candidate(
        self, skill_name: str, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        code = str(candidate.get("code") or "")
        test_command = str(
            candidate.get("test_command") or "python -m py_compile {path}"
        )
        start = time.monotonic()
        add_result = await self.tool_executor.registry.invoke(
            name="add_skill",
            arguments={
                "name": skill_name,
                "code": code,
                "storage": "artifact",
                "metadata": {"skill_type": "evolved_candidate"},
            },
            constitution=self._constitution(),
            task_id=f"evolution-{skill_name}",
            workspace_root_override=self.tool_executor.sandbox.prepare(
                f"evolution-{skill_name}", "add_skill", {}
            ).workspace_root,
        )
        elapsed = time.monotonic() - start
        path = str(add_result.get("path") or "")
        compile_result = await self.tool_executor.registry.invoke(
            name="terminal_exec",
            arguments={"command": test_command.format(path=path), "timeout": 30},
            constitution=self._constitution(),
            task_id=f"evolution-{skill_name}",
            workspace_root_override=self.tool_executor.sandbox.prepare(
                f"evolution-{skill_name}", "terminal_exec", {}
            ).workspace_root,
        )
        return {
            "correctness": (
                1.0 if int(compile_result.get("returncode", 1)) == 0 else 0.0
            ),
            "speed": max(0.0, 1.0 - min(elapsed / 10.0, 1.0)),
            "token_efficiency": 0.8,
            "raw": compile_result,
        }

    def score_skill(self, test_result: dict) -> float:
        correctness = float(test_result.get("correctness", 0.0))
        speed = float(test_result.get("speed", 0.0))
        token_efficiency = float(test_result.get("token_efficiency", 0.0))
        return round((correctness * 0.5) + (speed * 0.2) + (token_efficiency * 0.3), 4)

    async def note_task_failure(self, task_type: str) -> bool:
        self._failure_counts[task_type] = self._failure_counts.get(task_type, 0) + 1
        if self._failure_counts[task_type] < 3:
            return False
        self._failure_counts[task_type] = 0
        await self.event_bus.publish(
            LisaEvent(
                type="evolution.emergency_triggered",
                payload={"task_type": task_type},
            )
        )
        return True

    async def log_evolution(
        self, skill: dict[str, Any], candidate: dict[str, Any], score: float
    ) -> None:
        await self.memory.record_audit(
            component="evolution",
            event_type="skill_replaced",
            payload={
                "skill_name": skill["skill_name"],
                "score": score,
                "notes": candidate.get("notes") or [],
            },
            task_id=f"evolution-{skill['skill_name']}",
        )

    async def log_regression_avoided(
        self, skill: dict[str, Any], candidate: dict[str, Any], score: float
    ) -> None:
        await self.memory.record_audit(
            component="evolution",
            event_type="regression_avoided",
            payload={
                "skill_name": skill["skill_name"],
                "score": score,
                "notes": candidate.get("notes") or [],
            },
            task_id=f"evolution-{skill['skill_name']}",
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        payload = text.strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            payload = payload[start : end + 1]
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    @staticmethod
    def _constitution():
        from lisa.constitutions import ConstitutionMode

        return ConstitutionMode.RESTRICTED
