from __future__ import annotations

import json
from typing import Any

from lisa.events import EventBus, LisaEvent
from lisa.router import LLMRouter
from lisa.schemas import (
    EnrichedTask,
    ReActReflection,
    ReActThought,
    TaskContext,
    TaskResult,
    ToolCall,
    ToolResult,
)


class ReActEngine:
    MAX_ITERATIONS = 10
    MAX_JSON_PARSE_CHARS = 16_000

    def __init__(
        self,
        *,
        llm_router: LLMRouter,
        llm_client: Any,
        tool_executor: Any,
        memory: Any,
        event_bus: EventBus,
    ) -> None:
        self.llm_router = llm_router
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory = memory
        self.event_bus = event_bus

    async def run(self, enriched_task: EnrichedTask) -> TaskResult:
        scratchpad: list[dict[str, Any]] = []
        tool_results: list[ToolResult] = []
        task_context = TaskContext(
            task_id=enriched_task.task_id,
            agent_id=enriched_task.agent_id,
            session_id=enriched_task.inbound.session_id,
            constitution=enriched_task.constitution,
            approved_levels=list(
                enriched_task.metadata.get("approved_levels") or ["L0", "L1"]
            ),
            explicit_user_grant=bool(
                enriched_task.metadata.get("explicit_user_grant", False)
            ),
            two_factor_confirmed=bool(
                enriched_task.metadata.get("two_factor_confirmed", False)
            ),
            metadata=dict(enriched_task.metadata),
        )

        await self.memory.append_message(
            enriched_task.task_id, "user", enriched_task.description
        )

        for iteration in range(self.MAX_ITERATIONS):
            await self.memory.bump_iteration(enriched_task.task_id)
            thought = await self._think(enriched_task, scratchpad)
            thought_item = {
                "type": "thought",
                "iteration": iteration,
                "content": thought.model_dump(),
            }
            scratchpad.append(thought_item)
            await self.memory.write_scratchpad(enriched_task.task_id, thought_item)
            await self.memory.set_current_plan(enriched_task.task_id, thought.plan)

            if thought.action_type == "ANSWER":
                answer = (
                    thought.answer or thought.reasoning or "No answer was produced."
                )
                await self.memory.append_message(
                    enriched_task.task_id, "assistant", answer
                )
                result = TaskResult(
                    answer=answer,
                    scratchpad=scratchpad,
                    iterations=iteration + 1,
                    success=True,
                    tool_results=tool_results,
                )
                await self._store_episode(enriched_task, result)
                return result

            if thought.tool_call is None:
                fallback = TaskResult(
                    answer="Reasoning loop stopped because no valid tool call or answer was produced.",
                    scratchpad=scratchpad,
                    iterations=iteration + 1,
                    success=False,
                    failure_reason="missing_action",
                    tool_results=tool_results,
                )
                await self._store_episode(enriched_task, fallback)
                return fallback

            result = await self.tool_executor.execute(thought.tool_call, task_context)
            tool_results.append(result)
            observation_item = {
                "type": "observation",
                "iteration": iteration,
                "content": result.model_dump(),
            }
            scratchpad.append(observation_item)
            await self.memory.append_tool_call(
                enriched_task.task_id, observation_item["content"]
            )
            await self.memory.write_scratchpad(enriched_task.task_id, observation_item)

            reflection = await self._reflect(enriched_task, scratchpad)
            reflection_item = {
                "type": "reflection",
                "iteration": iteration,
                "content": reflection.model_dump(),
            }
            scratchpad.append(reflection_item)
            await self.memory.append_reflection(
                enriched_task.task_id, "; ".join(reflection.notes)
            )
            await self.memory.write_scratchpad(enriched_task.task_id, reflection_item)

            if reflection.updated_plan:
                await self.memory.set_current_plan(
                    enriched_task.task_id, reflection.updated_plan
                )

            if reflection.goal_achieved:
                answer = await self._final_answer(enriched_task, scratchpad)
                await self.memory.append_message(
                    enriched_task.task_id, "assistant", answer
                )
                result = TaskResult(
                    answer=answer,
                    scratchpad=scratchpad,
                    iterations=iteration + 1,
                    success=True,
                    tool_results=tool_results,
                )
                await self._store_episode(enriched_task, result)
                return result

        result = TaskResult(
            answer="The agent reached the maximum ReAct iterations without closing the task.",
            scratchpad=scratchpad,
            iterations=self.MAX_ITERATIONS,
            success=False,
            failure_reason="max_iterations",
            tool_results=tool_results,
        )
        await self._store_episode(enriched_task, result)
        return result

    async def _think(
        self, enriched_task: EnrichedTask, scratchpad: list[dict[str, Any]]
    ) -> ReActThought:
        route = await self.llm_router.route(
            enriched_task.description,
            {
                "memory_context": enriched_task.memory_context,
                "skill_context": enriched_task.skill_context,
                "stress_level": enriched_task.stress_level,
            },
        )
        system_prompt = (
            "You are LISA running a ReAct loop. "
            "Return JSON with keys action_type, reasoning, plan, tool_call, answer. "
            "action_type must be TOOL or ANSWER. "
            "When using a tool, tool_call must contain {name, arguments}."
        )
        user_prompt = json.dumps(
            {
                "task": enriched_task.description,
                "memory_context": enriched_task.memory_context,
                "skill_context": enriched_task.skill_context,
                "scratchpad": scratchpad[-8:],
            },
            ensure_ascii=False,
        )
        try:
            generation = await self.llm_client.generate_with_backend(
                route.brain,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=min(900, enriched_task.max_tokens),
                persona_weights=enriched_task.persona_weights,
                stress_level=enriched_task.stress_level,
            )
        except Exception:
            generation = None
        if generation is None:
            lowered = enriched_task.description.lower()
            if any(
                token in lowered for token in ("search", "find", "history", "memory")
            ):
                return ReActThought(
                    action_type="TOOL",
                    reasoning="Fallback heuristic selected notepad search.",
                    plan=["Search recent memory for relevant history."],
                    tool_call=ToolCall(
                        name="search_notepad",
                        arguments={"query": enriched_task.description, "limit": 5},
                    ),
                )
            return ReActThought(
                action_type="ANSWER",
                reasoning="Fallback deterministic response because no LLM backend is available.",
                answer=f"LISA received the task: {enriched_task.description}",
                plan=[],
            )
        parsed = self._parse_json(generation.text)
        if isinstance(parsed, dict):
            tool_payload = parsed.get("tool_call")
            if (
                tool_payload is None
                and isinstance(parsed.get("tool_calls"), list)
                and parsed["tool_calls"]
            ):
                first_tool = parsed["tool_calls"][0]
                if isinstance(first_tool, dict):
                    tool_payload = first_tool
            tool_call = None
            if isinstance(tool_payload, dict) and tool_payload.get("name"):
                tool_call = ToolCall(
                    name=str(tool_payload["name"]),
                    arguments=dict(tool_payload.get("arguments") or {}),
                )
            return ReActThought(
                action_type="TOOL" if tool_call is not None and str(parsed.get("action_type", "")).upper() != "ANSWER" else str(parsed.get("action_type") or "ANSWER").upper(),  # type: ignore[arg-type]
                reasoning=str(parsed.get("reasoning") or generation.text),
                plan=(
                    [str(item) for item in parsed.get("plan", [])]
                    if isinstance(parsed.get("plan"), list)
                    else []
                ),
                tool_call=tool_call,
                answer=str(parsed.get("answer") or "") or None,
            )
        if generation.tool_calls:
            first = generation.tool_calls[0]
            return ReActThought(
                action_type="TOOL",
                reasoning=generation.text or "Tool use selected.",
                tool_call=first,
                plan=[],
            )
        return ReActThought(
            action_type="ANSWER",
            reasoning=generation.text,
            answer=generation.text,
            plan=[],
        )

    async def _reflect(
        self, enriched_task: EnrichedTask, scratchpad: list[dict[str, Any]]
    ) -> ReActReflection:
        system_prompt = (
            "You are LISA reflecting on progress. "
            "Return JSON with goal_achieved, stuck, notes, updated_plan."
        )
        user_prompt = json.dumps(
            {
                "goal": enriched_task.description,
                "scratchpad": scratchpad[-10:],
            },
            ensure_ascii=False,
        )
        try:
            generation = await self.llm_client.generate_with_backend(
                (
                    "freellm_external"
                    if self.llm_client.external_backend_configured
                    else "tinyllama"
                ),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=300,
                persona_weights=enriched_task.persona_weights,
            )
        except Exception:
            generation = None
        if generation is None:
            last_observation = next(
                (
                    item
                    for item in reversed(scratchpad)
                    if item.get("type") == "observation"
                ),
                None,
            )
            observation_payload = (
                last_observation.get("content", {})
                if isinstance(last_observation, dict)
                else {}
            )
            success = (
                bool(observation_payload.get("success", False))
                if isinstance(observation_payload, dict)
                else False
            )
            return ReActReflection(
                goal_achieved=success,
                stuck=not success,
                notes=["Fallback reflection used because no LLM backend is available."],
                updated_plan=[],
            )
        parsed = self._parse_json(generation.text)
        if isinstance(parsed, dict):
            return ReActReflection(
                goal_achieved=bool(parsed.get("goal_achieved", False)),
                stuck=bool(parsed.get("stuck", False)),
                notes=(
                    [str(item) for item in parsed.get("notes", [])]
                    if isinstance(parsed.get("notes"), list)
                    else [generation.text]
                ),
                updated_plan=(
                    [str(item) for item in parsed.get("updated_plan", [])]
                    if isinstance(parsed.get("updated_plan"), list)
                    else []
                ),
            )
        last_observation = next(
            (
                item
                for item in reversed(scratchpad)
                if item.get("type") == "observation"
            ),
            None,
        )
        observation_payload = (
            last_observation.get("content", {})
            if isinstance(last_observation, dict)
            else {}
        )
        success = (
            bool(observation_payload.get("success", False))
            if isinstance(observation_payload, dict)
            else False
        )
        return ReActReflection(
            goal_achieved=success and len(scratchpad) >= 4,
            stuck=not success,
            notes=[generation.text],
            updated_plan=[],
        )

    async def _final_answer(
        self, enriched_task: EnrichedTask, scratchpad: list[dict[str, Any]]
    ) -> str:
        system_prompt = "You are LISA. Produce the final answer for the user based on the ReAct scratchpad."
        user_prompt = json.dumps(
            {"task": enriched_task.description, "scratchpad": scratchpad[-12:]},
            ensure_ascii=False,
        )
        try:
            generation = await self.llm_client.generate_with_backend(
                (
                    "freellm_external"
                    if self.llm_client.external_backend_configured
                    else "tinyllama"
                ),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=min(900, enriched_task.max_tokens),
                persona_weights=enriched_task.persona_weights,
            )
            return generation.text
        except Exception:
            last_observation = next(
                (
                    item
                    for item in reversed(scratchpad)
                    if item.get("type") == "observation"
                ),
                None,
            )
            if isinstance(last_observation, dict):
                content = last_observation.get("content", {})
                if isinstance(content, dict):
                    if content.get("success"):
                        return (
                            json.dumps(content.get("output"), ensure_ascii=False)
                            if not isinstance(content.get("output"), str)
                            else str(content.get("output"))
                        )
                    if content.get("error"):
                        return f"Task failed: {content['error']}"
            return f"LISA completed the task context for: {enriched_task.description}"

    async def _store_episode(
        self, enriched_task: EnrichedTask, result: TaskResult
    ) -> None:
        tools_used = [tool.model_dump() for tool in result.tool_results]
        await self.memory.store_episode(
            task_summary=f"{enriched_task.description}\n{result.answer}".strip(),
            task_type=str(enriched_task.metadata.get("task_type") or "react_task"),
            tools_used=tools_used,
            success=result.success,
            failure_reason=result.failure_reason,
            skill_artifacts=[
                {
                    "skill_name": str(item.get("skill_name") or ""),
                    "metadata": item.get("metadata") or {},
                }
                for item in enriched_task.skill_context
            ],
            metadata={
                "task_id": enriched_task.task_id,
                "session_id": enriched_task.inbound.session_id,
                "constitution": enriched_task.constitution,
                "persona_blend": enriched_task.persona_weights,
                "iterations": result.iterations,
            },
        )
        await self.memory.clear_task(enriched_task.task_id)
        await self.event_bus.publish(
            LisaEvent(
                type="react.completed",
                payload={
                    "task_id": enriched_task.task_id,
                    "success": result.success,
                    "iterations": result.iterations,
                },
                session_id=enriched_task.inbound.session_id,
                trace_id=enriched_task.task_id,
            )
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        payload = text[: ReActEngine.MAX_JSON_PARSE_CHARS].strip()
        if not payload:
            return None
        if payload.startswith("```"):
            payload = payload.strip("`")
            if payload.lower().startswith("json"):
                payload = payload[4:].strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            payload = payload[start : end + 1]
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None
