from __future__ import annotations

import asyncio
from pathlib import Path

from lisa.embeddings import deterministic_embedding
from lisa.events import EventBus
from lisa.evolution_engine import EvolutionEngine
from lisa.memory_system import HybridMemoryCoordinator
from lisa.react_engine import ReActEngine
from lisa.router import LLMRouter
from lisa.sandbox import ToolSandbox
from lisa.schemas import EnrichedTask, InboundMessage, ToolCall, ToolResult


class StubLLMClient:
    external_backend_configured = False

    async def classify_with_local(self, *, labels, task, context=None):
        return None

    async def generate_with_backend(self, backend, **kwargs):
        raise RuntimeError("no backend configured")


class StubToolExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, tool_call: ToolCall, task_context):
        self.calls.append(tool_call.name)
        return ToolResult(
            tool=tool_call.name, success=True, output={"echo": tool_call.arguments}
        )


def test_hybrid_memory_stores_and_queries_episodes() -> None:
    async def run() -> None:
        memory = HybridMemoryCoordinator(agent_id="lisa", namespace="test-memory")
        await memory.start()
        try:
            await memory.store_episode(
                task_summary="build secure api endpoint",
                task_type="code_gen",
                tools_used=[{"tool": "file_write"}],
                success=True,
                failure_reason=None,
                skill_artifacts=[],
                metadata={"task_id": "t1"},
            )
            matches = await memory.similar_episodes("secure api", limit=3)
            assert matches
            assert "secure" in matches[0]["task_summary"]
        finally:
            await memory.close()

    asyncio.run(run())


def test_router_uses_expected_heuristics() -> None:
    router = LLMRouter(StubLLMClient())

    assert asyncio.run(router.classify_task("What is FastAPI?")) == "simple_qa"
    assert (
        asyncio.run(router.classify_task("Write a Python function that parses JSON"))
        == "code_gen"
    )
    assert (
        asyncio.run(router.classify_task("Use the browser tool to search docs"))
        == "tool_use"
    )
    assert (
        asyncio.run(router.classify_task("Reflect on why the retry logic failed"))
        == "reflection"
    )


def test_tool_sandbox_enforces_permission_layers(tmp_path: Path) -> None:
    sandbox = ToolSandbox(tmp_path)
    prepared = sandbox.prepare("task-1", "file_write", {"path": "a.txt"})

    assert prepared.workspace_root == (tmp_path / "workspace" / "task-1").resolve()
    assert sandbox.permission_for("file_read") == "L0"
    assert sandbox.permission_for("terminal_exec") == "L2"
    assert (
        sandbox.has_permission(
            ["L0", "L1"], "L2", explicit_user_grant=False, two_factor_confirmed=False
        )
        is False
    )
    assert (
        sandbox.has_permission(
            ["L0", "L1"], "L2", explicit_user_grant=True, two_factor_confirmed=False
        )
        is True
    )
    assert (
        sandbox.has_permission(
            ["L0", "L1"], "L3", explicit_user_grant=True, two_factor_confirmed=False
        )
        is False
    )
    assert (
        sandbox.has_permission(
            ["L0", "L1"], "L3", explicit_user_grant=False, two_factor_confirmed=True
        )
        is True
    )


def test_react_engine_falls_back_without_llm() -> None:
    async def run() -> None:
        memory = HybridMemoryCoordinator(agent_id="lisa", namespace="react-fallback")
        await memory.start()
        try:
            router = LLMRouter(StubLLMClient())
            engine = ReActEngine(
                llm_router=router,
                llm_client=StubLLMClient(),
                tool_executor=StubToolExecutor(),
                memory=memory,
                event_bus=EventBus(),
            )
            task = EnrichedTask(
                task_id="task-1",
                agent_id="lisa",
                inbound=InboundMessage(
                    source="direct", user_id="u1", channel="chat", text="hello lisa"
                ),
                description="hello lisa",
                max_tokens=256,
                constitution="restricted",
                persona_weights={"architect": 1.0},
                memory_context=[],
                skill_context=[],
                working_memory_key=memory.working_memory_key("task-1"),
            )
            result = await engine.run(task)
            assert result.success is True
            assert "hello lisa" in result.answer.lower()
        finally:
            await memory.close()

    asyncio.run(run())


def test_evolution_engine_scores_candidate_replacements() -> None:
    async def run() -> None:
        memory = HybridMemoryCoordinator(agent_id="lisa", namespace="evolution-score")
        await memory.start()
        try:
            await memory.upsert_skill_artifact(
                skill_name="retry_guard",
                content="def retry_guard(): pass",
                metadata={"success_rate": 0.2, "skill_type": "python"},
            )

            class StubExecutor:
                def __init__(self) -> None:
                    self.registry = self
                    self.sandbox = ToolSandbox(Path.cwd())

                async def invoke(self, name, arguments, constitution, **kwargs):
                    if name == "add_skill":
                        return {"path": str(Path.cwd() / "retry_guard.py")}
                    if name == "terminal_exec":
                        return {"returncode": 0, "stdout": "ok", "stderr": ""}
                    raise AssertionError(name)

            class StubLLM:
                external_backend_configured = True

                async def generate_with_backend(self, backend, **kwargs):
                    class Result:
                        text = '{"code":"def retry_guard(context):\\n    return True\\n","notes":["improved"],"test_command":"python -m py_compile {path}"}'

                    return Result()

            engine = EvolutionEngine(
                memory=memory,
                llm_client=StubLLM(),
                tool_executor=StubExecutor(),
                event_bus=EventBus(),
                agent_id="lisa",
            )
            result = await engine.run_nightly_cycle()
            assert result["status"] == "completed"
            assert result["results"][0]["accepted"] is True
        finally:
            await memory.close()

    asyncio.run(run())


def test_deterministic_embedding_has_expected_shape() -> None:
    vector = deterministic_embedding("hello world")
    assert len(vector) == 1536
