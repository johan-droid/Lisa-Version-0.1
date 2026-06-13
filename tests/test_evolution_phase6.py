from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from lisa.config import Settings
from lisa.events import EventBus
from lisa.evolution import NightlyEvolutionScheduler


def test_evolution_scheduler_skips_when_recent_user_activity(tmp_path: Path) -> None:
    now = datetime(2026, 6, 11, 10, 0, tzinfo=timezone.utc)

    class StubNotepad:
        def latest_interaction_timestamp(self) -> datetime:
            return now - timedelta(minutes=10)

        def recent_failure_interactions(
            self, limit: int = 24, min_reward: float = 0.35
        ):
            return [
                {
                    "id": 1,
                    "payload": {
                        "error": "timeout",
                        "self_critique": "retry guard failed",
                        "user_input": "Build a retry guard",
                        "response": "It timed out.",
                    },
                    "score": 3.0,
                }
            ]

        def latest_entries(self, limit: int = 1):
            return []

        def evolution_candidates(self, limit: int, min_reward: float):
            return []

    class StubConductor:
        def is_idle(self) -> bool:
            return True

    class StubWriter:
        async def enqueue(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(1)
            return fut

        async def flush_pending(self) -> None:
            return None

    class StubTools:
        async def invoke(self, *args, **kwargs):
            return {
                "content": json.dumps(
                    {
                        "name": "retry_guard",
                        "code": "def retry_guard(context):\n    return True",
                        "test_command": "python -m py_compile {path}",
                    }
                )
            }

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        evolution_enabled=True,
        evolution_check_interval_seconds=1,
        evolution_window_start_hour=2,
        evolution_window_duration_hours=3,
        evolution_idle_after_seconds=1800,
        enable_browser_tools=False,
        message_hub_enabled=False,
    )
    runtime = SimpleNamespace(
        settings=settings,
        notepad=StubNotepad(),
        notepad_writer=StubWriter(),
        tools=StubTools(),
    )
    scheduler = NightlyEvolutionScheduler(
        runtime=runtime,
        conductor=StubConductor(),
        event_bus=EventBus(),
        clock=lambda: now,
    )

    result = asyncio.run(scheduler.run_once())

    assert result.status == "skipped"
    assert result.skipped_reason and result.skipped_reason.startswith("recent_activity")


def test_evolution_scheduler_runs_more_aggressively_at_night(tmp_path: Path) -> None:
    now = datetime(2026, 6, 11, 3, 0, tzinfo=timezone.utc)
    skill_path = tmp_path / "skills" / "retry_guard.py"

    class StubNotepad:
        def latest_interaction_timestamp(self) -> datetime:
            return now - timedelta(minutes=10)

        def recent_failure_interactions(
            self, limit: int = 24, min_reward: float = 0.35
        ):
            return [
                {
                    "id": 1,
                    "payload": {
                        "error": "timeout",
                        "self_critique": "retry guard failed",
                        "user_input": "Build a retry guard",
                        "response": "It timed out.",
                    },
                    "score": 3.0,
                }
            ]

        def latest_entries(self, limit: int = 1):
            return []

        def evolution_candidates(self, limit: int, min_reward: float):
            return []

    class StubConductor:
        def is_idle(self) -> bool:
            return True

    class StubWriter:
        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []

        async def enqueue(
            self,
            entry_type: str,
            payload: dict[str, object],
            constitution: object,
            personas=None,
            critique=None,
            reward=None,
        ):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self.records.append({"entry_type": entry_type, "payload": payload})
            future.set_result(len(self.records))
            return future

        async def flush_pending(self) -> None:
            return None

    class StubTools:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self._skills: dict[str, str] = {}

        async def invoke(
            self, name: str, arguments: dict[str, object], constitution: object
        ):
            self.calls.append((name, arguments))
            if name == "call_external_llm":
                prompt = str(arguments.get("prompt", ""))
                if "Create one new Python skill" in prompt:
                    return {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "content": json.dumps(
                            {
                                "name": "retry_guard",
                                "description": "Retries transient failures.",
                                "code": (
                                    "def retry_guard(context):\n"
                                    "    return {'status': 'ok'}\n"
                                ),
                                "test_command": "python -m py_compile {path}",
                                "notes": ["Nightly synthesis."],
                                "smoke_test_arguments": {},
                            }
                        ),
                        "usage": {"total_tokens": 15},
                    }
                return {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "content": json.dumps(
                        {"insight": "retry", "pattern": "bounded retry"}
                    ),
                    "usage": {"total_tokens": 5},
                }
            if name == "file_write":
                path = Path(str(arguments["path"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(arguments["content"]), encoding="utf-8")
                return {
                    "path": str(path),
                    "bytes_written": len(str(arguments["content"]).encode("utf-8")),
                }
            if name == "terminal_exec":
                return {"returncode": 0, "stdout": "compiled", "stderr": ""}
            if name == "add_skill":
                path = skill_path
                path.parent.mkdir(parents=True, exist_ok=True)
                code = str(arguments["code"])
                path.write_text(code, encoding="utf-8")
                self._skills[str(arguments["name"])] = code
                return {"path": str(path), "status": "saved"}
            if name == "dashboard_update":
                return {"status": "ok"}
            if name in self._skills:
                return {"status": "ok", "skill": name}
            raise AssertionError(name)

    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        evolution_enabled=True,
        evolution_check_interval_seconds=1,
        evolution_window_start_hour=2,
        evolution_window_duration_hours=3,
        evolution_idle_after_seconds=1800,
        enable_browser_tools=False,
        message_hub_enabled=False,
        evolution_staging_dir=tmp_path / "staging",
    )
    runtime = SimpleNamespace(
        settings=settings,
        notepad=StubNotepad(),
        notepad_writer=StubWriter(),
        tools=StubTools(),
    )
    scheduler = NightlyEvolutionScheduler(
        runtime=runtime,
        conductor=StubConductor(),
        event_bus=EventBus(),
        clock=lambda: now,
    )

    result = asyncio.run(scheduler.run_once())

    assert result.status == "registered"
    assert result.registered is True
    assert any(name == "call_external_llm" for name, _ in runtime.tools.calls)
    assert skill_path.exists()
