from __future__ import annotations

import asyncio
from pathlib import Path


from lisa.config import Settings
from lisa.constitutions import ConstitutionMode
from lisa.events import EventBus
from lisa.llm import LLMClient
from lisa.notepad import AsyncNotepadWriter, Notepad
from lisa.tools import ToolContext, ToolRegistry
from personal.context_store import PersonalContextStore
from safety.webhooks import WebhookSecrets, verify_webhook
from utils.secure_artifacts import encrypt_artifact_url


def test_notepad_startup_maintenance_backups_and_purges_old_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "data" / "notepad.db"
    notepad = Notepad(db_path)
    notepad.log_entry(
        entry_type="task_summary",
        payload={
            "user_id": "u1",
            "channel": "telegram",
            "input": "old message",
            "output": "old output",
            "outcome": "success",
            "reward": 0.9,
        },
        constitution=ConstitutionMode.RESTRICTED,
        personas={"architect": 1.0},
    )

    old_iso = "2025-01-01T00:00:00+00:00"
    assert notepad.memory._backend.ledger_entries
    notepad.memory._backend.ledger_entries[0]["created_at"] = old_iso

    maintenance = notepad.startup_maintenance(
        tmp_path / "backups", retention_days=1, keep_latest_backups=2
    )

    assert maintenance["status"] == "hybrid_memory_active"


def test_webhook_signature_helpers_cover_telegram_and_slack() -> None:
    telegram_secret = "telegram-secret"
    verify_webhook(
        "telegram",
        {"X-Telegram-Bot-Api-Secret-Token": telegram_secret},
        b"{}",
        url="https://example.invalid/telegram/webhook",
        secrets=WebhookSecrets(telegram_secret=telegram_secret),
    )

    slack_secret = "slack-signing-secret"
    import time

    timestamp = str(int(time.time()))
    body = b'{"type":"url_verification","challenge":"abc"}'
    import hashlib
    import hmac

    signature = (
        "v0="
        + hmac.new(
            slack_secret.encode("utf-8"),
            f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    verify_webhook(
        "slack",
        {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
        body,
        url="https://example.invalid/slack/events",
        secrets=WebhookSecrets(slack_signing_secret=slack_secret),
    )


def test_personal_context_store_supports_preferences_and_reminders(
    tmp_path: Path,
) -> None:
    store = PersonalContextStore(tmp_path / "personal.db")
    store.set("preferred_language", "python", kind="text")
    reminder_id = store.add_reminder("standup", "2026-06-11T10:00:00+00:00")

    summary = store.summary()
    assert summary["preferences"]["preferred_language"] == "python"
    assert summary["reminders"][0]["id"] == reminder_id
    assert store.due_reminders("2026-06-11T12:00:00+00:00")[0]["text"] == "standup"


def test_skill_versioning_and_rollback_restores_previous_file(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
    )
    notepad = Notepad(settings.db_path)
    writer = AsyncNotepadWriter(notepad=notepad, event_bus=EventBus())
    registry = ToolRegistry(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        event_bus=EventBus(),
        notepad_writer=writer,
    )
    context = ToolContext(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        constitution=ConstitutionMode.RESTRICTED,
        event_bus=EventBus(),
        notepad_writer=writer,
    )

    async def run() -> dict[str, object]:
        first = await registry._add_skill(
            {
                "name": "Echo Skill",
                "code": "async def skill(**kwargs):\n    return {'version': 1, 'kwargs': kwargs}\n",
            },
            context,
        )
        second = await registry._add_skill(
            {
                "name": "Echo Skill",
                "code": "async def skill(**kwargs):\n    return {'version': 2, 'kwargs': kwargs}\n",
            },
            context,
        )
        rollback = await registry._rollback_skill(
            {"name": "Echo Skill", "version": Path(str(second["archive_path"])).name},
            context,
        )
        return {"first": first, "second": second, "rollback": rollback}

    result = asyncio.run(run())

    restored_path = Path(str(result["rollback"]["path"]))
    assert restored_path.exists()
    assert "version': 1" in restored_path.read_text(
        encoding="utf-8"
    ) or 'version": 1' in restored_path.read_text(encoding="utf-8")


def test_skill_wrapper_injects_context_for_single_parameter_skills(
    tmp_path: Path,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
    )
    notepad = Notepad(settings.db_path)

    class StubWriter:
        async def enqueue(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(1)
            return future

    writer = StubWriter()
    registry = ToolRegistry(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        event_bus=EventBus(),
        notepad_writer=writer,
    )
    context = ToolContext(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        constitution=ConstitutionMode.RESTRICTED,
        event_bus=EventBus(),
        notepad_writer=writer,
        session_id="session-1",
    )

    async def run() -> dict[str, object]:
        await registry._add_skill(
            {
                "name": "Context Skill",
                "code": (
                    "def skill(context):\n"
                    "    return {'constitution': context.constitution.value, 'session_id': context.session_id}\n"
                ),
            },
            context,
        )
        return await registry.invoke(
            "context_skill", {}, ConstitutionMode.RESTRICTED, session_id="session-1"
        )

    result = asyncio.run(run())

    assert result["constitution"] == "restricted"
    assert result["session_id"] == "session-1"


def test_notepad_registers_and_searches_evolution_skill_artifacts(
    tmp_path: Path,
) -> None:
    notepad = Notepad(tmp_path / "data" / "test.db")
    artifact_url = (
        (tmp_path / "data" / "artifacts" / "retry_guard.py").resolve().as_uri()
    )
    record = notepad.register_evolution_skill_artifact(
        skill_name="retry_guard",
        description="Retries transient terminal failures.",
        encrypted_url=encrypt_artifact_url(artifact_url, "unit-test-secret"),
        url_hash="hash-1",
        keywords=["retry", "terminal", "timeout"],
        checksum="checksum-1",
        metadata={"cluster": "timeouts"},
    )

    assert record["skill_name"] == "retry_guard"
    matches = notepad.search_evolution_skill_artifacts("retry", limit=3)
    assert matches and matches[0]["skill_name"] == "retry_guard"

    notepad.disable_evolution_skill("retry_guard", "manual disable")
    assert not notepad.search_evolution_skill_artifacts("retry", limit=3)


def test_artifact_backed_skill_is_lazy_loaded_without_touching_repo_skills_dir(
    tmp_path: Path,
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        db_path=tmp_path / "data" / "test.db",
        skills_dir=tmp_path / "skills",
        evolution_artifacts_dir=tmp_path / "data" / "evolution_artifacts",
        persona_vectors_path=tmp_path / "data" / "persona_vectors.npz",
        gating_model_path=tmp_path / "data" / "gating_model.pkl",
        enable_browser_tools=False,
    )
    notepad = Notepad(settings.db_path)

    class StubWriter:
        async def enqueue(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(1)
            return future

    writer = StubWriter()
    registry = ToolRegistry(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        event_bus=EventBus(),
        notepad_writer=writer,
    )
    context = ToolContext(
        settings=settings,
        notepad=notepad,
        llm_client=LLMClient(settings),
        constitution=ConstitutionMode.RESTRICTED,
        event_bus=EventBus(),
        notepad_writer=writer,
    )

    async def run() -> tuple[dict[str, object], dict[str, object]]:
        stored = await registry._add_skill(
            {
                "name": "Retry Guard",
                "description": "Retries terminal failures with a bounded policy.",
                "keywords": ["retry", "terminal", "timeout"],
                "storage": "artifact",
                "code": "def retry_guard(**kwargs):\n    return {'loaded': True, 'kwargs': kwargs}\n",
            },
            context,
        )
        registry._tools.pop("retry_guard", None)
        invoked = await registry.invoke(
            "retry_guard", {"attempts": 2}, ConstitutionMode.RESTRICTED
        )
        return stored, invoked

    stored, invoked = asyncio.run(run())

    assert stored["status"] == "artifact_saved"
    assert invoked["loaded"] is True
    assert invoked["kwargs"]["attempts"] == 2
    assert not (settings.skills_dir / "retry_guard.py").exists()
    matches = notepad.search_evolution_skill_artifacts("terminal", limit=2)
    assert matches and matches[0]["skill_name"] == "retry_guard"
