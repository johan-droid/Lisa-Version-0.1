from __future__ import annotations

import asyncio
from pathlib import Path

from lisa.constitutions import ConstitutionMode
from lisa.notepad import AsyncNotepadWriter, Notepad, search_notepad


def test_notepad_interactions_are_searchable(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "lisa_notepad.db"
    notepad = Notepad(db_path)

    notepad.log_entry(
        entry_type="task_summary",
        payload={
            "user_id": "u-1",
            "channel": "telegram",
            "input": "Write a secure API endpoint",
            "output": "Implemented an authenticated endpoint.",
            "tool_calls": [
                {"name": "search_notepad", "arguments": {"query": "secure"}}
            ],
            "persona_blend": {"architect": 0.7, "oracle": 0.3},
            "outcome": "success",
            "reward": 0.9,
            "self_critique": "Good security coverage.",
        },
        constitution=ConstitutionMode.RESTRICTED,
        personas={"architect": 0.7, "oracle": 0.3},
        reward=0.9,
    )

    rows = search_notepad("secure", limit=5, db_path=db_path)

    assert rows
    row = rows[0]
    assert row["input"] == "Write a secure API endpoint"
    assert row["output"] == "Implemented an authenticated endpoint."
    assert row["tool_calls"][0]["name"] == "search_notepad"
    assert row["persona_blend"]["architect"] == 0.7


def test_async_notepad_writer_batches_ten_items(tmp_path: Path) -> None:
    class StubNotepad:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def log_entries_batch(self, entries):
            self.batch_sizes.append(len(entries))
            return list(range(1, len(entries) + 1))

    async def run() -> list[int]:
        notepad = StubNotepad()
        writer = AsyncNotepadWriter(notepad=notepad, batch_size=10, flush_interval=0.1)
        await writer.start()
        futures = []
        try:
            for index in range(11):
                future = await writer.enqueue(
                    entry_type="task_summary",
                    payload={"input": f"message {index}", "output": "ok"},
                    constitution="restricted",
                    personas={"architect": 1.0},
                )
                futures.append(future)

            await asyncio.wait_for(asyncio.gather(*futures), timeout=2)
        finally:
            await writer.close()

        return notepad.batch_sizes

    batch_sizes = asyncio.run(run())

    assert batch_sizes[:2] == [10, 1]
