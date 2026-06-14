from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class LockHolder:
    pid: int | None
    role: str
    acquired_at: str | None
    command: str | None


class ProcessLockHeldError(RuntimeError):
    def __init__(self, path: Path, holder: LockHolder | None) -> None:
        self.path = path
        self.holder = holder
        holder_text = "another instance"
        if holder is not None and holder.pid is not None:
            holder_text = f"PID {holder.pid} ({holder.role})"
        super().__init__(f"Lock {path} is already held by {holder_text}.")


class ProcessLock:
    def __init__(self, path: Path, *, role: str) -> None:
        self.path = Path(path)
        self.role = role
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                holder = self._read_holder()
                if holder is not None and holder.pid and psutil.pid_exists(holder.pid):
                    raise ProcessLockHeldError(self.path, holder)
                self._remove_stale_lock()
                continue

            payload = {
                "pid": os.getpid(),
                "role": self.role,
                "acquired_at": _utcnow(),
                "command": " ".join(sys.argv),
            }
            os.write(fd, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
            os.close(fd)
            self._fd = 1
            return

    def release(self) -> None:
        if self._fd is None:
            return
        self._fd = None
        try:
            payload = self._read_holder()
            if payload is None or payload.pid == os.getpid():
                self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _read_holder(self) -> LockHolder | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return LockHolder(
            pid=int(raw["pid"]) if str(raw.get("pid") or "").isdigit() else None,
            role=str(raw.get("role") or "unknown"),
            acquired_at=str(raw.get("acquired_at") or "") or None,
            command=str(raw.get("command") or "") or None,
        )

    def _remove_stale_lock(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            holder = self._read_holder()
            raise ProcessLockHeldError(self.path, holder) from exc
