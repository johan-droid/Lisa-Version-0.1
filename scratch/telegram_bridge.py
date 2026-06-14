import os
import sys
import time
import subprocess
import httpx
import psutil
from pathlib import Path
from dotenv import load_dotenv

from utils.process_lock import ProcessLock, ProcessLockHeldError

load_dotenv(".env.local")
BOT_TOKEN = os.environ.get("LISA_TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    print("Error: LISA_TELEGRAM_BOT_TOKEN not set in environment or .env.local")
    sys.exit(1)

WEBHOOK_SECRET = os.environ.get("LISA_TELEGRAM_WEBHOOK_SECRET", BOT_TOKEN)

WEBHOOK_URL = "http://127.0.0.1:8800/telegram/webhook"
LOG_FILE = "logs/telegram_poll.log"
LOCK_PATH = Path("data") / "locks" / "telegram_bridge.lock"
MAX_WEBHOOK_DELETE_FAILURES = 5
MAX_BACKOFF_SECONDS = 60


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line, end="")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def kill_existing_servers():
    log("Stopping existing uvicorn and supervisor processes...")
    current_pid = os.getpid()
    parent_pid = os.getppid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == current_pid or pid == parent_pid:
                continue
            name = (proc.info["name"] or "").lower()
            if "python" not in name:
                continue
            cmdline = proc.info["cmdline"]
            if cmdline:
                cmd_str = " ".join(cmdline).lower()
                if (
                    "supervisor.py" in cmd_str
                    or "main.py" in cmd_str
                    or "telegram_bridge.py" in cmd_str
                    or "uvicorn" in cmd_str
                ):
                    log(f"Killing process {pid}: {cmdline}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(2)


def start_supervisor():
    cmd = [sys.executable, "supervisor.py"]
    log(f"Spawning supervisor: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    os.makedirs("logs", exist_ok=True)
    lock = ProcessLock(LOCK_PATH, role="telegram-bridge")
    try:
        lock.acquire()
    except ProcessLockHeldError as exc:
        holder = exc.holder
        if holder is not None and holder.pid is not None:
            log(
                f"Telegram bridge lock is already held by PID {holder.pid}. Refusing to start a second bridge."
            )
        else:
            log("Telegram bridge lock is already held. Refusing to start.")
        sys.exit(1)
    kill_existing_servers()
    try:
        proc = start_supervisor()
        log("Waiting for LISA to initialize...")
        time.sleep(8)

        log("Starting Telegram Polling & Webhook Forwarding Bridge...")

        offset = 0
        client = httpx.Client(timeout=10.0)
        conflict_failures = 0
        backoff_seconds = 1

        while True:
            try:
                if proc.poll() is not None:
                    log("Supervisor process exited. Restarting supervisor...")
                    proc = start_supervisor()
                    time.sleep(8)

                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={offset}&timeout=5"
                resp = client.get(url)
                if resp.status_code == 200:
                    conflict_failures = 0
                    backoff_seconds = 1
                    data = resp.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            update_id = update["update_id"]
                            offset = max(offset, update_id + 1)

                            log(
                                f"Received update {update_id} from Telegram. Forwarding to local MessageHub..."
                            )
                            headers = {
                                "X-Telegram-Bot-Api-Secret-Token": WEBHOOK_SECRET,
                                "Content-Type": "application/json",
                            }
                            try:
                                fwd_resp = client.post(
                                    WEBHOOK_URL,
                                    json=update,
                                    headers=headers,
                                    timeout=15.0,
                                )
                                log(
                                    f"Forward response status: {fwd_resp.status_code}, body: {fwd_resp.text}"
                                )
                            except Exception as fwd_err:
                                log(
                                    f"Failed to forward update to local MessageHub: {fwd_err}"
                                )
                elif resp.status_code == 409:
                    conflict_failures += 1
                    if conflict_failures > MAX_WEBHOOK_DELETE_FAILURES:
                        log(
                            "Circuit breaker opened after repeated Telegram webhook conflicts. Stopping bridge instead of retry storming."
                        )
                        sys.exit(1)
                    log(
                        f"Conflict: Telegram Webhook might still be active. Deleting webhook with backoff {backoff_seconds}s (attempt {conflict_failures})."
                    )
                    try:
                        delete_resp = client.post(
                            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook",
                            json={"drop_pending_updates": True},
                        )
                        log(f"Delete Telegram Webhook response: {delete_resp.text}")
                    except Exception as delete_err:
                        log(f"Error deleting Telegram webhook: {delete_err}")
                    time.sleep(backoff_seconds)
                    backoff_seconds = min(MAX_BACKOFF_SECONDS, backoff_seconds * 2)
                    continue
                else:
                    log(
                        f"Telegram API getUpdates returned status {resp.status_code}: {resp.text}"
                    )

            except Exception as e:
                log(f"Error in main polling loop: {e}")

            time.sleep(1)
    finally:
        lock.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Shutting down bridge...")
