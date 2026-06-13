import os
import sys
import time
import subprocess
import httpx
import psutil
from dotenv import load_dotenv

load_dotenv(".env.local")
BOT_TOKEN = os.environ.get("LISA_TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    print("Error: LISA_TELEGRAM_BOT_TOKEN not set in environment or .env.local")
    sys.exit(1)

WEBHOOK_SECRET = os.environ.get("LISA_TELEGRAM_WEBHOOK_SECRET", BOT_TOKEN)

WEBHOOK_URL = "http://127.0.0.1:8800/telegram/webhook"
LOG_FILE = "logs/telegram_poll.log"


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
    os.makedirs("data", exist_ok=True)
    pid_file = "data/telegram_bridge.pid"

    # Try to write PID atomically
    import fcntl
    try:
        pid_fd = open(pid_file, "w")
        fcntl.flock(pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid_fd.write(str(os.getpid()))
        pid_fd.flush()
    except (IOError, BlockingIOError):
        try:
            with open(pid_file, "r") as f:
                holding_pid = f.read().strip()
        except IOError:
            holding_pid = "unknown"
        print(f"Cannot start Telegram bridge: already running (PID: {holding_pid}). Exiting.")
        sys.exit(1)

    os.makedirs("logs", exist_ok=True)

    kill_existing_servers()

    # Start supervisor
    proc = start_supervisor()

    # Wait for LISA to initialize
    log("Waiting for LISA to initialize...")
    time.sleep(8)

    log("Starting Telegram Polling & Webhook Forwarding Bridge...")

    # Delete webhook on telegram to enable getUpdates
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook",
            json={"drop_pending_updates": True},
        )
        log(f"Delete Telegram Webhook response: {r.json()}")
    except Exception as e:
        log(f"Error deleting Telegram webhook: {e}")

    offset = 0
    client = httpx.Client(timeout=10.0)

    while True:
        try:
            # Check if supervisor process is still alive
            if proc.poll() is not None:
                log("Supervisor process exited. Restarting supervisor...")
                proc = start_supervisor()
                time.sleep(8)

            # Poll Telegram Updates
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={offset}&timeout=5"
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        update_id = update["update_id"]
                        offset = max(offset, update_id + 1)

                        log(
                            f"Received update {update_id} from Telegram. Forwarding to local MessageHub..."
                        )

                        # Forward to local webhook
                        headers = {
                            "X-Telegram-Bot-Api-Secret-Token": WEBHOOK_SECRET,
                            "Content-Type": "application/json",
                        }

                        try:
                            fwd_resp = client.post(
                                WEBHOOK_URL, json=update, headers=headers, timeout=15.0
                            )
                            log(
                                f"Forward response status: {fwd_resp.status_code}, body: {fwd_resp.text}"
                            )
                        except Exception as fwd_err:
                            log(
                                f"Failed to forward update to local MessageHub: {fwd_err}"
                            )
            elif resp.status_code == 409:
                log(
                    "Conflict: Telegram Webhook might still be active. Attempting to delete webhook again..."
                )
                client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook",
                    json={"drop_pending_updates": True},
                )

                # Exponential backoff for conflicts
                conflict_retries = getattr(client, "_conflict_retries", 0) + 1
                client._conflict_retries = conflict_retries
                backoff = min(60, 2 ** conflict_retries)
                log(f"Backing off for {backoff} seconds after conflict...")
                time.sleep(backoff)
            else:
                if hasattr(client, "_conflict_retries"):
                    client._conflict_retries = 0
                log(

                    f"Telegram API getUpdates returned status {resp.status_code}: {resp.text}"
                )

        except Exception as e:
            log(f"Error in main polling loop: {e}")

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Shutting down bridge...")
