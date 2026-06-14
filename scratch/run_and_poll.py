import os
import sys
import time
import subprocess
import httpx
import psutil
import asyncio

BOT_TOKEN = os.environ.get("LISA_TELEGRAM_BOT_TOKEN", "").strip()


def kill_existing_servers():
    print("Stopping existing uvicorn and supervisor processes...")
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == current_pid:
                continue
            cmdline = proc.info["cmdline"]
            if cmdline:
                cmd_str = " ".join(cmdline).lower()
                if "supervisor.py" in cmd_str or "main.py" in cmd_str:
                    print(f"Killing process {pid}: {cmdline}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Give it a moment to release ports
    time.sleep(2)


async def poll_telegram():
    if not BOT_TOKEN:
        raise RuntimeError("LISA_TELEGRAM_BOT_TOKEN must be set before running this helper.")
    print("Starting Telegram polling loop...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        # First, let's delete any webhook just in case
        try:
            r = await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
            )
            print("Delete webhook response:", r.json())
        except Exception as e:
            print("Error deleting webhook:", e)

        offset = 0
        pinged_chats = set()

        # Let's run for a while to wait for user interactions
        for _ in range(60):
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={offset}&timeout=5"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            update_id = update["update_id"]
                            offset = max(offset, update_id + 1)

                            message = update.get("message")
                            if message:
                                chat = message.get("chat", {})
                                chat_id = chat.get("id")
                                username = chat.get("username", "unknown")
                                text = message.get("text", "")

                                print(
                                    f"Received message from @{username} (chat_id: {chat_id}): '{text}'"
                                )

                                if chat_id and chat_id not in pinged_chats:
                                    print(f"Pinging user on chat_id {chat_id}...")
                                    send_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                                    payload = {
                                        "chat_id": chat_id,
                                        "text": "Hello! LISA is now running in the terminal and has successfully pinged you. How can I help you today?",
                                    }
                                    send_resp = await client.post(
                                        send_url, json=payload
                                    )
                                    print("Send message response:", send_resp.json())
                                    pinged_chats.add(chat_id)
            except Exception as e:
                print("Error in polling:", e)
            await asyncio.sleep(2)


def start_supervisor():
    # Start supervisor.py
    cmd = [sys.executable, "supervisor.py"]
    print(f"Spawning supervisor: {' '.join(cmd)}")
    # We run it and let its stdout/stderr flow
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


async def main():
    kill_existing_servers()
    proc = start_supervisor()

    # Wait for startup
    await asyncio.sleep(5)

    try:
        await poll_telegram()
    finally:
        print("Terminating supervisor...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
