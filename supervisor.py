import subprocess
import time
import sys
import os
import signal
import psutil
import logging
import httpx

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [supervisor] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/supervisor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("lisa.supervisor")

# Configs
PORT = int(os.environ.get("LISA_PORT") or 8000)
HOST = os.environ.get("LISA_HOST") or "127.0.0.1"
LISA_URL = f"http://{HOST}:{PORT}"
RSS_LIMIT_MB = 950
CHECK_INTERVAL_SECONDS = 5
MAX_RESTART_ATTEMPTS = 5
BACKOFF_FACTOR = 2


def get_process_memory_usage(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        # Sum memory of parent and all child processes
        mem = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return mem
    except psutil.NoSuchProcess:
        return 0


def graceful_shutdown(process: subprocess.Popen) -> None:
    logger.info("Attempting graceful shutdown of LISA...")
    # Try calling the local shutdown endpoint first
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(f"{LISA_URL}/shutdown", json={"confirm": True})
            if response.status_code == 200:
                logger.info("Shutdown request accepted by LISA API.")
                # Wait for process to exit
                for _ in range(10):
                    if process.poll() is not None:
                        return
                    time.sleep(1)
    except Exception as e:
        logger.warning(f"Could not trigger shutdown via API: {e}")

    # Fallback to standard termination
    logger.info("Terminating process...")
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Graceful shutdown timed out. Force-killing process.")
        process.kill()
        process.wait()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        process.kill()


def start_lisa() -> subprocess.Popen:
    cmd = [sys.executable, "main.py"]
    logger.info(f"Spawning LISA: {' '.join(cmd)}")

    # Spawn in a new process group to handle CTRL_BREAK on Windows correctly
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        cmd,
        creationflags=creationflags,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def monitor_lisa() -> None:
    os.makedirs("logs", exist_ok=True)
    restart_attempts = 0

    while restart_attempts < MAX_RESTART_ATTEMPTS:
        process = start_lisa()
        logger.info(f"LISA started with PID {process.pid}")
        restart_attempts += 1

        # Wait for LISA to boot
        time.sleep(5)
        start_time = time.time()

        while True:
            # Check if process died
            if process.poll() is not None:
                exit_code = process.returncode
                logger.warning(f"LISA exited unexpectedly with code {exit_code}")
                if time.time() - start_time > 60:
                    restart_attempts = 0
                    logger.info(
                        "Process ran stably for >60s, resetting restart counter."
                    )
                break

            # Check memory usage
            rss = get_process_memory_usage(process.pid)
            rss_mb = rss / (1024 * 1024)

            if rss_mb > RSS_LIMIT_MB:
                logger.warning(
                    f"LISA exceeded critical memory limit: {rss_mb:.2f}MB > {RSS_LIMIT_MB}MB. Triggering restart."
                )
                graceful_shutdown(process)
                break
            elif rss_mb > 800:
                logger.warning(
                    f"LISA memory pressure detected: {rss_mb:.2f}MB > 800MB. Triggering memory shedding."
                )
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.post(f"{LISA_URL}/shed_memory")
                        logger.info(
                            f"Memory shedding status: {resp.status_code} - {resp.text}"
                        )
                except Exception as e:
                    logger.warning(f"Could not trigger memory shedding: {e}")

            # Heartbeat check
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{LISA_URL}/health")
                    if resp.status_code != 200:
                        logger.warning(
                            f"LISA health check returned status {resp.status_code}"
                        )
            except Exception as e:
                logger.warning(f"LISA health check failed: {e}")

            time.sleep(CHECK_INTERVAL_SECONDS)

        # Backoff logic
        backoff = BACKOFF_FACTOR**restart_attempts
        logger.info(
            f"Backing off for {backoff} seconds before restart #{restart_attempts + 1}..."
        )
        time.sleep(backoff)

    logger.error("Maximum restart attempts reached. Supervisor exiting.")
    sys.exit(1)


if __name__ == "__main__":
    try:
        monitor_lisa()
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user.")
