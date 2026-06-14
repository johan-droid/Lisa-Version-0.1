# TROUBLESHOOTING.md: Diagnostic & Recovery Guide

This document lists solutions to common errors, log analysis techniques, and emergency recovery instructions.

---

## 🔍 Diagnostic Commands Reference

To check the active system status, execute the following commands in the workspace root:

```bash
# Check if the database and WAL mode are initialized
sqlite3 data/lisa_notepad.db "PRAGMA journal_mode;"

# Manually verify that the local LLM model can load
.venv/bin/python main.py --check-model

# Review the supervisor and main process logs
tail -n 100 logs/lisa.log
```

---

## 🚨 Troubleshooting Scenarios

### 1. "Model failed to load (GGML read error)"
* **Cause**: Quantized model file is corrupt, incomplete, or incompatible with the CPU instruction set.
* **Solution**: Re-download the GGUF model from HuggingFace. Ensure the download completes fully. Verify file size is ~600MB-1.1GB depending on quant.

### 2. "Docker sandboxing failed or is disabled"
* **Cause**: The local user does not have permission to communicate with the Docker socket, or the Docker daemon is stopped.
* **Solution**: Run `docker ps` to verify connection. On Linux, run `sudo usermod -aG docker $USER` and log back in.

### 3. "Telegram / Slack bot is not responding"
* **Cause**: Incorrect API tokens, webhook configuration, or network blockage.
* **Solution**: Check that the token in `.env.local` is correct. Use curl to verify outbound connectivity to `api.telegram.org` or `slack.com`.

### 4. "Session token is required" or dashboard keeps reconnecting
* **Cause**: The dashboard shell loaded, but the protected websocket or `/personal` feed has no valid short-lived session cookie.
* **Solution**: Open `/dashboard/live` and mint a fresh session with `LISA_ADMIN_API_TOKEN` or `LISA_BOT_SECURITY_KEY`. If the session expires, re-authenticate and reconnect.

### 5. "Another main instance is already running" or repeated Telegram conflicts
* **Cause**: A second `main.py`, `supervisor.py`, or `scratch/telegram_bridge.py` instance tried to start while a lock was already held.
* **Solution**: Check `data/locks/` and the owning PID in the lock file. Stop the stale process cleanly before restarting. The bridge now exits instead of creating a webhook/delete retry storm.

---

## 🔄 Emergency Reset Procedures

If LISA gets caught in a state loop or experiences database corruption:
```bash
# 1. Stop the supervisor process
# Press Ctrl+C on supervisor console, or kill PID.

# 2. Backup and remove corrupted DB files
mv data/lisa_notepad.db data/lisa_notepad.db.bak

# 3. Clean up the evolution staging directory
rm -rf data/evolution/staging/*

# 4. Restart supervisor
python supervisor.py
```
This forces LISA to recreate a clean SQLite schema and reload skills.
