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
