# SETUP.md: Complete Installation & Configuration Guide

This guide details the complete installation steps, configuration keys, environment variables, and troubleshooting workflows for LISA.

---

## 🛠️ Prerequisites
Ensure your local development environment meets the following specifications:
* **Operating System**: Windows 10/11, macOS 13+, or Ubuntu 22.04+
* **Python**: 3.11.x or 3.12.x
* **Containerization**: Docker Desktop or Docker Engine installed and running
* **Git**: Version 2.40+

---

## 🚀 Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/user/lisa.git
cd lisa
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# Activate on Linux/macOS:
source .venv/bin/activate
# Activate on Windows:
.venv\Scripts\activate
```

### 3. Install Core Dependencies
Install the required packages from the root directory:
```bash
pip install -r requirements.txt
```

### 4. Fetch the GGUF Model
Create the models directory and download the TinyLlama quantized model:
```bash
mkdir models
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### 5. Configure Local Settings
Copy the env example file and verify your tokens:
```bash
cp .env.example .env.local
```
Open `.env.local` and customize the parameters (e.g. `LISA_BOT_SECURITY_KEY`, `LISA_TELEGRAM_BOT_TOKEN`).

### 6. Set Up Docker Sandboxing
Ensure the Docker daemon is running, and pull the python slim image:
```bash
docker pull python:3.12-slim
```

### 7. Run Setup Verification
Execute pytest to ensure all test suites and integrations compile:
```bash
python -m pytest
```

---

## ⚙️ Configuration Schema (`config.yaml`)

```yaml
settings:
  db_path: "data/lisa_notepad.db"
  skills_dir: "skills"
  persona_vectors_path: "data/persona_vectors.npz"
  gating_model_path: "data/gating_model.pkl"
  local_model_path: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  local_model_context_size: 2048
  local_model_n_gpu_layers: 0
  docker_image: "python:3.12-slim"
  enable_browser_tools: true
  autonomous_enabled: true

message_hub:
  enabled: true
  host: "localhost"
  port: 8800

evolution:
  enabled: true
  check_interval_seconds: 1800
  min_reward: 0.35
```

---

## 🔑 Environment Variables Reference

| Variable Name | Default Value | Purpose |
|---|---|---|
| `LISA_WORKSPACE_ROOT` | `.` | Root directory of the repository. |
| `LISA_BOT_SECURITY_KEY` | *(generated)* | Gating key for Telegram/Slack channel pairing. |
| `LISA_TELEGRAM_BOT_TOKEN` | `None` | Telegram HTTP API bot token. |
| `LISA_SLACK_BOT_TOKEN` | `None` | Slack Web Client API bot token. |
| `LISA_WHATSAPP_BOT_TOKEN` | `None` | WhatsApp cloud business gateway token. |

---

## 🛠️ Troubleshooting Common Setup Issues

### 1. Docker Daemon Not Found
* **Symptoms**: `terminal_exec` tool errors with "Docker is not available".
* **Solution**: Ensure Docker is running. On Windows, verify that "Expose daemon on tcp://localhost:2375 without TLS" or the WSL integration is enabled.

### 2. SQLite Database Lock Errors
* **Symptoms**: Error message `sqlite3.OperationalError: database is locked`.
* **Solution**: Ensure no other processes are locking the DB. Run `PRAGMA journal_mode=WAL;` to enable concurrent read/write execution.
