# LISA: Self-Evolving, Autonomous AI Developer Agent

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](#)

LISA is a self-directed, self-evolving autonomous AI developer agent built around a hybrid memory and routed-reasoning architecture. The active runtime uses FastAPI, Redis-style working memory, PostgreSQL episodic storage, Chroma skill retrieval, Docker sandboxes, and multi-channel delivery.

## 🏗️ System Architecture

```
                       +-------------------------------------------------+
                       |            Supervisor Process                   |
                       |  (Watchdog, Memory Shedder, Graceful Restarter) |
                       +-----------------------+-------------------------+
                                               | (Monitors & Restarts)
                                               v
                       +-------------------------------------------------+
                       |             Main Python Process                 |
                       |                                                 |
                       |   +-------------------+   +-----------------+   |
                       |   |    Cognitive      |   |  Task Conductor |   |
                       |   |   Brain Gating    |   |   & Plan DAG    |   |
                       |   +---------+---------+   +--------+--------+   |
                       |             |                      |            |
                       |             v                      v            |
                       |   +-------------------+   +-----------------+   |
                       |   | Hybrid Memory     |   |  Tool Sandbox   |   |
                       |   | (Redis/Postgres/  |   | (Docker/Scoped  |   |
                       |   |  Chroma)          |   |  Workspace/MCP) |   |
                       |   +---------+---------+   +--------+--------+   |
                       |             |                      |            |
                       |             +----------+-----------+            |
                       |                        |                        |
                       |                        v                        |
                       |             +---------------------+             |
                       |             | Nightly Evolution   |             |
                       |             |   (Practice Arena)  |             |
                       |             +---------------------+             |
                       +-------------------------------------------------+
```

## 🚀 Key Features

* **Dynamic Cognitive Personas**: Core reasoning based on quantized TinyLlama models running with dynamic soft prompt blends (Architect, Oracle, Guardian, Evolution Engine, Distributed Mind).
* **Plan DAG Execution Engine**: Decomposes abstract instructions into Directed Acyclic Graphs of dependent tasks with self-healing and task-rollback capability.
* **Dual Constitution Modes**: Strict permission control between Restricted (safe) and Unrestricted (lab/exploration) mode.
* **Hybrid Memory Store**: Working memory in Redis-style session state, episodic recall in PostgreSQL-compatible records with vector similarity, and semantic skill retrieval through Chroma collections.
* **Scored Evolution Loop**: Weak skills are evolved, sandbox-tested, scored for correctness/speed/token efficiency, and only promoted when they beat the current version.
* **Permission-Gated Tool Sandbox**: Tools are mapped to permission layers, executed in task-scoped workspaces, and audited with idempotency keys.
* **Routed ReAct Reasoning**: Tasks are enriched with episodic/skill context, routed to the right LLM brain, and solved through observe-reflect-plan-act loops.
* **Self-Repair Watchdog**: Identifies resource lockups, database corruption, or LLM timeouts and automatically routes tasks to failover pipelines.

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.11 or 3.12
* Redis
* PostgreSQL with `pgvector`
* ChromaDB persistence directory access
* Docker (for sandbox tool execution)
* Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/user/lisa.git
cd lisa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Download
Download the quantized TinyLlama Chat model to the `models/` directory:
```bash
mkdir models
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### 4. Running LISA
Initialize LISA under the supervisor watchdog:
```bash
python supervisor.py
```

### 5. Channel Control Plane
The public webhook routes are:
```text
POST /telegram/webhook
POST /slack/events
POST /whatsapp/webhook
```

The internal multiplexer routes are admin-protected with `Authorization: Bearer $LISA_ADMIN_API_TOKEN`:
```text
POST /v1/messages/dispatch
POST /v1/messages/ingest/{source}
GET  /v1/channels
POST /v1/channels/authorize
POST /v1/channels/revoke
```

Telegram command support is wired into the core runtime for `/start`, `/help`, `/status`, `/new`, and `/tools`, plus inline callback actions for the same shortcuts.

## ⚙️ Configuration
LISA is configured via `config.yaml` and `.env.local` files. Refer to [CONFIG.md](docs/SETUP.md) for a complete breakdown of configuration settings.

## 📖 Documentation Directory
* [Architecture Deep-Dive](docs/ARCHITECTURE.md)
* [Setup & Configuration](docs/SETUP.md)
* [Usage & Workflows](docs/USAGE.md)
* [API Reference](docs/API_REFERENCE.md)
* [Constitution & Safety Policies](docs/CONSTITUTION.md)
* [Evolution & Learning Loops](docs/EVOLUTION.md)
* [Troubleshooting & Recovery](docs/TROUBLESHOOTING.md)
* [Contribution Guidelines](docs/CONTRIBUTING.md)
* [Glossary of Terms](docs/GLOSSARY.md)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
