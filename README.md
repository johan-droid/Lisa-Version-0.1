# LISA: Self-Evolving, Autonomous AI Developer Agent

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](#)

LISA is a self-directed, self-evolving autonomous AI developer agent designed to operate inside a resource-constrained 1GB RAM environment as a single Python asyncio process.

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
                       |   | Hierarchical Mem  |   |  Tool Sandbox   |   |
                       |   | (FAISS/SQLite/WAL)|   |  (Docker/MCP)   |   |
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
* **Hierarchical Memory Store**: Multi-tier memory model utilizing working memory, SQLite WAL transactional episodic logs, FAISS semantic indexers, and local procedural skills.
* **Nightly Evolution Loop**: Proactively self-analyzes execution logs and synthesizes new tools, testing them inside clean Docker containers before deploying them.
* **Self-Repair Watchdog**: Identifies resource lockups, database corruption, or LLM timeouts and automatically routes tasks to failover pipelines.

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.11 or 3.12
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
