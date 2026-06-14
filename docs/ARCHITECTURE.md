# ARCHITECTURE.md: Core Engine & Subsystem Blueprint

This document details the high-level design, data flow, concurrency patterns, and safety boundaries of the LISA agent stack.

---

## 🏛️ High-Level System Architecture

```
                                +-------------------+
                                |    Supervisor     |
                                |     (Watchdog)    |
                                +---------+---------+
                                          | IPC
                                          v
                                +---------+---------+
                                |    Main Loop      |
                                | (1GB RAM Process) |
                                +----+----+----+----+
                                     |    |    |
           +-------------------------+    |    +-------------------------+
           v                              v                              v
+----------+----------+       +-----------+-----------+       +----------+----------+
|  Cognitive Brain    |       |    Memory Tier        |       |    Conductor DAG    |
| (TinyLlama/Persona) |       | (SQLite/FAISS/Proced) |       | (Priority / State)  |
+----------+----------+       +-----------+-----------+       +----------+----------+
           |                              |                              |
           +------------------------------+------------------------------+
                                          |
                                          v
                              +-----------+-----------+
                              |      Tool Layer       |
                              | (Docker / MCP Client) |
                              +-----------------------+
```

---

## 📦 Component Inventory

### 1. Supervisor Process
The `Supervisor` acts as a process monitor. It runs in a separate lightweight process, spawning LISA's main application. The supervisor and main runtime now both take explicit lock files under `data/locks/` so a second instance exits cleanly instead of creating duplicate watchdogs or Telegram pollers.

### 2. Main Process (1GB RAM Cap)
LISA runs inside a single Python process managed by an `asyncio` event loop. Performance tuning flags (like OMP/MKL thread counts) are restricted to prevent CPU over-subscription.

### 3. Cognitive Brain (Persona-Gated LLM)
Cognitive blending is achieved through dynamic soft prompt vector interpolation (PersonaSoftPromptBank) combined with a gating neural network. The gating model is now stored as signed `json` + `npz` artifacts and legacy pickles are migrated one time on load. The five core personas are:
* **Architect**: Focuses on task decomposition, DAG generation, and repository layouts.
* **Oracle**: Specializes in detailed code construction and algorithm synthesis.
* **Guardian**: Enforces constraints, filters malicious input, and reviews risk scores.
* **Evolution Engine**: Handles nightly code refactoring and tool testing.
* **Distributed Mind**: Monitors environment metrics and generates autonomous goals.

### 4. Memory Architecture
* **Working Memory**: In-memory dict containing short-term conversation logs and immediate variables.
* **Episodic Memory**: A SQLite transactional database (`interactions` table) operated in Write-Ahead Logging (WAL) mode to allow simultaneous reads and writes without blocking.
* **Semantic Memory**: Cosine similarity indexer computed via lightweight numpy vector math to find semantic matches from past tasks without requiring FAISS C++ binaries.
* **Procedural Memory**: Saved and compiled skill manifests (`skills/` directory) containing auto-generated Python functions that can be imported dynamically.

### 5. Tool Layer
Tools are separated into two lists:
* **Core Built-in Tools**: File operations, SQLite queries, and web browser scraping.
* **External Sandboxed Tools**: Executed inside an isolated Docker container with network restrictions, CPU limits (1 core), and memory boundaries (512MB).
* **MCP Client**: Standard JSON-RPC 2.0 communication to coordinate with external Model Context Protocol servers.

### 6. Task Conductor & Plan DAG
Abstract user input is analyzed by the `PlanDAGEngine`. It generates a Directed Acyclic Graph of independent work steps. The Conductor schedules frontier tasks across up to 10 concurrent execution arms using a Semaphore constraint.

### 7. Safety & Boundaries
* **Dashboard Session Tokens**: `/personal`, dashboard data endpoints, and `/ws/*` require a short-lived server-issued session token rather than trusting a bare query parameter or open websocket upgrade.
* **Localhost by Default**: `HOST` defaults to `127.0.0.1`; remote bind is an explicit opt-in through `LISA_ALLOW_REMOTE_BIND=true`.
* **Channel Access Control**: Per-channel authorization is persisted atomically in `data/channel_access.json` through `ChannelAccessController`.
* **Taint Tracking**: Inbound webhook messages from external sources are labeled as `tainted`. Taint flags propagate through string parsing operations. Any terminal execution using tainted parameters is automatically paused for Human-In-The-Loop (HITL) verification.

---

## 🔄 Core Data Flows

### A. Inbound Message to Response
```
[User Webhook] -> [MessageHub Signature Check] -> [ChannelAccessController]
      |
      v
[TaskConductor Queue] -> [Brain Gating Network Blending] -> [LLM Client Call]
      |
      v
[Response Dispatch] <- [Notepad Logging] <- [Tool Executor Execution]
```

### B. Nightly Evolution Loop
```
[Evolution Timer Trigger] -> [Scan Failed Interaction Logs] -> [Decompose Skill Gaps]
      |
      v
[Synthesize Skill Code] -> [Run Tests in Docker Sandbox] -> [Deploy / Register Manifest]
```

---

## 💾 Memory Management Strategy
LISA manages its current RAM budget using a tiered eviction system:
1. **Soft Eviction (RSS > 750MB)**: EVM model layer unloads soft prompt weights and clears HTTP connection pools.
2. **Hard Eviction (RSS > 850MB)**: Invokes `gc.collect()` and evicts browser cache files to database storage.
3. **Emergency Offloading (RSS > 950MB)**: Suspends all parallel tasks, saves a state snapshot, and forces a worker restart.

---

## 🔒 Security Model & Risk Boundaries
Risk calculations evaluate tool calls before execution:
$$\text{Risk Score} = \max(\text{Irreversibility}, \text{Taint}) \times 0.7 + (1 - \text{Confidence}) \times 0.3$$

| Risk Range | Action | Execution Path |
|---|---|---|
| $0.0 - 0.3$ | Auto-Execute | Background execution, log to audit trail. |
| $0.4 - 0.6$ | Warning | Run, update real-time dashboard with warning banner. |
| $0.7 - 1.0$ | Gate | Pause task, dispatch Slack/Telegram approval hook, wait 30m. |
