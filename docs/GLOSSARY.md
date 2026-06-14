# GLOSSARY.md: Vocabulary & Terms

This glossary defines terms and concepts key to the LISA agent stack architecture.

---

## Terms

* **Autonomous Mode**: An operating mode where LISA queries its own environmental state when idle to propose, utility-score, and execute goals without user prompt trigger.
* **Capability Token**: Cryptographically signed permission strings that allow/disallow LISA tools from accessing specific APIs.
* **Cognitive Personas**: Behavioral weights loaded into the LLM context to direct focus (e.g. Guardian for safety, Oracle for detail, Architect for design).
* **Dual Constitution**: Security enforcement modes dividing safety rules between `restricted` (highly-safe default) and `unrestricted` (developer research).
* **Episodic Memory**: Log records of interactions and tool executions stored in SQLite WAL database tables.
* **ChannelAccessController**: The channel authorization layer that persists per-source allow-lists to `data/channel_access.json`.
* **Dashboard Session Token**: A short-lived server-issued token required for `/personal`, protected dashboard endpoints, and dashboard websockets.
* **LISA_BOT_SECURITY_KEY**: A deprecated pairing-era key that is still accepted as a bootstrap credential for dashboard session issuance.
* **MCP (Model Context Protocol)**: An open standard protocol allowing LISA to declare, connect, and invoke schemas from remote data sources and tools.
* **Notepad**: LISA's episodic storage engine, wrapping SQLite reads, writes, and search index updates.
* **Plan DAG**: Directed Acyclic Graph structures describing a multi-step task and its dependent child nodes.
* **Practice Arena**: Isolated container sandboxes where evolution loops test synthesized code modules before deployment.
* **Signed Gating Artifacts**: The `gating_model.json`, `gating_model.npz`, and `gating_model.sig` files that replace raw pickle deserialization for persona gating.
* **Taint Tracking**: System tagging that marks data arriving from external webhooks as untrusted. Taint markers propagate through processing, preventing auto-execution of risky tasks.
* **Task Conductor**: The execution priority queue controller that coordinates background task processing and schedules concurrent runs.
