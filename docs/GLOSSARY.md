# GLOSSARY.md: Vocabulary & Terms

This glossary defines terms and concepts key to the LISA agent stack architecture.

---

## Terms

* **Autonomous Mode**: An operating mode where LISA queries its own environmental state when idle to propose, utility-score, and execute goals without user prompt trigger.
* **Capability Token**: Cryptographically signed permission strings that allow/disallow LISA tools from accessing specific APIs.
* **Cognitive Personas**: Behavioral weights loaded into the LLM context to direct focus (e.g. Guardian for safety, Oracle for detail, Architect for design).
* **Dual Constitution**: Security enforcement modes dividing safety rules between `restricted` (highly-safe default) and `unrestricted` (developer research).
* **Episodic Memory**: Log records of interactions and tool executions stored in SQLite WAL database tables.
* **LISA_BOT_SECURITY_KEY**: A randomly generated security token used to pair a specific chat client user (Telegram, WhatsApp, Slack) to LISA, locking out other incoming messages.
* **MCP (Model Context Protocol)**: An open standard protocol allowing LISA to declare, connect, and invoke schemas from remote data sources and tools.
* **Notepad**: LISA's episodic storage engine, wrapping SQLite reads, writes, and search index updates.
* **Plan DAG**: Directed Acyclic Graph structures describing a multi-step task and its dependent child nodes.
* **Practice Arena**: Isolated container sandboxes where evolution loops test synthesized code modules before deployment.
* **Taint Tracking**: System tagging that marks data arriving from external webhooks as untrusted. Taint markers propagate through processing, preventing auto-execution of risky tasks.
* **Task Conductor**: The execution priority queue controller that coordinates background task processing and schedules concurrent runs.
