# API_REFERENCE.md: Core Subsystems Developer Guide

This document describes the primary class interfaces, method parameters, and extension hooks for developer expansion.

---

## 🧠 Brain API: `PersonaGatedModel`
The core LLM inference wrapper handles persona soft prompt injection.

```python
from lisa.local_inference import PersonaGatedModel, LocalGenerationRequest

# Initialize a gated model
model = PersonaGatedModel(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    persona_bank=persona_bank,
    context_size=2048
)

# Run generation
request = LocalGenerationRequest(
    system_prompt="You are a senior code reviewer.",
    user_prompt="Audit this file.",
    max_tokens=256,
    persona_prefix=persona_prefix
)
result = await model.generate(request)
print(result.text)
```

---

## 🛠️ Tool API: Custom Tool Creation
To add a new tool to LISA, register it inside `lisa/tools.py` using the `register_tool` decorator or signature:

```python
from typing import Any
from lisa.tools import ToolContext, ToolRegistry, ToolSpec

async def get_system_uptime(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Retrieve local system uptime metrics."""
    import time
    return {"uptime_seconds": time.monotonic()}

# Registering tool in registry
registry.register_tool(
    ToolSpec(
        name="system_uptime",
        description="Gets system uptime metrics",
        restricted_safe=True,
        handler=get_system_uptime
    )
)
```

---

## 💾 Memory API: `Notepad` Database Querying
The `Notepad` class wraps SQLite interaction.

* **Method**: `log_entry(entry_type, payload, constitution, personas)`
  - Logs a task step, user interaction, or tool call to database tables.
* **Method**: `latest_entries(limit)`
  - Returns the last $N$ logged entries from the database.
* **Method**: `search(query, limit)`
  - Performs FTS5 full-text search across logged interactions.
* **Method**: `search_semantic(query_vector, limit)`
  - Executes numpy-based cosine similarity checks on vectorized notes.
