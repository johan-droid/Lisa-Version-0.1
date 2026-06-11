# CONSTITUTION.md: Ethical & Operational Framework

This document outlines the dual safety constitution, risk formula guidelines, and capability token system rules of LISA.

---

## 🛡️ Dual Constitution System

### 1. Restricted Mode (Default)
In Restricted Mode, LISA is strictly locked to developer-assistance boundaries:
* All shell executions are isolated inside a constrained Docker sandbox.
* Host file system edits are restricted to the active workspace.
* External API keys must not be logged or serialized in plaintext.
* Risk scores above `0.6` trigger a prompt requiring user approval.

### 2. Unrestricted Mode (Experimental)
Unrestricted mode can be enabled for lab research or self-refactoring:
* Activation Command: `ENABLE UNRESTRICTED MODE [reason]`.
* The reason is committed directly to the SQLite audit log.
* Extended file writing permissions outside the workspace directory are permitted (but still restricted from modifying system libraries).
* Network access to arbitrary domains is enabled.
* Critical system warnings are continuously broadcasted to the user interfaces.

---

## 🔒 Risk Evaluation Engine
Before any action, a risk score $R$ is calculated:
$$R = \max(\text{Irreversibility}, \text{Taint}) \times 0.7 + (1 - \text{Confidence}) \times 0.3$$

### Threshold Boundaries

* **$R < 0.4$ (Low)**: Automatically execute and log.
* **$0.4 \le R \le 0.6$ (Medium)**: Execute, log, and raise visual warnings on the dashboard.
* **$R > 0.6$ (High)**: Pause DAG task progression, send a Telegram/Slack/WhatsApp confirmation button prompt, and wait for confirmation.

---

## 🔑 Capability Token System
Tools require signed cryptographic capability tokens matching specific scopes:
* `file:read` / `file:write`: Permits reading and writing to files.
* `net:fetch`: Permits fetching web resources.
* `terminal:exec`: Permits running shell commands inside Docker.

Tokens are issued at session start and expire after 2 hours.
