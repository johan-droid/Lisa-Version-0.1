# SECURITY.md: LISA Agent Security Architecture

This document details the security posture, boundary enforcements, and threat models for the LISA agent stack.

---

## 🔒 Security Posture at a Glance
LISA follows a strict "defense in depth" philosophy:
* **Default Local-Only**: The agent binds exclusively to `127.0.0.1` by default, protecting it from network exploitation. Non-loopback binding requires explicit administrator authorization tokens.
* **Dual Constitution**: Workflows are divided into Restricted and Unrestricted modes.
* **Taint Tracking & Capability Gating**: Content originating from the web or other agents is tagged as `tainted`. High-risk tool calls (like shell execution) requested while handling tainted input are blocked and require Human-in-the-Loop (HITL) approval.

---

## 🛡️ Lessons from CVE-2026-25253 (ClawJacked)
LISA’s architecture includes specific defenses designed to prevent the class of vulnerabilities that affected OpenClaw (CVE-2026-25253):

* **Strict Endpoint Authentication**: All administrative and dashboard endpoints—including WebSockets (`/ws/dashboard` and `/ws/events`)—strictly require short-lived session token authentication. LISA does not trust connections implicitly. The token is derived from the pairing flow, rather than relying on a static shared secret, and must be verified *before* the server accepts the WebSocket connection.
* **Atomic Concurrency Controls**: LISA implements strict OS-level process locks and exponential backoff mechanisms. This prevents multi-instance polling storms (which can lead to denial-of-service and state corruption) and ensures exactly one active bridge instance holds the connection state to external APIs like Telegram.
* **Safe Model Serialization**: The internal Persona Gating Network is serialized using safe `numpy` formats (`.npz` and `.json`), signed with HMAC to prevent tampering. LISA explicitly rejects raw Python `pickle` files for model storage to eliminate a common arbitrary-code-execution vector.
