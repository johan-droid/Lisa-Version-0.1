# SECURITY.md: Control Plane and Runtime Hardening

This document tracks the concrete security controls currently implemented in LISA.

---

## Localhost-First Exposure Model

* `HOST` now defaults to `127.0.0.1`.
* Binding to a non-loopback host requires `LISA_ALLOW_REMOTE_BIND=true`.
* LISA refuses to start if a non-loopback bind is requested without a configured dashboard-session bootstrap credential (`LISA_ADMIN_API_TOKEN` or `LISA_BOT_SECURITY_KEY`).

---

## Dashboard and WebSocket Authentication

The dashboard HTML shell is allowed to load without data, but the data-carrying surfaces are session-gated:

* `/personal`
* `/dashboard`
* `/dashboard/snapshot`
* `/tools`
* `/notepad/search`
* `/v1/channels`
* `/ws/events`
* `/ws/dashboard`
* aiohttp dashboard data and websocket routes when the standalone listener is enabled

Sessions are:

* server-issued
* short-lived
* signed with the same HMAC key family used for snapshots
* validated against a server-side session record before trust is granted

LISA does not accept a raw query-parameter token as sufficient proof by itself; the token must match a live server-issued session record.

---

## Lessons from CVE-2026-25253

Open websocket upgrades that trust any browser are an agent hijack primitive. LISA now rejects unauthenticated websocket upgrades for `/ws/events` and `/ws/dashboard` before `accept()`, and applies the same short-lived session requirement to dashboard snapshot and personal-context endpoints. This closes the same class of "visit a webpage, lose your agent" trust failure.

---

## Signed Gating Model Artifacts

Raw gating-model pickles were removed from normal runtime loading. LISA now stores persona-gating models as:

* `gating_model.json`
* `gating_model.npz`
* `gating_model.sig`

On first load, a legacy pickle is migrated once into the signed format and then removed. Subsequent loads verify the signature and reject tampered or unsigned artifacts.

---

## Admin Route Guarding

`require_admin_request()` performs constant-time credential comparison for admin-tagged routes. Unsafe runtime routes such as shutdown and memory shedding are not mounted at all unless `LISA_ENABLE_UNSAFE_ADMIN_ENDPOINTS=true`.

---

## Multi-Instance and Channel-Access Integrity

* `main.py`, `supervisor.py`, and `scratch/telegram_bridge.py` now use lock files under `data/locks/` to prevent duplicate runtime instances.
* `ChannelAccessController` persists `data/channel_access.json` through atomic write-and-replace behavior guarded by a lock file, reducing cross-process corruption risk.
