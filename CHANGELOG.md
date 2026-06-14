# CHANGELOG.md

## 2026-06-14

### Priority 0 security and stability hardening

* Added short-lived dashboard session authentication for `/personal`, protected dashboard data endpoints, and `/ws/*`, with websocket rejection before `accept()`.
* Switched default control-plane bind behavior to `127.0.0.1` and added startup refusal for unsafe non-loopback binds without explicit opt-in and bootstrap credentials.
* Replaced raw gating-model pickle persistence with signed `json` + `npz` artifacts, plus one-time legacy migration and tamper rejection.
* Audited `safety/admin_auth.py`, kept constant-time credential validation, and stopped mounting unsafe runtime admin routes unless explicitly enabled.
* Added cross-process lock files for `main.py`, `supervisor.py`, and `scratch/telegram_bridge.py`, plus atomic `channel_access.json` persistence to reduce duplicate-instance races and access-file corruption.
* Expanded regression coverage for session auth, websocket rejection, legacy gating migration, signature failure, and second-instance lock refusal.
