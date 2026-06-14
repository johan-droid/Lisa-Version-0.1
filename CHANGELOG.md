
## [1.1.0] - Security & Stability Updates
### Fixed
* **WebSocket Authentication:** Added strict session token verification for `/ws/dashboard`, `/ws/events`, and `/personal` endpoints. Upgrades are now rejected *before* `accept()` if the token is invalid or missing.
* **Network Binding:** LISA now binds to `127.0.0.1` (localhost) by default to prevent unintentional network exposure. Binding to non-loopback interfaces requires explicit configuration and an admin API token.
* **Crash-Loop & Concurrency:** Fixed an issue where multiple instances of `telegram_bridge.py`, `supervisor.py`, or `main.py` could spawn concurrently. Introduced an atomic PID file lock (`data/lisa.pid`) to enforce a single-instance policy.
* **Polling Storms:** Added an exponential backoff mechanism to the Telegram webhook deletion loop to prevent API rate-limiting during connection conflicts.
* **API Error Handling:** Implemented a top-level exception handler in `lisa/api.py` to catch unhandled errors and return structured JSON responses with tracebacks (logged securely) rather than opaque `500 Internal Server Error` messages.
* **File Locking Race Conditions:** Implemented atomic tempfile-rename-under-lock behavior for `ChannelAccessController` in `lisa/channel_access.py` to prevent data corruption during concurrent file system access.
* **Safe Model Serialization:** Replaced raw `pickle` serialization for the Persona Gating Network with safe `numpy` (`.npz` and `.json`) formats, complete with HMAC integrity signing to prevent tampering. Added automatic migration from legacy `.pkl` files.
* **Admin Endpoint Guards:** Updated `require_admin_request` in `safety/admin_auth.py` to strictly enforce the `enable_unsafe_admin_endpoints` flag dynamically, ensuring unsafe actions remain blocked unless explicitly permitted.
* **Goal Thrashing Resolution:** Inserted a permanent `evolution_goal_resolved` audit log for `safety/admin_auth.py` into the episodic memory database to stop the autonomous conductor from constantly re-evaluating the module.
