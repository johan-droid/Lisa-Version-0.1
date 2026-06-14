# USAGE.md: Day-to-Day Operations Guide

This guide explains how to interact with LISA, deploy tasks, manage constitutions, and coordinate plans.

---

## ⚡ Channel Access and Dashboard Sessions

LISA no longer uses `data/bound_users.json`. Channel authorization is persisted through `lisa/channel_access.py` into `data/channel_access.json`.

1. Configure allow-lists with `LISA_TELEGRAM_ALLOWED_USER_IDS`, `LISA_SLACK_ALLOWED_USER_IDS`, or `LISA_WHATSAPP_ALLOWED_USER_IDS`, or use the admin endpoints to authorize users.
2. Open `/dashboard/live` in a browser.
3. Enter the current `LISA_ADMIN_API_TOKEN` or `LISA_BOT_SECURITY_KEY` to mint a short-lived session for `/personal`, `/dashboard/snapshot`, `/tools`, `/notepad/search`, `/v1/channels`, and `/ws/*`.

The dashboard HTML shell is public, but the data feeds behind it are not.

---

## 💬 Core Interaction Commands

### 1. Code Generation
* **Format**: `generate a [language] script to [description]`
* **Example**: `"generate a python script to parse logs and extract error counts"`

### 2. Code Review & Audits
* **Format**: `audit [file_path]`
* **Example**: `"audit safety/risk_controller.py for resource leaks"`

### 3. Constitution Gating Mode Switch
* **Restricted Mode**: Default, safe. Limits tool scopes.
* **Unrestricted Mode**: Enable command: `ENABLE UNRESTRICTED MODE [reason]`. In unrestricted mode, tool executing risk limits are extended to allow experimental actions.

---

## 🎭 Persona Blending Guide
You can manually direct LISA to adjust its persona weights to influence its responses:
* **Architect**: Focuses on design structure.
* **Oracle**: Writes implementation detail.
* **Guardian**: Checks vulnerabilities.
* **Evolution Engine**: Refactors skill files.
* **Distributed Mind**: Prioritizes autonomous workflows.

To manually trigger a persona:
* `"Write a script to upload images to S3 (Dominant Persona: Architect)"`

---

## 🤖 Autonomous Plan Execution

You can assign LISA multi-step goals:
* `"Goal: Refactor database models to support soft delete and write tests"`

LISA will:
1. Decompose the goal into a Plan DAG.
2. Output the plan node-by-node.
3. Pause for approval if a high-risk tool is hit (e.g. deleting files).
4. Run through the DAG and report status on completion.

---

## 🔐 Security Posture at a Glance

Safe-by-default runtime behavior:
* Control plane bind defaults to `127.0.0.1`.
* Non-loopback bind requires `LISA_ALLOW_REMOTE_BIND=true` plus a bootstrap credential.
* Unsafe runtime admin routes are not mounted unless `LISA_ENABLE_UNSAFE_ADMIN_ENDPOINTS=true`.
* Gating models load only from signed `json` + `npz` artifacts after migration.
