# USAGE.md: Day-to-Day Operations Guide

This guide explains how to interact with LISA, deploy tasks, manage constitutions, and coordinate plans.

---

## ⚡ Pairing with LISA

On the first start, LISA outputs a generated `LISA_BOT_SECURITY_KEY` to the CLI logs (unless defined in `.env.local`). 
1. Open your messaging client (e.g. Telegram, Slack, WhatsApp).
2. Type and send the pairing key: `/pair <LISA_BOT_SECURITY_KEY>` or simply message the security key directly.
3. LISA will bind to your user ID, saving it in `data/bound_users.json`. All other incoming messages from other users will be rejected.

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
