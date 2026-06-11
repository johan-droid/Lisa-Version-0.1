# EVOLUTION.md: Self-Directed Code Refactoring & Learning

LISA features an autonomous, nightly evolution loop to continuously improve its skills and address execution failures.

---

## 🔄 The Nightly Evolution Cycle

The evolution cycle is managed by `NightlyEvolutionScheduler` and runs during the scheduled window (default: 3:00 AM to 5:00 AM).

```
[3:00 AM Start]
      |
      v
[Notepad Audit] ---> Identify failures or low-reward task interactions.
      |
      v
[Skill Synthesis] -> Write new python modules inside data/evolution/staging/
      |
      v
[Practice Arena] --> Launch Docker container and run test assertions.
      |
      v
[Deployment] ------> Move validated skill to skills/ and reload manifest.
```

---

## 🧪 Practice Arena & Sandboxing
Before any generated skill code is deployed:
1. LISA writes test assertions for the synthesized Python function.
2. An isolated Docker sandbox is launched.
3. The newly generated code and tests are mounted.
4. The test command is executed.
5. If tests fail, the evolution loop initiates a self-correction pass (up to 3 times) before discarding the candidate.

---

## 📈 Evolution Performance Tracking
Evolution outcomes are logged directly to SQLite:
* **Metric**: Evolution Reward Score. Calculated based on test success rate and performance profile delta.
* **Rollback Trigger**: If a newly deployed skill causes an unhandled exception or process panic in the main application flow, the supervisor automatically rolls back the `skills_manifest.json` configuration to the previous Git commit hash.
