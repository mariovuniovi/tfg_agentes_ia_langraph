---
name: run-benchmark
description: Run the MLOps benchmark suite and compare experience pool before/after. Use when evaluating agent changes, comparing strategies, or verifying regressions before merging.
---

# run-benchmark

## Workflow

- [ ] **1. Snapshot pool before run**

  ```bash
  uv run python scripts/pool_snapshot.py
  ```

  Save this JSON as the **pre-run snapshot**.

- [ ] **2. Run the benchmark**

  Use the Agent tool to dispatch the `test-runner` subagent defined in `.claude/agents/test-runner.md`.
  Instruct it to execute:

  ```bash
  uv run python scripts/run_benchmark.py
  ```

  Wait for the structured test report.

- [ ] **3. Snapshot pool after run**

  ```bash
  uv run python scripts/pool_snapshot.py
  ```

  Save this JSON as the **post-run snapshot**.

- [ ] **4. Diff the snapshots**

  For each `dataset_name` in entries:
  - Compare `champion_model` (changed? → new winner or regression)
  - Compare `validation_score` (delta = post − pre)
  - New entries (not in pre) → fresh baselines

- [ ] **5. Summarize results**

  Report per dataset:
  - **Win** — higher score or new champion
  - **Regression** — score dropped
  - **Unchanged** — same champion, score delta < 1%
  - **New entry** — first result for this dataset

- [ ] **6. Flag regressions**

  If any dataset shows a score drop > 5%:
  > ⚠️ **BLOCKING REGRESSION** on `<dataset>`: score dropped from X to Y (−Z%). Requires user decision before merging.

  Do NOT auto-fix regressions — surface them for the user.

## Notes

- Benchmark results persist in SQLite (`settings.experience_db_path`) and MLflow experiment `mlops-agents-benchmark`
- If the pre-run pool was empty, all post-run entries are new baselines — no regressions to report
- `scripts/pool_snapshot.py` must exist (created in Task 2 of the project setup)
