# Pipeline UI Design — Streamlit Demo Page

**Date:** 2026-04-16
**Scope:** `dashboard/pages/01_pipeline.py` — fix and polish for thesis demo
**Goal:** A polished single-page Streamlit UI that runs the full MLOps pipeline, streams agent progress live, and presents the HITL approval moment as the visual centrepiece.

---

## Context

The existing `01_pipeline.py` has two broken pieces:
1. HITL interrupt detection uses fragile string-search on state values instead of checking the `__interrupt__` event key
2. No approval UI is shown — the pipeline silently stalls

The other dashboard pages (Experiments, Monitoring, Chat) are out of scope for this iteration.

---

## State Machine

The page cycles through three phases stored in `st.session_state["phase"]`:

| Phase | Trigger | UI shown |
|-------|---------|----------|
| `idle` | Page load or "Run Again" | Dataset selector + Run button |
| `awaiting_approval` | `__interrupt__` event in stream | Log (frozen) + approval panel |
| `complete` | Stream finishes after resume | Log (full) + outcome banner |

Each new pipeline run gets a unique `thread_id` (`f"streamlit-run-{int(time.time())}"`) to avoid checkpoint collisions between runs. The compiled `graph` (with `InMemorySaver`) lives at module level in `mlops_graph.py` and persists within the Streamlit session.

---

## Pipeline Log

- Rendered via `st.empty()` + `st.markdown()`, growing line by line as the graph streams events
- Each `__interrupt__`-free event yields one log line per node:
  - Supervisor nodes: extract the routing target and reasoning from the last message — rendered as `→ data_validator | <reasoning>`
  - Worker nodes: rendered as `✅ [data_validator] completed`
- Log content is accumulated in `st.session_state["log_lines"]` so it survives the Streamlit re-run that occurs when the user clicks Approve/Reject
- Tool-level detail (e.g. "Loaded iris.csv — 30 rows") is not captured in the UI — loguru writes it to stderr in the terminal, which is sufficient for the demo

---

## HITL Approval Panel

Shown when `phase == "awaiting_approval"`. Rendered below the frozen log.

**Layout:**
- Bold header: `⚠️ Human Approval Required`
- Approval question from `interrupt_value["question"]`
- `st.expander("Registration details")` containing `interrupt_value["registration_summary"]`
- Two columns: green `✅ Approve` button (primary) | red `❌ Reject` button (secondary)
- If Reject is clicked: `st.text_input("Rejection reason (optional)")` appears, plus a `Confirm Rejection` button

**On approval decision:**
1. Build `resume = {"approved": True}` or `{"approved": False, "reason": reason}`
2. Call `graph.stream(Command(resume=resume), config=st.session_state["pipeline_config"])`
3. Stream remaining events into the log
4. Transition phase to `complete`

The interrupt payload keys used: `question`, `registration_summary` — matching the dict passed to `interrupt()` in `deployer_node`.

---

## Completion Banner

Shown when `phase == "complete"`, below the full log.

| Condition | Banner style | Message |
|-----------|-------------|---------|
| `deployment_decision == "approved"` | `st.success` | "Pipeline complete. Model promoted to champion." |
| `deployment_decision == "rejected"` | `st.warning` | "Deployment rejected. Reason: `<reason>`" |
| Pipeline finished early (validation/eval failure) | `st.error` | Last supervisor message content |

A `🔄 Run Again` button below the banner calls `st.session_state.clear()` and `st.rerun()` to reset to `idle`.

---

## Out of Scope

- Pages 02–04 (Experiments, Monitoring, Chat) — pending future iteration
- Real-time tool-level log lines in the UI
- Multiple concurrent pipeline runs
- Persistent run history across browser sessions
