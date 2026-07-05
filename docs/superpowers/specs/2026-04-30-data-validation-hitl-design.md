# Data Validation HITL Gate — Design Spec

**Date:** 2026-04-30
**Branch:** feature/fastapi-backend

## Overview

Add a Human-in-the-Loop gate after the data validation step. Once the data agent produces the cleaned dataset, the pipeline pauses and shows the user a dataset preview (shape, columns, sample rows, validation summary) in the Dataset tab of the ResultsDashboard. The user can approve (proceed to training) or reject with an optional comment (re-route back to the data agent for another attempt). The supervisor orchestrates the retry loop as it does all routing.

A single `max_attempts_per_agent` config setting limits how many times each agent (including retries triggered by HITL rejection) can be invoked. The supervisor enforces this limit before routing.

---

## Design Decisions

- **HITL placement:** `interrupt()` inside `data_validator_node`, after the agent finishes and results are extracted. Follows the exact same pattern as the existing deployer HITL.
- **Retry orchestration:** Supervisor handles it. On rejection, the comment is injected as a `HumanMessage` into the shared message history; the supervisor sees it and re-routes to `data_validator`. No special-case routing needed.
- **Max attempts:** Single config value (`max_attempts_per_agent: int = 3`) applied uniformly to all agents. Supervisor checks the per-agent attempt count in state before routing and forces `END` if the limit is reached.
- **UI placement:** Review panel integrated at the bottom of the Dataset tab (not a separate overlay), consistent with the "Option B" decision from brainstorming.
- **Deployment HITL unchanged:** `HITLGate` component continues to handle deployment approval; it only renders when `interruptValue.type !== "data_validation"`.

---

## Changes

### 1. Config — `src/mlops_agents/config/settings.py`

Add one field:

```python
max_attempts_per_agent: int = 3
```

Applies to all agents. An "attempt" is any invocation of the agent node, including the initial call. `3` means first call + 2 retries.

---

### 2. State — `src/mlops_agents/state/agent_state.py`

Replace `retry_count: int` with:

```python
agent_attempt_counts: dict[str, int]  # {"data_validator": 1, "trainer": 2, …}
```

The **supervisor** increments the count for the target agent before routing to it. Worker nodes do not touch the count. This avoids double-counting: nodes with `interrupt()` are re-run from scratch on resume by LangGraph, so any increment inside the node would fire twice per user-visible attempt.

---

### 3. LangGraph graph — `src/mlops_agents/graphs/mlops_graph.py`

#### 3a. `data_validator_node` — add HITL interrupt

After the agent runs and results are extracted, before returning:

1. Read the processed CSV (`dataset_path`, up to 20 rows) into a preview dict.
2. Call `interrupt()` with:

```python
{
    "type": "data_validation",
    "question": "Review the processed dataset before training begins.",
    "dataset_preview": {
        "shape": [n_rows, n_cols],
        "columns": [{"name": str, "dtype": str}, ...],
        "sample_rows": [dict, ...]   # up to 20 rows
    },
    "validation_summary": {
        "passed": bool,
        "missing_values": dict,      # from quality_report
        "schema_validated": bool
    }
}
```

3. On **approved** — return `Command(goto="supervisor", update={...})` with the normal state fields.
4. On **rejected** — append a `HumanMessage` with the rejection comment, set `validation_passed: False`, return `Command(goto="supervisor", update={...})`.

The node is safe to re-run on resume: all data tools are deterministic (read from files), so re-execution produces the same results. `interrupt()` returns the resume value without re-pausing on the second run.

#### 3b. Supervisor node — increment count and enforce max attempts

After deciding the target but before incrementing, check if routing would exceed the limit:

```python
counts = dict(state.get("agent_attempt_counts") or {})
if counts.get(target, 0) >= settings.max_attempts_per_agent:
    logger.warning(f"Max attempts reached for {target} — forcing END")
    return Command(goto=END, update={"next": "FINISH"})
counts[target] = counts.get(target, 0) + 1
return Command(goto=target, update={"next": target, "agent_attempt_counts": counts})
```

---

### 4. API — models, store, router, pipeline

#### `api/models/run.py` — `HITLDecision`

Add `comment: str = ""` field (used for data validation rejection; ignored by deployment HITL).

#### `api/services/run_store.py` — `RunEntry`

Add `hitl_comment: str = ""` field.

#### `api/routers/runs.py` — `POST /runs/{run_id}/approve`

Store `entry.hitl_comment = body.comment` alongside `entry.hitl_decision = body.decision`.

#### `api/services/pipeline.py`

Two fixes:

1. **HITL agent label:** derive from interrupt payload instead of hardcoding `"deployer"`:
   ```python
   hitl_type = interrupt_val.get("type", "deployer")
   hitl_event["agent"] = hitl_type
   ```

2. **Multiple HITL rounds:** change single `if` block to a `while` loop so data validation rejections (which trigger a new interrupt after retry) are handled automatically:
   ```python
   while entry.status == "awaiting_approval":
       entry.hitl_event = asyncio.Event()   # fresh event for each round
       await entry.hitl_event.wait()
       entry.status = "running"
       resume = {
           "approved": entry.hitl_decision == "approve",
           "comment": entry.hitl_comment,
       }
       await asyncio.wait_for(_stream(Command(resume=resume)), timeout=_STREAM_TIMEOUT)
   ```

---

### 5. Frontend

#### `frontend/types/api.ts`

Add `data_validation` to the interrupt value type discriminant so components can distinguish HITL types:

```ts
export type HITLType = "data_validation" | "deployment"

export interface DataValidationInterrupt {
  type: "data_validation"
  question: string
  dataset_preview: {
    shape: [number, number]
    columns: { name: string; dtype: string }[]
    sample_rows: Record<string, unknown>[]
  }
  validation_summary: {
    passed: boolean
    missing_values: Record<string, number>
    schema_validated: boolean
  }
}
```

#### `frontend/hooks/use-approve.ts`

Pass `comment` in the mutation body:

```ts
approveRun(runId, { decision, comment: comment ?? "" })
```

#### `frontend/components/pipeline/ResultsDashboard.tsx`

In the Dataset tab, add a **Dataset Review panel** at the bottom, rendered only when `hitlPending && interruptValue?.type === "data_validation"`.

Panel contents (matching the approved mockup):
- Header: "Dataset Review" label + "awaiting approval" badge + attempt counter (dots: N of max filled)
- Sub-text: "Approve to proceed to training, or reject with a comment so the data agent can fix the issue and reprocess."
- Optional comment `<textarea>`
- **Approve dataset** button (green) + **Reject & retry** button (red)
- Attempt counter reads `interruptValue.attempt` (integer, included in the interrupt payload by `data_validator_node`)

#### `frontend/components/pipeline/HITLGate.tsx`

Add guard so it only renders for deployment HITL:

```tsx
if (interruptValue?.type === "data_validation") return null
```

---

## Event flow (updated)

```
1. data_validator_node runs agent, extracts results
2. interrupt() emits hitl_request event { type: "data_validation", dataset_preview, validation_summary }
3. Frontend receives event → sets hitlPending=true, interruptValue
4. ResultsDashboard Dataset tab shows review panel
5. User clicks Approve/Reject (+ optional comment)
6. POST /runs/{id}/approve { decision, comment }
7. pipeline.py resumes with Command(resume={ approved, comment })
8. data_validator_node: approved → supervisor routes forward
                         rejected → HumanMessage injected, supervisor re-routes to data_validator
9. If attempt count >= max_attempts_per_agent → supervisor forces END
```

---

## Attempt counter display

The review panel shows how many attempts have been used (e.g. "Attempt 2 of 3" with filled dots). The current attempt number is derived from the `agent_attempt_counts.data_validator` field, which must be included in the `hitl_request` event payload by the pipeline service.

Pipeline service adds it when building the hitl event:
```python
hitl_event["data"]["attempt"] = interrupt_val.get("attempt", 1)
```

And `data_validator_node` includes the current count in the interrupt payload:
```python
"attempt": counts.get("data_validator", 1)
```

---

## What is NOT in scope

- Paginating the sample rows table (capped at 20 rows)
- Downloading the processed CSV from the UI
- Per-agent different max attempt limits
- Persisting HITL decisions to a database
