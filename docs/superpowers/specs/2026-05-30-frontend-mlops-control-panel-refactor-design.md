# Frontend MLOps Control Panel Refactor — Design Spec

**Date:** 2026-05-30
**Branch:** `feature/container` (continues current branch)
**Status:** Approved for planning

## Goal

Refactor the dashboard UI so it looks and reads like a high-quality MLOps control panel rather than a raw debug console. The UI must clearly communicate the thesis story:

> Agentic reasoning where ambiguity exists, deterministic execution where correctness matters, human approval at critical gates.

## Non-goals

- Real-time per-token cost accounting (not implemented; UI must not pretend it is)
- Post-deployment model performance monitoring (already covered by Evidently drift in `/monitoring`)
- Authentication, RBAC, multi-user state
- Refactoring the Experiments page beyond minor visual alignment
- Mobile / small-screen optimization (desktop control-panel only)
- Internationalization

## Architecture decisions (locked in brainstorming + grilling)

| Decision | Choice |
|---|---|
| Top navigation | 4 tabs: **Pipeline · Experiments · Observability · Monitoring** (Monitoring kept because drift detection is real) |
| Scope | Backend + frontend, clean-slate events (no display-layer hacks for new state) |
| Experiments page | Minimal polish only — adopt new tokens, add champion badge + problem-type subtitle; do not rebuild |
| Visual direction | Linear/Vercel-style — dense, neutral zinc, single indigo accent, monospace for IDs/metrics |
| Sequencing | 5 vertical slices, each independently mergeable and demoable |
| Token migration | Hard cut in slice 1 — `navy`/`amber` tokens deleted; all 12 affected components migrate in slice 1 |
| Dataset endpoints data source | `processed_dataset_path` stashed on `RunEntry` when dataset-approval HITL fires (no LangGraph checkpointer coupling) |
| Champion resolution | Backend resolves once via `_resolve_champion_model_name(state)`; frontend just renders the string |
| Reject comment | Required (≥ 4 chars) — silent rejection breaks the retry loop's auditability |
| Pipeline health persistence | In-memory only; card labelled `"resets on restart"`. No SQLite work this iteration |
| Eval-rejection UX | Treated as `Candidate rejected` (sky/info pill), NOT `Failed` (red). Audit tab auto-focuses; deploy stages marked `skipped` |
| Tab auto-switch | Triggered only by data-arrival events (`dataset_preview`, `planner_context`, `tool_result train_model`, `audit_report`). Manual click is a hard pin until run reset |
| Stepper retries | Stage resets to `running`/`waiting_human` on retry; attempt counter shown in run header next to active stage |

## High-level slice plan

| Slice | Frontend | Backend | Visible win |
|---|---|---|---|
| 1 | Tokens, primitives, `RunHeader`, `PipelineStepper`, supervisor→controller rename, Observability tab placeholder | Stop emitting `agent: "supervisor"` for new events | UI looks tighter; users see pipeline stages without reading logs |
| 2 | `DatasetApprovalCard` with Head/Tail/Schema/Validation tabs; Copy artifact path button | `GET /runs/{id}/dataset-preview`; `GET /runs/{id}/dataset-download`; add `tail` to HITL payload | Dataset approval is artifact-driven, downloadable |
| 3 | `AuditReportPanel` (Audit tab); `DeploymentApprovalCard` replaces JSON dump; **explicit "Deployment action" field** | Emit `audit_report` SSE event; enrich deployer HITL payload; **fix deployment_decision bug** | Deployment approval shows champion, metric, risks, and the actual action being authorized |
| 4 | `EventLog` becomes tabbed (Timeline / Tool Details / Raw); aggregation helper; Download raw trace JSON | none | Log is concise and auditable |
| 5 | `/observability` page (LLM activity / Tool usage / Pipeline health); Experiments polish | `GET /runs?limit=` | New page lands; Experiments matches new style |

Each slice ends with a runnable container build and a manual smoke test.

---

## Section 1 — Visual System & Tokens (slice 1 foundation)

### Tailwind v4 theme tokens

Added to `frontend/app/globals.css` under `@theme inline`. Replaces ad-hoc `navy` and `slate` usage across components.

```css
@theme inline {
  /* Semantic colors */
  --color-bg:            theme(colors.zinc.50);
  --color-surface:       #ffffff;
  --color-surface-2:     theme(colors.zinc.50);
  --color-border:        theme(colors.zinc.200);
  --color-border-strong: theme(colors.zinc.300);
  --color-fg:            theme(colors.zinc.900);
  --color-fg-muted:      theme(colors.zinc.500);
  --color-fg-subtle:     theme(colors.zinc.400);
  --color-accent:        theme(colors.indigo.600);
  --color-success:       theme(colors.emerald.600);
  --color-warning:       theme(colors.amber.500);
  --color-danger:        theme(colors.red.600);
  --color-info:          theme(colors.sky.600);    /* deterministic node */
  --color-llm:           theme(colors.violet.600); /* LLM/agent node */

  /* Typography */
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);

  /* Spacing/radius */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
}
```

Type scale: 11 / 12 / 13 / 14 / 16 / 20 / 24 px. No new font dependencies — Geist ships with Next.js.

### Primitives shipped in slice 1

Three components under `frontend/components/ui/`. They MUST be the basis for all later slices.

- **`<Card title actions children />`** — replaces every ad-hoc `rounded-lg border border-slate-200 bg-white p-4`. Optional title row with right-aligned actions slot.
- **`<Badge variant="success|warning|danger|info|llm|neutral" />`** — single source for status pills. Removes the ~6 scattered styles in `ResultsDashboard`, `RunStatusBadge`, `EventLog`.
- **`<NodeTypeBadge type="agent|llm|deterministic|hitl" />`** — semantic badge for stepper + run header. Maps to color tokens:
  - `agent` → llm + filled
  - `llm` → llm + outline
  - `deterministic` → info + outline
  - `hitl` → warning + filled

### Display-layer agent-name mapping

`frontend/lib/agent-display.ts`:

```ts
export const DISPLAY_AGENT: Record<string, string> = {
  supervisor:           'Controller',      // historical events only
  controller:           'Controller',
  workflow_controller:  'Controller',
  data_validator:       'Data Validator',
  dataset_approval:     'Dataset Approval',
  planner:              'Model Planner',
  executor:             'Training Executor',
  evaluation:           'Evaluation',
  report_writer:        'Audit Report',
  deployment_approval:  'Deployment Approval',
  deployer:             'Deployer',
  system:               'System',
}
export function displayAgentName(raw: string): string {
  return DISPLAY_AGENT[raw] ?? raw
}
```

Used by every component that renders an `agent` field. Historical events containing `"supervisor"` keep working; new events use `"controller"`.

### Backend rename (slice 1, backend half)

[api/services/pipeline.py:208](api/services/pipeline.py#L208) — change:

```python
# was
"agent": "supervisor",  # UI label, preserved for FE
# becomes
"agent": "controller",
```

Same change for `_emit_error` ([line 234](api/services/pipeline.py#L234)) and `complete_event` ([line 261](api/services/pipeline.py#L261)). The `data_validator`/`planner`/etc. names are real graph nodes and stay unchanged.

### Top nav update

[frontend/components/TopNav.tsx:7-11](frontend/components/TopNav.tsx#L7-L11) — `TABS` becomes:

```ts
const TABS = [
  { label: 'Pipeline',      href: '/pipeline' },
  { label: 'Experiments',   href: '/experiments' },
  { label: 'Observability', href: '/observability' },
  { label: 'Monitoring',    href: '/monitoring' },
]
```

`/observability` ships in slice 1 as a placeholder ("Coming in next iteration") so the tab works; real content lands in slice 5.

---

## Section 2 — Pipeline Stepper & Run Header (slice 1 continued)

### `<RunHeader>` — sticky bar at top of `/pipeline`

```
┌───────────────────────────────────────────────────────────────────┐
│ Run 37c1·7107   Forecasting    Waiting for human    2m 18s   ●    │
│ LLM:            data_validator · planner · report_writer          │
│ Deterministic:  controller · executor · evaluation · deployer     │
└───────────────────────────────────────────────────────────────────┘
```

- Run ID: monospace, first 8 chars, full ID copyable on click.
- Stage label: derived from stepper state (see below).
- Status pill (right): success / accent (running) / warning (waiting) / danger / muted.
- Elapsed: ticks every second client-side from `run_started` timestamp.
- LLM nodes list comes from the `run_info` event (already only emits `data_validator`, `planner`, `report_writer`). Deterministic list is hardcoded.

### `<PipelineStepper>` — horizontal 8-stage

Eight steps, each with name, status icon, and `<NodeTypeBadge>`:

| Stage | Node type |
|---|---|
| Data Validation | agent |
| Dataset Approval | hitl |
| Model Planning | llm |
| Training | deterministic |
| Evaluation | deterministic |
| Audit Report | llm |
| Deploy Approval | hitl |
| Deploy | deterministic |

Status: `pending | running | completed | waiting_human | failed | skipped`. Colors derived from tokens. On 1024px width: full horizontal; on narrower: wrap into a 4×2 grid.

### Stage-derivation function

Pure, testable. Lives in `frontend/lib/stage-derive.ts`:

```ts
export type StageKey =
  | 'data_validation' | 'dataset_approval' | 'model_planning'
  | 'training' | 'evaluation' | 'audit_report'
  | 'deploy_approval' | 'deploy'

export type StageStatus =
  | 'pending' | 'running' | 'completed'
  | 'waiting_human' | 'failed' | 'skipped'

export function deriveStages(
  events: PipelineEvent[],
  runStatus: RunStatus,
): {
  stages: Record<StageKey, StageStatus>
  attempts: { data_validator: number }   // attempt counter shown in RunHeader next to active stage
  runOutcome: 'running' | 'complete' | 'failed' | 'candidate_rejected'
}
```

Mapping rules (each rule is independently testable):

- `routing { next: 'data_validator' }` → `data_validation = running`
- `tool_result` from `validate_against_schema` → `data_validation = completed`
- `hitl_request { type: 'data_validation' }` → `dataset_approval = waiting_human`
- approval received for data_validation → `dataset_approval = completed`
- `routing { next: 'planner' }` → `model_planning = running`
- `planner_context` event → `model_planning = completed`
- `routing { next: 'executor' }` → `training = running`
- `tool_result` from `train_model` / `tune_hyperparameters` → `training = completed`
- `routing { next: 'evaluation' }` → `evaluation = running` / next non-evaluation routing → `completed`
- `audit_report` event (new, slice 3) → `audit_report = completed`. Until then, slice 1 leaves it at `pending` when routing has passed it; slice 3 fills it in.
- `hitl_request { type: 'deployer' }` → `deploy_approval = waiting_human`
- approval received → `deploy_approval = completed`
- `run_complete` with no error and approval was `approve` → `deploy = completed`
- `run_complete` with deploy_approval rejection or earlier user-rejection → remaining stages = `skipped`
- `run_complete { error }` → current stage = `failed`; `runOutcome = 'failed'`
- **Eval-rejection path**: `evaluation = completed`, `audit_report = completed`, `evaluation_passed = false` in any `audit_report` event payload → `deploy_approval = skipped`, `deploy = skipped`, `runOutcome = 'candidate_rejected'`. RunHeader pill uses `info` (sky), NOT `danger` (red).
- **Retries**: if `routing { next: 'data_validator' }` arrives after `dataset_approval` was already `completed`, reset: `data_validation = running`, `dataset_approval = pending`. Increment `attempts.data_validator`. (The stepper visibly "rewinds.")

### Pipeline page layout

[frontend/app/pipeline/page.tsx](frontend/app/pipeline/page.tsx) changes from `flex h-... w-2/5 + flex-1` to a vertical stack:

```
RunHeader                       (sticky top)
PipelineStepper                 (full-width)
─────────────────────────────────
[grid grid-cols-5 gap-4]
  ResultsDashboard + HITL cards (col-span-3, scrollable)
  EventLog (tabs, slice 4)      (col-span-2, scrollable)
```

The `ResultsDashboard` inner tabs (Dataset / Planner / Model — plus new Audit in slice 3) auto-switch **only on data-arrival events**:

| Event | Auto-switches to |
|---|---|
| `dataset_preview` arrives in HITL payload OR `tool_result load_dataset` | Dataset |
| `planner_context` | Planner |
| `tool_result train_model` / `tune_hyperparameters` | Model |
| `audit_report` | Audit |

Stage changes that don't carry new tab content (e.g., Training → Evaluation) leave the selection alone. A manual click sets a `pinnedTab` flag in component state; once pinned, no auto-switch happens until the next run is started.

---

## Section 3 — Dataset Approval Gate (slice 2)

### Backend

**New endpoints in `api/routers/runs.py`** (or wherever run-scoped routes live):

```
GET /runs/{run_id}/dataset-preview?limit=50&offset=0
  → { columns: [{name, dtype, non_null_count, sample_value}],
      rows: Array<Record<string, JSONValue>>,
      total_rows: number }

GET /runs/{run_id}/dataset-download
  → text/csv stream of the processed dataset
```

Both read `processed_dataset_path` from `RunEntry.processed_dataset_path` (new field, stashed during the dataset-approval HITL emission — see below). This keeps the HTTP layer decoupled from LangGraph's checkpointer. Errors: 404 if run unknown, 409 if no processed dataset path is known yet (HITL hasn't fired).

**RunEntry extension** in [api/services/run_store.py](api/services/run_store.py): add `processed_dataset_path: str | None = None`. Set inside `pipeline.py._stream` when a `dataset_approval` HITL request is being emitted — copy the `dataset_preview.path` from the interrupt payload onto the entry.

**HITL payload enrichment** in [src/mlops_agents/graphs/approval_nodes.py](src/mlops_agents/graphs/approval_nodes.py) `dataset_approval_node`. Today the payload includes `dataset_preview`. Add a `tail` field (last 5 rows) for forecasting problems. The full preview object becomes:

```json
{
  "type": "data_validation",
  "attempt": 1,
  "dataset_preview": {
    "path":          "data/processed/...",
    "row_count":     60,
    "column_count":  4,
    "columns":       [{ "name": "...", "dtype": "..." }],
    "head":          [ { ... }, ... ],   // first 5 rows
    "tail":          [ { ... }, ... ]    // last 5 rows (forecasting only; empty array otherwise)
  },
  "validation_report": { ... }
}
```

Problem type comes from `state["problem_type"]`; tail is only computed when `problem_type == "forecasting"`.

### Frontend `<DatasetApprovalCard>` — replaces `DatasetReviewPanel`

```
┌─ Dataset approval required ─────────────── Attempt 1 of 3 ●○○ ─┐
│                                                                 │
│  energy_forecast.csv          [ Copy artifact path ]            │
│  60 rows · 4 columns · forecasting · validation ✓ passed        │
│                                                                 │
│  [ Head ] [ Tail ] [ Schema ] [ Validation report ]             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ week_date    kwh_consumed  avg_temp_c  is_holiday        │   │
│  │ 2024-01-01   412.3         5.1         0                 │   │
│  │ ...                                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Comment (optional, required if rejecting)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [ Approve dataset ]  [ Reject & retry ]                        │
│  [ View full dataset ↗ ]  [ Download CSV ↓ ]                    │
└─────────────────────────────────────────────────────────────────┘
```

Behavior:
- Local-state tabs. Default = Head for tabular, **Head** with Tail tab visible+enabled for forecasting.
- **Copy artifact path** button copies `dataset_preview.path` to clipboard, shows a Sonner toast `Path copied`.
- **View full dataset** opens a modal that paginates via `/dataset-preview?limit=50&offset=N`.
- **Download CSV** is a regular `<a href>` to `/dataset-download` (browser handles save).
- **Schema tab** = table of `name / dtype / non_null_count / sample_value`.
- **Validation report tab** = today's violations list inside a Card.
- **Reject** is disabled until comment ≥ 4 chars. Approve is always enabled.
- Attempt indicator shows `Attempt N of M` where `M = settings.max_attempts_per_agent` (currently 3); pulled from the HITL payload, not hardcoded.

---

## Section 4 — Deployment Approval Gate + Audit Report Panel + bug fix (slice 3)

### Backend bug fix (must ship with slice 3)

**Bug:** [src/mlops_agents/graphs/workflow_controller.py:77](src/mlops_agents/graphs/workflow_controller.py#L77) routes to `deployer` only if `deployment_decision == "pending"`. [src/mlops_agents/graphs/approval_nodes.py:54](src/mlops_agents/graphs/approval_nodes.py#L54) overwrites `deployment_decision` to `"approved"` on user approval. Result: an approved deployment falls through to `END` without the deployer ever running.

**Fix:** `deployment_approval_node` must NOT touch `deployment_decision`. It only writes `deployment_approved: True|False`. The deployer node is responsible for flipping `deployment_decision` to `"deployed"` after it runs. Controller stays unchanged.

```python
# approval_nodes.py — corrected
return Command(
    goto="workflow_controller",
    update={"deployment_approved": approved},
)
```

Regression test: `tests/graph/test_deployment_flow.py` runs the graph end-to-end with mocked MLflow, approves at Gate 2, asserts `deployer_node` was called and `deployment_status == "deployed"`.

### Backend: new `audit_report` SSE event

In [api/services/pipeline.py](api/services/pipeline.py) `_stream`, when an `updates` chunk contains `report_writer` with `evaluation_report_audit` set, emit:

```json
{
  "type": "audit_report",
  "agent": "report_writer",
  "timestamp_ms": ...,
  "data": {
    "audit":              { /* full EvaluationReport */ },
    "champion_model":     "seasonal_naive",   // pre-resolved server-side
    "evaluation_passed":  true,               // drives "Candidate rejected" UX path
    "candidate_metrics":  { /* from state */ },
    "champion_metrics":   { /* from state */ },
    "thresholds_applied": { /* from state */ }
  }
}
```

Because `report_writer` always runs immediately before `deployment_approval` in the controller, this event is guaranteed to arrive in the SSE stream before the deployer's `hitl_request`. No buffering needed.

### Backend: enrich deployer HITL payload

In `deployment_approval_node`, the `interrupt({...})` call must include enough to render the approval card without an extra fetch:

```python
interrupt({
    "type": "deployer",
    "evaluation_report":       state.get("evaluation_report", {}),
    "evaluation_report_audit": state.get("evaluation_report_audit", {}),
    "candidate_metrics":       state.get("candidate_metrics", {}),
    "champion_metrics":        state.get("champion_metrics", {}),
    "thresholds_applied":      state.get("thresholds_applied", {}),
    "training_plan":           state.get("training_plan", {}),
    "candidate_run_id":        state.get("training_run_id", ""),
    # NEW: explicit action description
    "deployment_action": {
        "verb":   "register_and_promote",
        "model":  _resolve_champion_model_name(state),  # see champion derivation
        "alias":  "champion",
        "summary": "This approval will register the candidate run as a new model version and assign it the champion alias.",
    },
})
```

`_resolve_champion_model_name(state)` runs the same fallback chain as the frontend (see below).

### Champion model resolution (backend only)

`src/mlops_agents/evaluation/champion.py`:

```python
def resolve_champion_model_name(state: dict) -> str:
    """Single source of truth for the human-readable champion name.

    Fallback chain:
      1. state["evaluation_report_audit"]["champion_model"]
      2. state["champion_candidate"]["model_key"]
      3. state["training_plan"]["selected_model"]
      4. state["training_run_id"][:8]   # monospace short id fallback
    """
```

Used in **two** places:
- `audit_report` SSE event: added as `data.champion_model: str`
- `deployer` HITL payload: `deployment_action.model: str`

Frontend reads the pre-resolved string from whichever event arrived. **No frontend fallback logic** — single source of truth.

### Frontend `<AuditReportPanel>` — new "Audit" tab in `ResultsDashboard`

Auto-shows when an `audit_report` event has arrived. Tab order becomes: Dataset · Planner · Model · **Audit**.

```
┌─ Audit Report ──────────────────────────────────────── retry-ok ─┐
│                                                                  │
│  Champion model      seasonal_naive                              │
│  Primary metric      RMSE = 7.5573  (lower is better)            │
│  Deterministic eval  ✓ passed                                    │
│                                                                  │
│  ▸ Summary                                                       │
│  ▸ Why this model won                                            │
│  ▸ Planner alignment                                             │
│  ▸ Deviations from planner expectations                          │
│  ▸ Evidence consistency warnings                                 │
│  ▼ Risks & warnings                                              │
│      ⚠ season_length=7 surprising for weekly data                │
│      ⚠ single split on short history may be unstable             │
│      ⚠ seasonal_naive ignores exogenous variables                │
│  ▸ Human review notes                                            │
│                                                                  │
│  [ View full audit JSON ▾ ]                                      │
└──────────────────────────────────────────────────────────────────┘
```

Implemented with native `<details>` for keyboard/screen-reader support. Risks default open; everything else closed.

### Frontend `<DeploymentApprovalCard>` — replaces deployer JSON dump in `HITLGate`

```
┌─ Deployment approval required ───────────────────────────────────┐
│  Approve deployment: seasonal_naive                              │
│                                                                  │
│  Candidate run     c5e7fe98…       [ ↗ MLflow ]                  │
│  Primary metric    RMSE = 7.5573 (lower is better)               │
│  Promotion         eligible — passed deterministic thresholds    │
│                                                                  │
│  ── Deployment action ──                                         │
│  Register model version + assign champion alias.                 │
│  This approval will promote `seasonal_naive` as champion.        │
│                                                                  │
│  ── Top risks from audit ──                                      │
│  ⚠ season_length=7 surprising for weekly data                    │
│  ⚠ single split on short history may be unstable                 │
│  ⚠ seasonal_naive ignores exogenous variables                    │
│  See full audit in the Audit tab above.                          │
│                                                                  │
│  [ Approve deployment ]  [ Reject deployment ]                   │
│  [ Raw payload ▾ ]                                               │
└──────────────────────────────────────────────────────────────────┘
```

- "Deployment action" block reads `interruptValue.deployment_action` verbatim from the enriched HITL payload. If the field is missing (older runs), fall back to: `"Register model version + assign champion alias."` + monospace candidate run ID.
- Top 3 risks come from `audit.risks_and_warnings.slice(0, 3)`.
- Raw JSON moved to a `<details>` block at the bottom. Not the default view.
- MLflow run link uses `NEXT_PUBLIC_API_URL`-derived base, points to `http://localhost:5000/#/experiments/.../runs/<id>` — same pattern as existing Experiments charts.

---

## Section 5 — Event Log Redesign + Deduplication (slice 4)

`<EventLog>` becomes a Card with three tabs:

### Timeline tab (default)

Allow-listed event types, plus synthetic milestones derived from `tool_result`:

| Source event | Timeline line |
|---|---|
| `run_info` | `Pipeline started · LLM nodes: data_validator, planner, report_writer` |
| `routing` (only when target changes) | `Workflow moved to <Stage Name>` |
| `tool_result` for `load_dataset` | `Dataset loaded · N rows × M cols` |
| `tool_result` for `validate_against_schema` | `Validation passed` / `Validation failed (K violations)` |
| `hitl_request` | `<Gate> approval requested` |
| approval received | `<Gate> <approved\|rejected>` |
| `planner_context` | `Planner selected K candidates` |
| `tool_result` for `train_model` | `Training completed — <model>` |
| `audit_report` | `Audit report generated` |
| `run_complete` | `Run complete` / `Run failed: <reason>` |

Everything else (`tool_call`, raw `agent_reasoning`, repeated tool_result) hidden from Timeline.

### Tool Details tab

Two subsections — tool-loop agents vs single-shot LLM nodes are different shapes and should not be conflated.

**Tool calls (agentic loops)** — derived from `tool_call` + `tool_result` pairs, aggregated by `(agent, tool_name)`:

```
Data Validator
  load_dataset             2 calls    416 ms
  parse_datetime_column    2 calls    992 ms
  validate_against_schema  1 call     201 ms
```

**LLM nodes (single-shot)** — derived from `routing { next: <node> }` intervals (start = routing-in, end = next routing-out):

```
Planner          1 activation    23.4 s    retry-ok
Report Writer    1 activation     8.2 s    ok
```

Two aggregators in `frontend/lib/events-aggregate.ts`: `aggregateToolUsage(events)` and `aggregateLlmNodeActivity(events)`. Both pure-function unit-tested. Observability page (slice 5) re-uses both.

### Raw Logs tab

Today's `EventLog` content, unchanged. Mono font, every event line. Plus a **Download raw trace JSON** button at the top:

```
[ ↓ Download raw trace JSON ]
```

Builds a Blob from `useRunStore.events`, triggers `<a download="run-<id>.json">`. No backend involvement.

### Deduplication helper

`frontend/lib/events-aggregate.ts`:

```ts
// Collapses consecutive same-(type, agent, tool) events into one row with count.
// Used by Timeline and Observability cards.
export function aggregateConsecutive(events: PipelineEvent[]): AggregatedRow[]

// Aggregates tool_call/tool_result pairs by (agent, tool_name) across whole run.
export function aggregateToolUsage(events: PipelineEvent[]): ToolUsageRow[]
```

Both pure, vitest unit-tested. The zustand store is unchanged — aggregation is render-time only.

---

## Section 6 — Observability page + Experiments polish (slice 5)

### `/observability` page

Three Cards. Same layout idiom as `/monitoring`.

```
┌─ Pipeline health (last 20 runs since server start) ───────────┐
│  18 successful · 2 failed · avg 2m 41s · 1 awaiting human     │
│  resets on container restart                                  │
└───────────────────────────────────────────────────────────────┘
┌─ LLM activity (current run) ──────────────────────────────────┐
│  data_validator   3 calls   1.4 s     ok                      │
│  planner          1 call    23.4 s    retry-ok                │
│  report_writer    1 call    8.2 s     ok                      │
│  (token counts shown if available; otherwise —)               │
└───────────────────────────────────────────────────────────────┘
┌─ Tool usage (current run) ────────────────────────────────────┐
│  load_dataset             2 calls    416 ms                   │
│  parse_datetime_column    2 calls    992 ms                   │
│  ...                                                          │
└───────────────────────────────────────────────────────────────┘
```

**Card 1 — Pipeline health** uses a new endpoint `GET /runs?limit=20` returning a list of `{ run_id, problem_type, status, started_at, duration_ms, error }`. The `run_store` is in-memory only — the card subtitle states this honestly so the jury isn't misled into expecting durable history. Persistent storage is explicitly out-of-scope for this iteration.

**Card 2 — LLM activity** (renamed from "LLM usage" — we are NOT claiming token/cost accounting we don't have). Columns:
- calls — count of `routing { next: <llm_node> }`
- duration — sum of time from routing-in to next routing-out, if derivable; else "—"
- tokens — only if an `agent_reasoning` event carries usage metadata; else "—"
- status — derived from `planner_status` / `evaluation_report_audit_status` fields where applicable; else "ok"

Never invent token counts. Never invent cost.

**Card 3 — Tool usage** uses the existing `aggregateToolUsage()` helper from slice 4.

### Experiments page polish

Touch only what aligns it visually:
- Wrap `RunSidebar` and `ChartPanel` in `<Card>` primitives from slice 1.
- In `RunSidebar`, add a `Champion` `<Badge variant="success">` next to runs flagged as champion, and a problem-type subtitle under the run name.
- Apply zinc/indigo tokens to active-run highlight.
- No changes to charts, MLflow integration, or filters.

---

## Testing strategy

| Slice | Test target | Approach |
|---|---|---|
| 1 | `deriveStages()` | vitest unit — feed synthetic event arrays, assert stage map |
| 1 | `<RunHeader>` / `<PipelineStepper>` | RTL render tests; one per stage status |
| 1 | Backend rename | grep CI check: `agent.*=.*['"]supervisor['"]` returns 0 in `api/` |
| 2 | Dataset preview endpoint | pytest with a stub `state` containing a tiny CSV; assert pagination |
| 2 | `<DatasetApprovalCard>` | RTL: head/tail tab visibility per problem_type; reject button disabled until comment |
| 3 | **Bug-fix regression** | pytest end-to-end graph with mocked MLflow + approval; assert `deployment_status == "deployed"` |
| 3 | `audit_report` event emission | unit test on `_stream` with a fake `astream` yielding a `report_writer` chunk |
| 3 | `<AuditReportPanel>` | RTL: collapsible sections, default-open risks |
| 3 | `<DeploymentApprovalCard>` | RTL: renders `deployment_action.model` and `deployment_action.summary` verbatim from HITL payload |
| 3 | `resolve_champion_model_name` | pytest: 4-step fallback chain unit-tested in Python (single source of truth) |
| 3 | Eval-rejection UX path | E2E: graph with mocked MLflow returning failing metric; assert stepper marks `deploy_*` as skipped, runOutcome = 'candidate_rejected', RunHeader uses sky pill |
| 4 | `aggregateConsecutive` / `aggregateToolUsage` | vitest — pure-function unit tests |
| 4 | `<EventLog>` tabs | RTL: tab switching, Download JSON triggers Blob URL |
| 5 | `GET /runs?limit=` | pytest |
| 5 | `/observability` rendering | RTL: "—" fallback when token data absent |

Existing test suite (`uv run pytest`, `cd frontend && npm test`) must stay green at each slice's end.

## Open boundaries / what could change

- If the slice-1 `<Card>` / `<Badge>` primitives turn out to need a variant we didn't predict (e.g., elevated vs flat), add it then. Don't pre-design.
- `Pipeline health` card may need a tiny in-memory aggregation in `run_store` if the existing structure doesn't expose duration / status counts cheaply — decide during slice 5.
- Slice-1 demos will show the `audit_report` stage as perpetually `pending` until slice 3 emits the new event. Acceptable transient state.

## Acceptance criteria (overall)

The refactor is done when:

1. The user can identify the current pipeline stage at a glance from the stepper, without reading logs.
2. The user can approve/reject the dataset after seeing a real preview (head, tail for forecasting, schema, validation).
3. The user can approve/reject deployment after seeing: champion model name, primary metric and direction, deterministic decision, top risks, **and the explicit action being authorized** (register + champion alias).
4. Approving a deployment actually runs the deployer node (regression test passes; manual smoke confirms `deployment_status == "deployed"`).
5. The UI no longer displays `supervisor` as if it were an active agent. New events use `controller`.
6. Repeated planner/tool events are aggregated in Timeline; full fidelity available in Raw Logs.
7. Raw trace JSON is downloadable from Raw Logs.
8. The UI visually distinguishes agent / LLM / deterministic / human-gate nodes via `<NodeTypeBadge>`.
9. `/observability` exists with three cards; never claims token/cost data it doesn't have.
10. `/monitoring` keeps working (Evidently drift untouched).
11. When the deterministic evaluator rejects the candidate, the UI says `Candidate rejected` (sky/info), auto-focuses the Audit tab, and marks deploy stages `skipped` — never `Failed` / red.
12. Dataset rejection is gated on a comment ≥ 4 chars (no silent rejections).
13. `processed_dataset_path` is exposed via `RunEntry`, not the LangGraph checkpointer.
14. `champion_model` is resolved server-side once; no duplicated fallback in the frontend.
