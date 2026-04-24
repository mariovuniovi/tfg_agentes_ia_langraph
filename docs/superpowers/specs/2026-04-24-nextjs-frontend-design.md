# Next.js Frontend — Design Spec

**Date:** 2026-04-24
**Status:** Approved
**Branch:** `feature/nextjs-frontend` (to be forked from `feature/fastapi-backend`)
**Sub-project:** 2 of 2 — Next.js Frontend (FastAPI backend is sub-project 1, complete)

---

## Problem

The FastAPI backend (sub-project 1) exposes the MLOps pipeline, MLflow data, and Evidently drift detection via HTTP + WebSocket. There is currently no web frontend to consume it — the Streamlit dashboard remains in-process and cannot be decoupled. A Next.js frontend will replace the Streamlit UI for demo purposes while the Streamlit dashboard remains runnable.

## Goal

Build a typed Next.js frontend that consumes the FastAPI API at `http://localhost:8000`. The UI is a desktop-only thesis demo — no authentication, no persistence across server restarts.

---

## Decisions

| Question | Decision |
|---|---|
| Framework | Next.js 15, App Router, TypeScript strict mode |
| Navigation | 3 top tabs: Pipeline · Experiments · Monitoring |
| Pipeline tab layout | Two-column: controls + HITL gate left, event stream right |
| Experiments tab layout | Sidebar run list + right chart panel |
| Monitoring tab layout | Two sub-tabs: Latest Report · Ad-hoc Analysis |
| Live run state | Zustand |
| Server / cached state | TanStack Query |
| Charts | Recharts |
| Styling | Tailwind CSS, Enterprise Blue + Amber palette, Fira Code / Fira Sans fonts |
| Testing | Vitest + React Testing Library |
| Mobile | Out of scope (desktop demo) |
| Auth | Out of scope |

---

## Directory Structure

```
frontend/
├── app/
│   ├── layout.tsx               # root layout — font vars, TopNav tabs
│   ├── page.tsx                 # redirect → /pipeline
│   ├── pipeline/page.tsx
│   ├── experiments/page.tsx
│   └── monitoring/page.tsx
├── components/
│   ├── pipeline/
│   │   ├── TriggerPanel.tsx     # dataset path input + Run button
│   │   ├── EventLog.tsx         # scrollable live event list
│   │   ├── RunStatusBadge.tsx   # status pill
│   │   └── HITLGate.tsx         # approval panel (renders when hitlPending)
│   ├── experiments/
│   │   ├── RunSidebar.tsx       # experiment dropdown + run list
│   │   ├── ChartPanel.tsx       # renders charts for selected run
│   │   └── charts/
│   │       ├── TrainerLineChart.tsx
│   │       ├── EvaluatorRadarChart.tsx
│   │       ├── EvaluatorBarChart.tsx      # a11y fallback alongside Radar
│   │       └── DeploymentBarChart.tsx
│   └── monitoring/
│       ├── LatestReport.tsx     # drift badge + area chart + DriftTable
│       ├── AdHocForm.tsx        # file upload + index selectors + Run Drift
│       └── DriftTable.tsx       # column · score · drift detected
├── stores/
│   └── run-store.ts             # Zustand: live run state
├── hooks/
│   ├── use-run-stream.ts        # WebSocket lifecycle → writes to run-store
│   └── use-approve.ts           # POST /runs/{id}/approve mutation
├── lib/
│   ├── api.ts                   # typed fetch wrappers for all REST endpoints
│   └── query-client.ts          # TanStack QueryClient singleton
└── types/
    └── api.ts                   # TypeScript mirrors of FastAPI Pydantic models
```

**Dependencies:** `next`, `react`, `zustand`, `@tanstack/react-query`, `recharts`, `tailwindcss`.

---

## State Boundary

> **Zustand** owns **live run state** — data that arrives via WebSocket during an active pipeline run: the event log, current status, HITL interrupt payload, and whether approval is pending. This state is push-based and changes in real-time.
>
> **TanStack Query** owns **server state fetched on demand** — MLflow experiments, run history, metric charts, drift reports. It is pull-based, cached, and benefits from automatic refetch and stale-time management.
>
> The rule: **WebSocket / real-time → Zustand. REST / cached → TanStack Query.**

### Zustand store (`stores/run-store.ts`)

```ts
interface RunState {
  runId: string | null
  status: "idle" | "running" | "awaiting_approval" | "complete" | "failed"
  events: PipelineEvent[]
  interruptValue: Record<string, unknown> | null
  hitlPending: boolean
}
```

The `use-run-stream` hook connects to `WS /ws/{run_id}`, parses each JSON message, and writes into this store. On reconnect it calls `GET /runs/{run_id}` to rehydrate state before re-attaching the WebSocket.

### TanStack Query responsibilities

| Query key | Endpoint | When fetched |
|---|---|---|
| `["experiments"]` | `GET /experiments` | Experiments tab mount |
| `["runs", expId]` | `GET /experiments/{id}/runs` | Experiment selected |
| `["monitoring", "latest"]` | `GET /monitoring/latest` | Monitoring tab mount |
| `["monitoring", "adhoc"]` | `POST /monitoring/drift` | Ad-hoc form submit (mutation) |
| `["health"]` | `GET /health` | App mount, 30s refetch interval |

---

## Pages

### Pipeline tab (`/pipeline`)

Two-column layout. Left column (≈38% width): `TriggerPanel`, `RunStatusBadge`, `HITLGate`. Right column (≈62%): `EventLog`.

**TriggerPanel:** text input for `dataset_paths` (comma-separated), Run button calls `POST /runs`, stores returned `run_id` in Zustand and connects WebSocket via `use-run-stream`.

**EventLog:** scrollable list auto-scrolling to latest event. Events are color-coded by type:
- `routing` → blue
- `tool_call` / `tool_result` → slate
- `agent_reasoning` → indigo
- `hitl_request` → amber
- `run_complete` → green

**HITLGate:** renders only when `hitlPending: true`. Shows the `interruptValue` payload (model name, metrics) and Approve / Reject buttons. Approve calls `POST /runs/{id}/approve` with `{ decision: "approve" }`, Reject with `{ decision: "reject" }`. On response: `hitlPending: false`.

### Experiments tab (`/experiments`)

Left sidebar (≈30%): experiment dropdown (populated from `GET /experiments`), scrollable run list with accuracy preview. Clicking a run sets `selectedRunId`.

Right panel (≈70%): `ChartPanel` renders three charts for the selected run:
1. **Trainer — Line chart** (`metric_series`): loss and accuracy over training steps. Multi-series lines cycle `solid → dashed → dotted` (a11y: not color alone).
2. **Evaluator — Radar chart + Bar chart fallback** (`metrics`): precision / recall / F1 / AUC. Bar chart always rendered alongside Radar (a11y grade B).
3. **Deployment — Horizontal Bar chart** (`metrics`): model version comparison, sorted descending. Value labels on every bar. CSV export button in `#D97706`.

### Monitoring tab (`/monitoring`)

Two sub-tabs rendered as inner tab bar:

**Latest Report sub-tab:** calls `GET /monitoring/latest`.
- Drift badge: green "No drift" / red "Drift detected"
- Drift share displayed as a prominent percentage number
- `DriftTable`: column name · score · drift detected (✓/✗)
- `404` response → empty state: "No pipeline run completed yet"

Note: no trend chart here — the backend stores only the single latest report, not a historical series.

**Ad-hoc Analysis sub-tab:** drag-and-drop file upload (multiple CSVs). After upload, reference index and current index dropdowns populate. "Run Drift" button calls `POST /monitoring/drift`. Results render as a `DriftReport` below the form (same badge + table as Latest Report).

---

## HITL Flow (Frontend Perspective)

1. WebSocket delivers `{ type: "hitl_request", data: interruptValue }` → store sets `hitlPending: true`, `status: "awaiting_approval"`
2. `HITLGate` component renders in the left column
3. User clicks Approve or Reject → `use-approve` calls `POST /runs/{run_id}/approve`
4. Store sets `hitlPending: false`; WebSocket resumes streaming post-approval events until `run_complete`

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| REST error (4xx/5xx) | TanStack surfaces `isError` → inline error banner per page |
| WebSocket disconnect | `use-run-stream` retries with exponential backoff (3 attempts), then calls `GET /runs/{id}` to rehydrate and reconnects |
| `POST /approve` while not `awaiting_approval` | Backend returns `400` → toast notification |
| `GET /monitoring/latest` returns `404` | Empty state prompt, not an error banner |
| Health check fails | Warning badge in `TopNav` |

---

## Testing

- **Vitest + React Testing Library** for component tests. Mock `api.ts` functions at the module level.
- Key cases: `HITLGate` renders only when `hitlPending: true`; `EventLog` renders and auto-scrolls; `RunSidebar` selects a run and updates chart panel.
- **Zustand store tested in isolation** — no React needed, call actions and assert state shape.
- No E2E tests (FastAPI integration tests cover the backend contract).

---

## Out of Scope

- Authentication / API keys
- Mobile / responsive layout
- Persistent run history across backend restarts (in-memory backend)
- Dark mode
- Next.js frontend deployment / Docker changes
