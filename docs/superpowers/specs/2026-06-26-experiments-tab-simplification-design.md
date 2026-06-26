# Experiments Tab Simplification — Design

**Date:** 2026-06-26
**Status:** Approved (design), pending implementation plan
**Area:** `frontend/` (Next.js dashboard) — Experiments tab

## Problem

The Experiments tab renders four charts across three sections for a single
selected run, and they all visualize the same one or two numbers:

- **Trainer — Loss & Accuracy** (`TrainerLineChart`): plots `metric_series`
  over training steps. Forecasting models (ARIMA, ETS, naive, tree ensembles)
  log each metric once, so `get_metric_history` returns a single point per
  metric. The chart is always floating dots, never a real training curve.
- **Evaluator Metrics** (`EvaluatorRadarChart` + `EvaluatorBarChart`, side by
  side): both render the same `run.metrics` dict. A radar of one metric is a
  degenerate single spoke; the bar is a single bar. Two charts, one number.
- **Deployment Comparison** (`DeploymentBarChart`): the same `run.metrics`
  again, just sorted. Labeled "Comparison" but compares nothing.

Net result: 4 charts, 3 sections, all drawing the same 1–2 values of one run.
The `EvaluatorBarChart` also hard-codes an `[0, 1]` X-axis domain, which is
wrong for error metrics (e.g. `rmse = 18.5` overflows the axis).

## Decision

Keep the **one-run-at-a-time** model (no cross-run comparison in this pass).
Replace the four charts with a clean, chart-free **run detail readout** showing
the selected run's metrics, configuration, and metadata — all from data the API
already returns in `RunOut`. No backend changes.

(Cross-model comparison was considered and explicitly deferred. The data
supports it — `fetchExperimentRuns` returns every run — but the chosen scope is
simplification, not a new comparison feature.)

## Layout

```
┌─ Metrics ───────────────────────────────────────────────┐
│  extra_trees_forecaster   [✓ complete]   2026-06-26 06:27 │
│  run_id: a1b2c3d4…                          [Export CSV]  │
│  ─────────────────────────────────────────────────────── │
│  Metrics                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐             │
│  │ 18.500 │ │  12.30 │ │ 0.0412 │ │  0.920 │             │
│  │  mae   │ │  mape  │ │   r2   │ │  rmse  │  (alpha sort)│
│  └────────┘ └────────┘ └────────┘ └────────┘             │
│  ─────────────────────────────────────────────────────── │
│  Configuration                                           │
│  lags              [1, 2, 3, 12]                          │
│  model_type        extra_trees_forecaster                │
│  n_estimators      200                                    │
│  season_length     12                                     │
└──────────────────────────────────────────────────────────┘
```

Three parts, all sourced from `RunOut`:

1. **Header** — `run_name` (the API already falls back to `run_id[:8]`), a
   status badge (see micro-decision 2, reusing `DatasetPanel`'s pill classes),
   `start_time`, and the truncated `run_id`.
2. **Metrics** — a responsive grid of metric cards (label + value), one per
   entry in `run.metrics`, sorted alphabetically. Reuses the `StatCard` visual
   language from `TrainingCompletePanel`.
3. **Configuration** — `run.params` as a two-column key/value table, keys
   sorted alphabetically.

## Components & files

| File | Change |
|---|---|
| `frontend/components/experiments/ChartPanel.tsx` | Rewrite contents; rename file/export to `RunDetailPanel.tsx` (header + metrics grid + config table). Drop the inner `<Card>` wrapper; root is `h-full overflow-y-auto` so it scrolls inside the page's existing bordered container. |
| `frontend/app/experiments/page.tsx` | Update import/usage `ChartPanel` → `RunDetailPanel`. Sidebar + panel split unchanged; the panel slot div keeps its border/padding (single source of chrome now that the inner Card is gone). |
| `frontend/components/experiments/charts/` | Delete the entire folder (`TrainerLineChart`, `EvaluatorRadarChart`, `EvaluatorBarChart`, `DeploymentBarChart`). |
| `frontend/__tests__/components/experiments/ChartPanel.test.tsx` | Rename to `RunDetailPanel.test.tsx`; update import; update the empty-state assertion to the new copy ("Select a run to view its metrics"); add two assertions — a metric value renders and a param key renders — so coverage is not reduced vs today. |
| `frontend/package.json` + `package-lock.json` | Remove `recharts` (used only by the deleted charts — verified by grep); run `npm install` to sync the lockfile. |

`StatCard` and the metric-card grid markup are lifted from `TrainingCompletePanel`
in `ResultsDashboard.tsx` and kept as small **local** components inside
`RunDetailPanel.tsx` (trivial, not worth a shared export; the existing one stays
local too). The two views — Experiments panel and the live-run "Model" tab —
should look identical.

### Chrome & scrolling

Today the panel slot in `page.tsx` is already a bordered, padded, rounded box,
and `ChartPanel` wraps its content in a second `<Card>` (border-in-border,
doubled padding). The rewrite removes the inner `<Card>`; the new component
renders the header and sections directly with `h-full overflow-y-auto` on its
root so a long metrics+config readout scrolls within the fixed-height container.

## Data flow

No backend or type changes. `RunOut` already exposes everything needed:

- `run_id`, `run_name`, `status`, `start_time` → header
- `metrics: Record<string, number>` → metric cards
- `params: Record<string, string>` → configuration table

`metric_series` is no longer read by the frontend. The backend still computes it
in `mlflow_client.py` (one `get_metric_history` call per metric per run) — this
is now dead work, but removing it is deferred (see Out of scope).

## Micro-decisions

1. **Panel heading** — the old `<Card title="Metrics">` is removed (see Chrome &
   scrolling). With the Card gone there is no misleading "Metrics" title to
   correct; the **run name** (`run_name`) serves as the panel's heading at the
   top of the header row.
2. **Status badge** — `status` is MLflow's raw uppercase value
   (`run.info.status`), not the app's `RunStatus` union. Map via a small table,
   reusing the validation-pill classes from `DatasetPanel`:
   - `FINISHED` → "complete", emerald (`bg-emerald-50 text-emerald-700`)
   - `FAILED` / `KILLED` → "failed" / "killed", red (`bg-red-50 text-red-600`)
   - `RUNNING` / `SCHEDULED` → "running" / "scheduled", amber
   - anything else → raw status lowercased, zinc/gray
3. **Date display** — `start_time` is an ISO UTC string. Render in the viewer's
   **local timezone** via `toLocaleString` with
   `{ year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false }`
   → `YYYY-MM-DD HH:mm` (minute precision). Fall back to the raw string if
   unparseable.
4. **Number formatting** — generic, not metric-aware. Values with absolute
   value ≥ 1 render to 3 decimals (`18.500`); values < 1 render to 4 decimals
   (`0.0412`). No percentage conversion (not all metrics are accuracies).
   Non-finite values render as `—`.
5. **Export CSV** — kept, moved to the header. Exports all metrics **and**
   params (was metrics-only on the old deployment chart). Format is RFC 4180:
   header row `type,key,value`; rows tagged `metric` or `param`; every field
   quoted with embedded `"` escaped as `""` (param values such as
   `lags = [1, 2, 3, 12]` contain commas). Filename
   `run-<run_id-prefix>-metrics.csv`. Button is **hidden when there are no
   metrics and no params** (nothing to export).
6. **Metric grid columns** — responsive `grid-cols-2 sm:grid-cols-3
   lg:grid-cols-4`; card markup identical to `TrainingCompletePanel`'s. Config
   table stays full-width below.
7. **Metric & param ordering** — alphabetical, stable. No "primary metric"
   highlighting; `RunOut` does not expose which metric was the optimization
   target (confirmed by the existing `TODO`s in `RunSidebar`).
8. **Empty states** — per section: "No metrics logged" / "No parameters
   logged". The no-selection placeholder becomes "Select a run to view its
   metrics."
9. **Sidebar `acc:` subtitle** — out of scope, unchanged. (It shows `acc: —`
   for forecasting runs, but that is a separate sidebar concern not raised
   here.)

## Out of scope

- Cross-model / cross-run comparison.
- Backend changes (`RunOut`, `mlflow_client.py`, exposing champion or
  problem_type).
- **Removing the now-dead `metric_series` computation** from
  `mlflow_client.py` / `RunOut` / the TS type. Deferred follow-up: it touches the
  API contract and `api/tests/test_experiments.py` / `test_models.py`, which is
  backend scope creep on a frontend-simplification task. Cost is small (run
  listing is not hot-path).
- The sidebar's `acc:` subtitle.
- A forecast-vs-actuals plot (already exists in the live-run "Model" tab via
  `TrainingCompletePanel`'s `forecast_chart_png`; not duplicated here).

## Testing

- Frontend unit test (`RunDetailPanel.test.tsx`, vitest + Testing Library):
  empty state when `run` is null; a metric value renders; a param key renders.
- Frontend type check / lint passes; `recharts` removal leaves no dangling
  imports (verified — only the deleted charts used it).
- Existing API tests (`api/tests/test_experiments.py`, `test_models.py`) are
  unaffected — no backend change, `metric_series` still produced.
- Manual: select a forecasting run → metric cards + config table, no
  floating-dot chart, no broken `[0,1]` axis, single border (no nested Card),
  scrolls when long; select a run with no params → empty-state copy; a run with
  neither metrics nor params → both empty states, Export CSV hidden; Export CSV
  on a run with `lags`-style comma params → valid quoted CSV.
