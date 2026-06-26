# Experiments Tab Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four redundant single-run charts in the Experiments tab with a chart-free run-detail readout (metric cards + config table + run metadata).

**Architecture:** A single React component (`RunDetailPanel`) renders the selected `RunOut` directly — header (name/status/time/id), a responsive grid of metric cards, and a config table — reusing the metric-card visual language from `TrainingCompletePanel`. Pure formatting/CSV helpers move to `lib/format.ts` for unit testing. The four chart components and `recharts` are deleted. No backend changes.

**Tech Stack:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4, vitest + Testing Library (jsdom), npm.

## Global Constraints

- **No backend changes.** `RunOut`, `mlflow_client.py`, and the API contract stay as-is; `metric_series` is still produced server-side but no longer read by the frontend.
- **Reuse existing patterns.** Metric card markup must match `TrainingCompletePanel` in `frontend/components/pipeline/ResultsDashboard.tsx`; status badge reuses the validation-pill classes from `DatasetPanel` in the same file.
- **Number formatting:** abs(value) ≥ 1 → 3 decimals; < 1 → 4 decimals; non-finite → `—`.
- **Date display:** local timezone, format `YYYY-MM-DD HH:mm` (minute precision); unparseable input falls back to the raw string.
- **Metric & param ordering:** alphabetical by key, everywhere (display and CSV).
- **Empty-state copy (verbatim):** no-selection → `Select a run to view its metrics`; no metrics → `No metrics logged`; no params → `No parameters logged`.
- **CSV:** RFC 4180 — header row `"type","key","value"`; data rows tagged `metric` or `param`; **every field double-quoted** (header and data) with embedded `"` escaped as `""`. Filename `run-<run_id first 8 chars>-metrics.csv`. Export button hidden when there are no metrics AND no params.
- **Commit policy:** committing is allowed; never `git merge` or `git push`. Never add a Claude/Anthropic co-author trailer.
- **Run commands from the `frontend/` directory.**

---

### Task 1: Formatting & CSV helpers in `lib/format.ts`

Pure, dependency-free functions so the component stays thin and the logic is unit-testable.

**Files:**
- Modify: `frontend/lib/format.ts`
- Test: `frontend/__tests__/lib/format.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `formatMetricValue(v: number): string`
  - `formatRunTime(iso: string): string`
  - `buildRunCsv(metrics: Record<string, number>, params: Record<string, string>): string`

- [ ] **Step 1: Write the failing tests**

Append to `frontend/__tests__/lib/format.test.ts`:

```ts
import { formatMetricValue, formatRunTime, buildRunCsv } from '@/lib/format'

describe('formatMetricValue', () => {
  it('uses 3 decimals for magnitude >= 1', () => {
    expect(formatMetricValue(18.5)).toBe('18.500')
    expect(formatMetricValue(-2)).toBe('-2.000')
  })

  it('uses 4 decimals for magnitude < 1', () => {
    expect(formatMetricValue(0.0412)).toBe('0.0412')
    expect(formatMetricValue(0)).toBe('0.0000')
  })

  it('renders an em dash for non-finite values', () => {
    expect(formatMetricValue(NaN)).toBe('—')
    expect(formatMetricValue(Infinity)).toBe('—')
  })
})

describe('formatRunTime', () => {
  it('formats a valid ISO string as YYYY-MM-DD HH:mm', () => {
    expect(formatRunTime('2026-06-26T06:27:05+00:00')).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/)
  })

  it('falls back to the raw string when unparseable', () => {
    expect(formatRunTime('not-a-date')).toBe('not-a-date')
  })
})

describe('buildRunCsv', () => {
  it('quotes every field and tags rows by type', () => {
    const csv = buildRunCsv({ rmse: 18.5 }, { model_type: 'ets' })
    expect(csv).toBe('"type","key","value"\n"metric","rmse","18.5"\n"param","model_type","ets"')
  })

  it('escapes commas and quotes in param values', () => {
    const csv = buildRunCsv({}, { lags: '[1, 2, 3, 12]' })
    expect(csv).toContain('"param","lags","[1, 2, 3, 12]"')
    const csv2 = buildRunCsv({}, { note: 'say "hello"' })
    expect(csv2).toContain('"param","note","say ""hello"""')
  })

  it('sorts metrics and params alphabetically by key', () => {
    const csv = buildRunCsv({ rmse: 1, mae: 2 }, { z: '1', a: '2' })
    const lines = csv.split('\n')
    expect(lines[1]).toContain('"mae"')
    expect(lines[2]).toContain('"rmse"')
    expect(lines[3]).toContain('"a"')
    expect(lines[4]).toContain('"z"')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- format`
Expected: FAIL — `formatMetricValue` / `formatRunTime` / `buildRunCsv` are not exported.

- [ ] **Step 3: Implement the helpers**

Append to `frontend/lib/format.ts`:

```ts
export function formatMetricValue(v: number): string {
  if (!Number.isFinite(v)) return '—'
  return Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4)
}

export function formatRunTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}`
  )
}

export function buildRunCsv(
  metrics: Record<string, number>,
  params: Record<string, string>,
): string {
  const cell = (v: string) => `"${v.replace(/"/g, '""')}"`
  const rows = ['"type","key","value"']
  for (const [k, v] of Object.entries(metrics).sort(([a], [b]) => a.localeCompare(b))) {
    rows.push([cell('metric'), cell(k), cell(String(v))].join(','))
  }
  for (const [k, v] of Object.entries(params).sort(([a], [b]) => a.localeCompare(b))) {
    rows.push([cell('param'), cell(k), cell(v)].join(','))
  }
  return rows.join('\n')
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm run test -- format`
Expected: PASS (all `formatK`, `formatCost`, `formatMetricValue`, `formatRunTime`, `buildRunCsv` suites green).

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/format.ts frontend/__tests__/lib/format.test.ts
git commit -m "feat(experiments): metric/date/csv formatting helpers"
```

---

### Task 2: `RunDetailPanel` component (replaces `ChartPanel`)

Builds the new panel, wires it into the page, and removes the old `ChartPanel` + its test. Keeps the build green: the component exists and the page imports it before the old one is deleted.

**Files:**
- Create: `frontend/components/experiments/RunDetailPanel.tsx`
- Create: `frontend/__tests__/components/experiments/RunDetailPanel.test.tsx`
- Modify: `frontend/app/experiments/page.tsx`
- Delete: `frontend/components/experiments/ChartPanel.tsx`
- Delete: `frontend/__tests__/components/experiments/ChartPanel.test.tsx`

**Interfaces:**
- Consumes: `formatMetricValue`, `formatRunTime`, `buildRunCsv` from `@/lib/format` (Task 1); `RunOut` from `@/types/api`.
- Produces: `RunDetailPanel({ run }: { run: RunOut | null })` — used by `app/experiments/page.tsx`.

- [ ] **Step 1: Write the failing component test**

Create `frontend/__tests__/components/experiments/RunDetailPanel.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { RunDetailPanel } from '@/components/experiments/RunDetailPanel'
import type { RunOut } from '@/types/api'

const run: RunOut = {
  run_id: 'a1b2c3d4e5f6',
  run_name: 'extra_trees_forecaster',
  status: 'FINISHED',
  start_time: '2026-06-26T06:27:05+00:00',
  params: { season_length: '12', model_type: 'extra_trees_forecaster' },
  metrics: { rmse: 18.5, mae: 12.3 },
  metric_series: [],
}

describe('RunDetailPanel', () => {
  it('shows empty state when run is null', () => {
    render(<RunDetailPanel run={null} />)
    expect(screen.getByText(/select a run/i)).toBeInTheDocument()
  })

  it('renders a formatted metric value', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('18.500')).toBeInTheDocument()
  })

  it('renders a param key', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('season_length')).toBeInTheDocument()
  })

  it('renders a status badge label', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('complete')).toBeInTheDocument()
  })

  it('hides Export CSV when there are no metrics and no params', () => {
    render(<RunDetailPanel run={{ ...run, metrics: {}, params: {} }} />)
    expect(screen.queryByText(/export csv/i)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test -- RunDetailPanel`
Expected: FAIL — cannot resolve `@/components/experiments/RunDetailPanel`.

- [ ] **Step 3: Implement the component**

Create `frontend/components/experiments/RunDetailPanel.tsx`:

```tsx
'use client'
import type { RunOut } from '@/types/api'
import { formatMetricValue, formatRunTime, buildRunCsv } from '@/lib/format'

const STATUS_BADGE: Record<string, { label: string; className: string }> = {
  FINISHED: { label: 'complete', className: 'bg-emerald-50 text-emerald-700' },
  FAILED: { label: 'failed', className: 'bg-red-50 text-red-600' },
  KILLED: { label: 'killed', className: 'bg-red-50 text-red-600' },
  RUNNING: { label: 'running', className: 'bg-amber-50 text-amber-700' },
  SCHEDULED: { label: 'scheduled', className: 'bg-amber-50 text-amber-700' },
}

function statusBadge(status: string): { label: string; className: string } {
  return STATUS_BADGE[status] ?? { label: status.toLowerCase(), className: 'bg-zinc-100 text-zinc-600' }
}

function MetricCard({ name, value }: { name: string; value: number }) {
  return (
    <div className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs">
      <div className="text-zinc-500">{name}</div>
      <div className="font-mono text-zinc-800">{formatMetricValue(value)}</div>
    </div>
  )
}

function downloadCsv(run: RunOut) {
  const csv = buildRunCsv(run.metrics, run.params)
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `run-${run.run_id.slice(0, 8)}-metrics.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function RunDetailPanel({ run }: { run: RunOut | null }) {
  if (!run) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-400">
        Select a run to view its metrics
      </div>
    )
  }

  const metrics = Object.entries(run.metrics).sort(([a], [b]) => a.localeCompare(b))
  const params = Object.entries(run.params).sort(([a], [b]) => a.localeCompare(b))
  const badge = statusBadge(run.status)
  const canExport = metrics.length > 0 || params.length > 0

  return (
    <div className="h-full overflow-y-auto">
      <div className="space-y-6">
        {/* Header */}
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-zinc-900">{run.run_name}</h3>
            <span className={`rounded-full px-3 py-0.5 text-xs font-semibold ${badge.className}`}>
              {badge.label}
            </span>
            <span className="text-xs text-zinc-400">{formatRunTime(run.start_time)}</span>
            {canExport && (
              <button
                onClick={() => downloadCsv(run)}
                className="ml-auto rounded px-2 py-1 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
              >
                Export CSV
              </button>
            )}
          </div>
          <div className="font-mono text-[11px] text-zinc-400">run_id: {run.run_id.slice(0, 8)}…</div>
        </div>

        {/* Metrics */}
        <section>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Metrics</p>
          {metrics.length > 0 ? (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
              {metrics.map(([name, value]) => (
                <MetricCard key={name} name={name} value={value} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-400">No metrics logged</p>
          )}
        </section>

        {/* Configuration */}
        <section>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Configuration</p>
          {params.length > 0 ? (
            <table className="w-full text-xs">
              <tbody>
                {params.map(([k, v]) => (
                  <tr key={k} className="border-b border-zinc-100 last:border-0">
                    <td className="py-1 pr-4 align-top font-medium whitespace-nowrap text-zinc-600">{k}</td>
                    <td className="py-1 font-mono break-all text-zinc-800">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-zinc-400">No parameters logged</p>
          )}
        </section>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm run test -- RunDetailPanel`
Expected: PASS (all 5 cases).

- [ ] **Step 5: Wire the page to the new component**

Replace the contents of `frontend/app/experiments/page.tsx` with:

```tsx
'use client'
import { useState } from 'react'
import { RunSidebar } from '@/components/experiments/RunSidebar'
import { RunDetailPanel } from '@/components/experiments/RunDetailPanel'
import type { RunOut } from '@/types/api'

export default function ExperimentsPage() {
  const [selectedRun, setSelectedRun] = useState<RunOut | null>(null)
  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="w-64 shrink-0 overflow-hidden">
        <RunSidebar selectedRunId={selectedRun?.run_id ?? null} onSelectRun={setSelectedRun} />
      </div>
      <div className="flex-1 overflow-hidden rounded-lg border border-slate-200 bg-white p-4">
        <RunDetailPanel run={selectedRun} />
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Delete the old component and its test**

```bash
git rm frontend/components/experiments/ChartPanel.tsx frontend/__tests__/components/experiments/ChartPanel.test.tsx
```

- [ ] **Step 7: Verify type-check and full test suite pass**

Run: `npx tsc --noEmit && npm run test`
Expected: PASS — no references to `ChartPanel` remain; `RunDetailPanel` and `format` suites green. (The `charts/` folder still exists and still compiles at this point — it is removed in Task 3.)

- [ ] **Step 8: Commit**

```bash
git add frontend/components/experiments/RunDetailPanel.tsx \
        frontend/__tests__/components/experiments/RunDetailPanel.test.tsx \
        frontend/app/experiments/page.tsx
git commit -m "feat(experiments): replace ChartPanel with chart-free RunDetailPanel"
```

---

### Task 3: Delete chart components and remove `recharts`

Removes the now-orphaned chart files and the dependency they were the sole users of.

**Files:**
- Delete: `frontend/components/experiments/charts/` (folder: `TrainerLineChart.tsx`, `EvaluatorRadarChart.tsx`, `EvaluatorBarChart.tsx`, `DeploymentBarChart.tsx`)
- Modify: `frontend/package.json` (remove `recharts`)
- Modify: `frontend/package-lock.json` (regenerated by `npm install`)

**Interfaces:**
- Consumes: nothing (these files are orphaned once Task 2 deletes `ChartPanel`).
- Produces: nothing.

- [ ] **Step 1: Confirm nothing imports the charts or recharts anymore**

Run: `grep -rn "experiments/charts\|from 'recharts'" frontend --include=*.tsx --include=*.ts | grep -v node_modules`
Expected: only matches inside `frontend/components/experiments/charts/` itself (the files about to be deleted). No matches elsewhere.

- [ ] **Step 2: Delete the charts folder**

```bash
git rm -r frontend/components/experiments/charts
```

- [ ] **Step 3: Remove `recharts` from dependencies**

In `frontend/package.json`, delete this line from `dependencies`:

```json
    "recharts": "^3.8.1",
```

- [ ] **Step 4: Sync the lockfile**

Run: `npm install`
Expected: completes; `package-lock.json` updated to drop `recharts`.

- [ ] **Step 5: Verify build, type-check, tests, and lint all pass**

Run: `npx tsc --noEmit && npm run test && npm run lint && npm run build`
Expected: PASS — no dangling imports, no `recharts` references, production build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore(experiments): delete chart components and drop recharts"
```

---

## Manual verification (after all tasks)

Run the stack (`docker compose up`, or the API + `npm run dev` in `frontend/`), open the Experiments tab, and confirm:

- Select a forecasting run → header (name + `complete` badge + local time + `run_id`), metric cards, config table. No floating-dot chart, no broken `[0,1]` axis.
- Single border around the panel (no nested-Card double border).
- Panel scrolls when metrics + params are long.
- Select a run with no params → `No parameters logged`.
- A run with neither metrics nor params → both empty states shown, Export CSV button absent.
- Export CSV on a run with a comma-bearing param (e.g. `lags = [1, 2, 3, 12]`) → downloaded file opens with intact columns (value stays in one cell).
