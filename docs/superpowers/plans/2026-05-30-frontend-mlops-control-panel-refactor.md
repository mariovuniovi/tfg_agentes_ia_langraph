# Frontend MLOps Control Panel Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the dashboard UI into a Linear/Vercel-style MLOps control panel that visually distinguishes agentic, deterministic, and HITL nodes, with a pipeline stepper, structured HITL gates, an audit-report panel, a tabbed event log, and an observability page.

**Architecture:** Five vertical slices. Each slice ends with a runnable container build (`docker compose up --build`), a green test suite (`uv run pytest`, `cd frontend && npm test`), and a manual smoke check. Slice 1 lays the design-token foundation that every later slice consumes. Slice 3 includes a backend bug fix (`deployment_decision` overwrite) that is currently breaking the deploy path.

**Tech stack:** Next.js 16 + React 19 + Tailwind v4 + zustand + react-query + sonner (frontend); FastAPI + LangGraph + Pydantic v2 (backend). Tests: vitest + React Testing Library (frontend), pytest (backend). Geist font already ships with Next.js — no new deps.

**Spec:** [docs/superpowers/specs/2026-05-30-frontend-mlops-control-panel-refactor-design.md](../specs/2026-05-30-frontend-mlops-control-panel-refactor-design.md)

---

## File map

### Slice 1 — Foundation, stepper, run header
Create:
- `frontend/components/ui/Card.tsx`
- `frontend/components/ui/Badge.tsx`
- `frontend/components/ui/NodeTypeBadge.tsx`
- `frontend/lib/agent-display.ts`
- `frontend/lib/stage-derive.ts`
- `frontend/components/pipeline/RunHeader.tsx`
- `frontend/components/pipeline/PipelineStepper.tsx`
- `frontend/app/observability/page.tsx` (placeholder)
- `frontend/__tests__/lib/stage-derive.test.ts`
- `frontend/__tests__/components/pipeline/RunHeader.test.tsx`
- `frontend/__tests__/components/pipeline/PipelineStepper.test.tsx`

Modify:
- `frontend/app/globals.css` (token system)
- `frontend/components/TopNav.tsx` (4 tabs, new tokens)
- `frontend/app/pipeline/page.tsx` (new layout)
- `frontend/components/pipeline/RunStatusBadge.tsx` (use new tokens)
- `frontend/components/pipeline/EventLog.tsx`, `HITLGate.tsx`, `ResultsDashboard.tsx`, `TriggerPanel.tsx` (token migration)
- `frontend/components/experiments/ChartPanel.tsx`, `RunSidebar.tsx`, `charts/DeploymentBarChart.tsx` (token migration)
- `frontend/components/monitoring/AdHocForm.tsx`, `LatestReport.tsx` (token migration)
- `frontend/app/monitoring/page.tsx` (token migration)
- `api/services/pipeline.py` (rename `"supervisor"` → `"controller"` at 3 sites; add `problem_type` into `run_info` payload)

### Slice 2 — Dataset Approval Gate
Create:
- `frontend/components/pipeline/DatasetApprovalCard.tsx`
- `frontend/__tests__/components/pipeline/DatasetApprovalCard.test.tsx`
- `tests/api/test_dataset_preview.py`

Modify:
- `api/services/run_store.py` (add `processed_dataset_path` field)
- `api/services/pipeline.py` (stash path on HITL emission)
- `api/routers/runs.py` (add 2 endpoints — bare paths, no `/api` prefix per existing convention)
- `src/mlops_agents/graphs/approval_nodes.py` (**rename `preview` → `dataset_preview`** AND build full shape; add `tail` for forecasting)
- `frontend/components/pipeline/ResultsDashboard.tsx` (use `DatasetApprovalCard`)
- `frontend/types/api.ts` (extend `DataValidationInterrupt`)
- `frontend/lib/api.ts` (add `fetchDatasetPreview` helper)

### Slice 3 — Audit Report + Deployment Gate + Bug Fix
Create:
- `src/mlops_agents/evaluation/champion.py`
- `tests/test_evaluation/test_champion.py`
- `tests/test_graphs/test_deployment_flow.py` (regression for bug fix)
- `frontend/components/pipeline/AuditReportPanel.tsx`
- `frontend/components/pipeline/DeploymentApprovalCard.tsx`
- `frontend/__tests__/components/pipeline/AuditReportPanel.test.tsx`
- `frontend/__tests__/components/pipeline/DeploymentApprovalCard.test.tsx`

Modify:
- `src/mlops_agents/graphs/approval_nodes.py` (**bug fix** + enrich deployer HITL payload)
- `api/services/pipeline.py` (emit `audit_report` SSE event)
- `frontend/types/api.ts` (add `AuditReportEventData`, `DeployerInterrupt`)
- `frontend/lib/stage-derive.ts` (handle `audit_report`, eval-rejection path)
- `frontend/components/pipeline/ResultsDashboard.tsx` (Audit tab + auto-switch on `audit_report`)
- `frontend/components/pipeline/HITLGate.tsx` (use `DeploymentApprovalCard`)

### Slice 4 — Event Log Redesign
Create:
- `frontend/lib/events-aggregate.ts`
- `frontend/components/pipeline/EventLogTimeline.tsx`
- `frontend/components/pipeline/EventLogToolDetails.tsx`
- `frontend/__tests__/lib/events-aggregate.test.ts`
- `frontend/__tests__/components/pipeline/EventLogTimeline.test.tsx`

Modify:
- `frontend/components/pipeline/EventLog.tsx` (becomes tabbed shell)

### Slice 5 — Observability + Experiments Polish
Create:
- `frontend/components/observability/PipelineHealthCard.tsx`
- `frontend/components/observability/LlmActivityCard.tsx`
- `frontend/components/observability/ToolUsageCard.tsx`
- `tests/api/test_runs_list.py`

Modify:
- `api/services/run_store.py` (add `list_entries` accessor)
- `api/main.py` or `api/routers/runs.py` (add `GET /api/runs?limit=`)
- `frontend/lib/api.ts` (`fetchRunsList`)
- `frontend/app/observability/page.tsx` (real content)
- `frontend/components/experiments/RunSidebar.tsx` (champion badge + problem-type subtitle)
- `frontend/components/experiments/ChartPanel.tsx` (Card primitive)

---

# Slice 1 — Foundation, stepper, run header

## Task 1.1 — Add Tailwind v4 theme tokens

**Files:**
- Modify: `frontend/app/globals.css`

- [ ] **Step 1: Replace `globals.css` body**

```css
@import "tailwindcss";

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
  --color-info:          theme(colors.sky.600);
  --color-llm:           theme(colors.violet.600);

  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
}

body {
  background: var(--color-bg);
  color: var(--color-fg);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
}
```

The old `--color-navy` and `--color-amber` tokens are **deleted**. Tasks 1.10–1.12 migrate every component that still references them.

- [ ] **Step 2: Rebuild frontend and confirm the dev page still loads (visual breakage expected)**

```
cd frontend && npm run build
```

Expected: build succeeds. Some components will render unstyled (`bg-navy` etc. is now an unknown class); that is fine — tasks 1.10–1.12 will fix them.

- [ ] **Step 3: Commit**

```
git add frontend/app/globals.css
git commit -m "feat(frontend): introduce zinc/indigo design tokens; remove navy/amber"
```

---

## Task 1.2 — Create `<Card>` primitive

**Files:**
- Create: `frontend/components/ui/Card.tsx`

- [ ] **Step 1: Implement**

```tsx
import type { ReactNode } from 'react'

interface CardProps {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}

export function Card({ title, actions, children, className = '' }: CardProps) {
  return (
    <section
      className={`rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-4 py-2.5">
          {title && (
            <h3 className="text-sm font-semibold text-[var(--color-fg)]">{title}</h3>
          )}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}
```

- [ ] **Step 2: Commit**

```
git add frontend/components/ui/Card.tsx
git commit -m "feat(ui): add Card primitive"
```

---

## Task 1.3 — Create `<Badge>` primitive

**Files:**
- Create: `frontend/components/ui/Badge.tsx`

- [ ] **Step 1: Implement**

```tsx
import type { ReactNode } from 'react'

export type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'llm' | 'neutral' | 'accent'

const STYLES: Record<BadgeVariant, string> = {
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  warning: 'bg-amber-50  text-amber-700  ring-amber-200',
  danger:  'bg-red-50    text-red-700    ring-red-200',
  info:    'bg-sky-50    text-sky-700    ring-sky-200',
  llm:     'bg-violet-50 text-violet-700 ring-violet-200',
  accent:  'bg-indigo-50 text-indigo-700 ring-indigo-200',
  neutral: 'bg-zinc-100  text-zinc-700   ring-zinc-200',
}

interface BadgeProps {
  variant?: BadgeVariant
  children: ReactNode
  className?: string
}

export function Badge({ variant = 'neutral', children, className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${STYLES[variant]} ${className}`}
    >
      {children}
    </span>
  )
}
```

- [ ] **Step 2: Commit**

```
git add frontend/components/ui/Badge.tsx
git commit -m "feat(ui): add Badge primitive with 7 variants"
```

---

## Task 1.4 — Create `<NodeTypeBadge>`

**Files:**
- Create: `frontend/components/ui/NodeTypeBadge.tsx`

- [ ] **Step 1: Implement**

```tsx
import { Badge, type BadgeVariant } from './Badge'

export type NodeType = 'agent' | 'llm' | 'deterministic' | 'hitl'

const MAP: Record<NodeType, { variant: BadgeVariant; label: string }> = {
  agent:         { variant: 'llm',     label: 'Agent' },
  llm:           { variant: 'llm',     label: 'LLM' },
  deterministic: { variant: 'info',    label: 'Deterministic' },
  hitl:          { variant: 'warning', label: 'HITL' },
}

export function NodeTypeBadge({ type }: { type: NodeType }) {
  const { variant, label } = MAP[type]
  return <Badge variant={variant}>{label}</Badge>
}
```

- [ ] **Step 2: Commit**

```
git add frontend/components/ui/NodeTypeBadge.tsx
git commit -m "feat(ui): add NodeTypeBadge mapping 4 node types"
```

---

## Task 1.5 — Display-layer agent name map

**Files:**
- Create: `frontend/lib/agent-display.ts`

- [ ] **Step 1: Implement**

```ts
export const DISPLAY_AGENT: Record<string, string> = {
  supervisor:          'Controller',      // historical events only
  controller:          'Controller',
  workflow_controller: 'Controller',
  data_validator:      'Data Validator',
  dataset_approval:    'Dataset Approval',
  planner:             'Model Planner',
  executor:            'Training Executor',
  evaluation:          'Evaluation',
  report_writer:       'Audit Report',
  deployment_approval: 'Deployment Approval',
  deployer:            'Deployer',
  system:              'System',
}

export function displayAgentName(raw: string): string {
  return DISPLAY_AGENT[raw] ?? raw
}
```

- [ ] **Step 2: Commit**

```
git add frontend/lib/agent-display.ts
git commit -m "feat(lib): add agent display-name map (supervisor->Controller)"
```

---

## Task 1.6 — `deriveStages()` — write tests first

**Files:**
- Create: `frontend/__tests__/lib/stage-derive.test.ts`

- [ ] **Step 1: Create test file with full test set**

```ts
import { describe, it, expect } from 'vitest'
import { deriveStages, type StageKey } from '@/lib/stage-derive'
import type { PipelineEvent } from '@/types/api'

function ev(type: string, agent: string, data: Record<string, unknown> = {}): PipelineEvent {
  return { type, agent, timestamp_ms: Date.now(), data } as PipelineEvent
}

describe('deriveStages', () => {
  it('returns all stages pending for an empty event list', () => {
    const { stages, attempts, runOutcome } = deriveStages([], 'running')
    const keys: StageKey[] = [
      'data_validation', 'dataset_approval', 'model_planning',
      'training', 'evaluation', 'audit_report', 'deploy_approval', 'deploy',
    ]
    for (const k of keys) expect(stages[k]).toBe('pending')
    expect(attempts.data_validator).toBe(0)
    expect(runOutcome).toBe('running')
  })

  it('marks data_validation as running on routing event', () => {
    const events = [ev('routing', 'controller', { next: 'data_validator' })]
    const { stages, attempts } = deriveStages(events, 'running')
    expect(stages.data_validation).toBe('running')
    expect(attempts.data_validator).toBe(1)
  })

  it('marks data_validation as completed on validate_against_schema result', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
    ]
    expect(deriveStages(events, 'running').stages.data_validation).toBe('completed')
  })

  it('marks dataset_approval as waiting_human on hitl_request', () => {
    const events = [
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
    ]
    expect(deriveStages(events, 'awaiting_approval').stages.dataset_approval).toBe('waiting_human')
  })

  it('handles full happy path through deploy', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
      ev('approval_received', 'dataset_approval', { decision: 'approve' }),
      ev('routing', 'controller', { next: 'planner' }),
      ev('planner_context', 'planner'),
      ev('routing', 'controller', { next: 'executor' }),
      ev('tool_result', 'executor', { tool_name: 'train_model' }),
      ev('routing', 'controller', { next: 'evaluation' }),
      ev('routing', 'controller', { next: 'report_writer' }),
      ev('audit_report', 'report_writer', { evaluation_passed: true }),
      ev('hitl_request', 'deployer', { type: 'deployer' }),
      ev('approval_received', 'deployer', { decision: 'approve' }),
      ev('run_complete', 'controller'),
    ]
    const { stages, runOutcome } = deriveStages(events, 'complete')
    expect(stages.deploy).toBe('completed')
    expect(runOutcome).toBe('complete')
  })

  it('eval-rejection path produces candidate_rejected outcome', () => {
    const events = [
      ev('routing', 'controller', { next: 'evaluation' }),
      ev('routing', 'controller', { next: 'report_writer' }),
      ev('audit_report', 'report_writer', { evaluation_passed: false }),
      ev('run_complete', 'controller'),
    ]
    const { stages, runOutcome } = deriveStages(events, 'complete')
    expect(stages.evaluation).toBe('completed')
    expect(stages.audit_report).toBe('completed')
    expect(stages.deploy_approval).toBe('skipped')
    expect(stages.deploy).toBe('skipped')
    expect(runOutcome).toBe('candidate_rejected')
  })

  it('retry increments attempts and rewinds dataset_approval', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
      ev('approval_received', 'dataset_approval', { decision: 'reject' }),
      ev('routing', 'controller', { next: 'data_validator' }),
    ]
    const { stages, attempts } = deriveStages(events, 'running')
    expect(stages.data_validation).toBe('running')
    expect(stages.dataset_approval).toBe('pending')
    expect(attempts.data_validator).toBe(2)
  })

  it('run_complete with error marks current stage failed', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }),
      ev('run_complete', 'controller', { error: 'boom' }),
    ]
    const { stages, runOutcome } = deriveStages(events, 'failed')
    expect(stages.model_planning).toBe('failed')
    expect(runOutcome).toBe('failed')
  })

  it('deploy rejection skips deploy stage', () => {
    const events = [
      ev('hitl_request', 'deployer', { type: 'deployer' }),
      ev('approval_received', 'deployer', { decision: 'reject' }),
      ev('run_complete', 'controller'),
    ]
    expect(deriveStages(events, 'complete').stages.deploy).toBe('skipped')
  })
})
```

- [ ] **Step 2: Run test, expect failure**

```
cd frontend && npm test -- stage-derive
```

Expected: `Cannot find module '@/lib/stage-derive'`.

---

## Task 1.7 — Implement `deriveStages()`

**Files:**
- Create: `frontend/lib/stage-derive.ts`

- [ ] **Step 1: Implement**

```ts
import type { PipelineEvent, RunStatus } from '@/types/api'

export type StageKey =
  | 'data_validation' | 'dataset_approval' | 'model_planning'
  | 'training' | 'evaluation' | 'audit_report'
  | 'deploy_approval' | 'deploy'

export type StageStatus =
  | 'pending' | 'running' | 'completed'
  | 'waiting_human' | 'failed' | 'skipped'

export type RunOutcome = 'running' | 'complete' | 'failed' | 'candidate_rejected'

const INITIAL: Record<StageKey, StageStatus> = {
  data_validation: 'pending',
  dataset_approval: 'pending',
  model_planning: 'pending',
  training: 'pending',
  evaluation: 'pending',
  audit_report: 'pending',
  deploy_approval: 'pending',
  deploy: 'pending',
}

const STAGE_ORDER: StageKey[] = [
  'data_validation', 'dataset_approval', 'model_planning',
  'training', 'evaluation', 'audit_report', 'deploy_approval', 'deploy',
]

export function deriveStages(
  events: PipelineEvent[],
  runStatus: RunStatus | 'idle',
): {
  stages: Record<StageKey, StageStatus>
  attempts: { data_validator: number }
  runOutcome: RunOutcome
} {
  const stages = { ...INITIAL }
  const attempts = { data_validator: 0 }
  let currentStage: StageKey | null = null
  let lastDeployDecision: 'approve' | 'reject' | null = null
  let evaluationPassed: boolean | null = null

  for (const e of events) {
    const next = (e.data as { next?: string }).next
    const tool = (e.data as { tool_name?: string }).tool_name
    const hitlType = (e.data as { type?: string }).type
    const decision = (e.data as { decision?: string }).decision

    if (e.type === 'routing' && next === 'data_validator') {
      // retry: rewind dataset_approval if already completed
      if (stages.dataset_approval === 'completed') {
        stages.dataset_approval = 'pending'
      }
      stages.data_validation = 'running'
      currentStage = 'data_validation'
      attempts.data_validator += 1
    }
    if (e.type === 'tool_result' && tool === 'validate_against_schema') {
      stages.data_validation = 'completed'
    }
    if (e.type === 'hitl_request' && hitlType === 'data_validation') {
      stages.dataset_approval = 'waiting_human'
      currentStage = 'dataset_approval'
    }
    if (e.type === 'approval_received' && e.agent === 'dataset_approval') {
      stages.dataset_approval = decision === 'approve' ? 'completed' : 'pending'
    }
    if (e.type === 'routing' && next === 'planner') {
      stages.model_planning = 'running'
      currentStage = 'model_planning'
    }
    if (e.type === 'planner_context') {
      stages.model_planning = 'completed'
    }
    if (e.type === 'routing' && next === 'executor') {
      stages.training = 'running'
      currentStage = 'training'
    }
    if (e.type === 'tool_result' && (tool === 'train_model' || tool === 'tune_hyperparameters')) {
      stages.training = 'completed'
    }
    if (e.type === 'routing' && next === 'evaluation') {
      stages.evaluation = 'running'
      currentStage = 'evaluation'
    }
    if (e.type === 'routing' && next === 'report_writer') {
      if (stages.evaluation === 'running') stages.evaluation = 'completed'
      stages.audit_report = 'running'
      currentStage = 'audit_report'
    }
    if (e.type === 'audit_report') {
      stages.audit_report = 'completed'
      const passed = (e.data as { evaluation_passed?: boolean }).evaluation_passed
      if (typeof passed === 'boolean') evaluationPassed = passed
    }
    if (e.type === 'hitl_request' && hitlType === 'deployer') {
      stages.deploy_approval = 'waiting_human'
      currentStage = 'deploy_approval'
    }
    if (e.type === 'approval_received' && e.agent === 'deployer') {
      stages.deploy_approval = 'completed'
      lastDeployDecision = decision === 'approve' ? 'approve' : 'reject'
    }
    if (e.type === 'run_complete') {
      const errorMsg = (e.data as { error?: string }).error
      if (errorMsg && currentStage) {
        stages[currentStage] = 'failed'
      }
    }
  }

  // Determine outcome and finalize skipped stages on terminal status
  let runOutcome: RunOutcome = 'running'
  if (runStatus === 'failed') {
    runOutcome = 'failed'
  } else if (runStatus === 'complete') {
    if (evaluationPassed === false) {
      runOutcome = 'candidate_rejected'
      stages.deploy_approval = 'skipped'
      stages.deploy = 'skipped'
    } else if (lastDeployDecision === 'reject') {
      runOutcome = 'complete'
      stages.deploy = 'skipped'
    } else if (lastDeployDecision === 'approve' || stages.deploy === 'pending') {
      runOutcome = 'complete'
      if (lastDeployDecision === 'approve') stages.deploy = 'completed'
    } else {
      runOutcome = 'complete'
    }
    // mark any still-pending downstream as skipped on terminal
    let foundActive = false
    for (let i = STAGE_ORDER.length - 1; i >= 0; i--) {
      const k = STAGE_ORDER[i]
      if (stages[k] === 'pending' && !foundActive) stages[k] = 'skipped'
      else foundActive = true
    }
  }

  return { stages, attempts, runOutcome }
}
```

- [ ] **Step 2: Run tests, expect all pass**

```
cd frontend && npm test -- stage-derive
```

Expected: 9 tests pass.

- [ ] **Step 3: Commit**

```
git add frontend/lib/stage-derive.ts frontend/__tests__/lib/stage-derive.test.ts
git commit -m "feat(lib): add deriveStages() with full state machine + tests"
```

---

## Task 1.8 — `<PipelineStepper>` component

**Files:**
- Create: `frontend/components/pipeline/PipelineStepper.tsx`
- Create: `frontend/__tests__/components/pipeline/PipelineStepper.test.tsx`

- [ ] **Step 1: Test first**

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PipelineStepper } from '@/components/pipeline/PipelineStepper'

const allPending = {
  data_validation: 'pending', dataset_approval: 'pending', model_planning: 'pending',
  training: 'pending', evaluation: 'pending', audit_report: 'pending',
  deploy_approval: 'pending', deploy: 'pending',
} as const

describe('<PipelineStepper>', () => {
  it('renders 8 named stages', () => {
    render(<PipelineStepper stages={allPending} />)
    expect(screen.getByText('Data Validation')).toBeInTheDocument()
    expect(screen.getByText('Dataset Approval')).toBeInTheDocument()
    expect(screen.getByText('Model Planning')).toBeInTheDocument()
    expect(screen.getByText('Training')).toBeInTheDocument()
    expect(screen.getByText('Evaluation')).toBeInTheDocument()
    expect(screen.getByText('Audit Report')).toBeInTheDocument()
    expect(screen.getByText('Deploy Approval')).toBeInTheDocument()
    expect(screen.getByText('Deploy')).toBeInTheDocument()
  })

  it('marks a completed stage with a check', () => {
    render(<PipelineStepper stages={{ ...allPending, training: 'completed' }} />)
    expect(screen.getByTestId('stage-training').textContent).toContain('✓')
  })

  it('renders waiting_human stage with a clock', () => {
    render(<PipelineStepper stages={{ ...allPending, dataset_approval: 'waiting_human' }} />)
    expect(screen.getByTestId('stage-dataset_approval').textContent).toContain('⏱')
  })
})
```

- [ ] **Step 2: Implement**

```tsx
'use client'
import { NodeTypeBadge, type NodeType } from '@/components/ui/NodeTypeBadge'
import type { StageKey, StageStatus } from '@/lib/stage-derive'

const STAGES: Array<{ key: StageKey; label: string; type: NodeType }> = [
  { key: 'data_validation',  label: 'Data Validation',  type: 'agent' },
  { key: 'dataset_approval', label: 'Dataset Approval', type: 'hitl' },
  { key: 'model_planning',   label: 'Model Planning',   type: 'llm' },
  { key: 'training',         label: 'Training',         type: 'deterministic' },
  { key: 'evaluation',       label: 'Evaluation',       type: 'deterministic' },
  { key: 'audit_report',     label: 'Audit Report',     type: 'llm' },
  { key: 'deploy_approval',  label: 'Deploy Approval',  type: 'hitl' },
  { key: 'deploy',           label: 'Deploy',           type: 'deterministic' },
]

const ICON: Record<StageStatus, string> = {
  pending:       '·',
  running:       '◐',
  completed:     '✓',
  waiting_human: '⏱',
  failed:        '✗',
  skipped:       '—',
}

const COLOR: Record<StageStatus, string> = {
  pending:       'text-[var(--color-fg-subtle)]',
  running:       'text-[var(--color-accent)]',
  completed:     'text-[var(--color-success)]',
  waiting_human: 'text-[var(--color-warning)]',
  failed:        'text-[var(--color-danger)]',
  skipped:       'text-[var(--color-fg-subtle)] opacity-60',
}

export function PipelineStepper({
  stages,
}: {
  stages: Record<StageKey, StageStatus>
}) {
  return (
    <ol className="flex flex-wrap items-stretch gap-2 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-2">
      {STAGES.map(({ key, label, type }) => {
        const status = stages[key]
        return (
          <li
            key={key}
            data-testid={`stage-${key}`}
            className={`flex min-w-[140px] flex-1 flex-col gap-1 rounded border border-[var(--color-border)] px-3 py-2 text-xs ${COLOR[status]}`}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-base leading-none">{ICON[status]}</span>
              <span className="font-medium text-[var(--color-fg)]">{label}</span>
            </div>
            <NodeTypeBadge type={type} />
          </li>
        )
      })}
    </ol>
  )
}
```

- [ ] **Step 3: Run tests, expect pass**

```
cd frontend && npm test -- PipelineStepper
```

- [ ] **Step 4: Commit**

```
git add frontend/components/pipeline/PipelineStepper.tsx frontend/__tests__/components/pipeline/PipelineStepper.test.tsx
git commit -m "feat(pipeline): add PipelineStepper with 8 stages"
```

---

## Task 1.9 — `<RunHeader>` component

**Files:**
- Create: `frontend/components/pipeline/RunHeader.tsx`
- Create: `frontend/__tests__/components/pipeline/RunHeader.test.tsx`

- [ ] **Step 1: Test first**

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RunHeader } from '@/components/pipeline/RunHeader'

describe('<RunHeader>', () => {
  it('renders truncated run id and problem type', () => {
    render(
      <RunHeader
        runId="37c17107abcd"
        problemType="forecasting"
        stageLabel="Waiting for human"
        startedMs={Date.now() - 5000}
        runOutcome="running"
        attemptCount={1}
        llmModels={['data_validator', 'planner', 'report_writer']}
      />,
    )
    expect(screen.getByText(/37c17107/)).toBeInTheDocument()
    expect(screen.getByText(/forecasting/i)).toBeInTheDocument()
    expect(screen.getByText(/Waiting for human/)).toBeInTheDocument()
  })

  it('renders sky pill for candidate_rejected outcome', () => {
    render(
      <RunHeader
        runId="abc"
        problemType="classification"
        stageLabel="Candidate rejected"
        startedMs={Date.now()}
        runOutcome="candidate_rejected"
        attemptCount={1}
        llmModels={[]}
      />,
    )
    const pill = screen.getByTestId('run-status-pill')
    expect(pill.className).toMatch(/sky/)
  })

  it('shows attempt counter when > 1', () => {
    render(
      <RunHeader
        runId="abc"
        problemType="forecasting"
        stageLabel="Data Validation"
        startedMs={Date.now()}
        runOutcome="running"
        attemptCount={2}
        llmModels={[]}
      />,
    )
    expect(screen.getByText(/attempt 2/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Implement**

```tsx
'use client'
import { useEffect, useState } from 'react'

const OUTCOME_PILL: Record<string, string> = {
  running:             'bg-indigo-50 text-indigo-700 ring-indigo-200',
  complete:            'bg-emerald-50 text-emerald-700 ring-emerald-200',
  failed:              'bg-red-50 text-red-700 ring-red-200',
  candidate_rejected:  'bg-sky-50 text-sky-700 ring-sky-200',
}

function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}m ${s}s`
}

interface RunHeaderProps {
  runId: string
  problemType: string
  stageLabel: string
  startedMs: number
  runOutcome: 'running' | 'complete' | 'failed' | 'candidate_rejected'
  attemptCount: number
  llmModels: string[]
}

export function RunHeader({
  runId, problemType, stageLabel, startedMs,
  runOutcome, attemptCount, llmModels,
}: RunHeaderProps) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (runOutcome !== 'running') return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [runOutcome])

  return (
    <header className="sticky top-0 z-10 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-mono text-xs text-[var(--color-fg-muted)]">
          Run <button title={runId} className="text-[var(--color-fg)] hover:underline" onClick={() => navigator.clipboard.writeText(runId)}>{runId.slice(0, 8)}</button>
        </span>
        <span className="text-xs text-[var(--color-fg-muted)] capitalize">{problemType || '—'}</span>
        <span className="text-xs text-[var(--color-fg)]">{stageLabel}</span>
        {attemptCount > 1 && (
          <span className="text-xs text-[var(--color-warning)]">Attempt {attemptCount}</span>
        )}
        <span className="ml-auto text-xs text-[var(--color-fg-muted)]">
          {formatElapsed(now - startedMs)}
        </span>
        <span
          data-testid="run-status-pill"
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${OUTCOME_PILL[runOutcome] ?? OUTCOME_PILL.running}`}
        >
          {runOutcome.replace('_', ' ')}
        </span>
      </div>
      <div className="mt-1 text-[11px] text-[var(--color-fg-subtle)]">
        <span>LLM: {llmModels.join(' · ') || '—'}</span>
        <span className="ml-4">Deterministic: controller · executor · evaluation · deployer</span>
      </div>
    </header>
  )
}
```

- [ ] **Step 3: Run tests, expect pass**

```
cd frontend && npm test -- RunHeader
```

- [ ] **Step 4: Commit**

```
git add frontend/components/pipeline/RunHeader.tsx frontend/__tests__/components/pipeline/RunHeader.test.tsx
git commit -m "feat(pipeline): add RunHeader with elapsed timer and outcome pill"
```

---

## Task 1.10 — Migrate `TopNav` + add Observability tab + new tokens

**Files:**
- Modify: `frontend/components/TopNav.tsx`
- Create: `frontend/app/observability/page.tsx`

- [ ] **Step 1: Replace `TopNav.tsx` body**

```tsx
'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '@/lib/api'

const TABS = [
  { label: 'Pipeline',      href: '/pipeline' },
  { label: 'Experiments',   href: '/experiments' },
  { label: 'Observability', href: '/observability' },
  { label: 'Monitoring',    href: '/monitoring' },
]

export function TopNav() {
  const pathname = usePathname()
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  const unhealthy = health && (!health.mlflow || !health.graph)

  return (
    <nav className="flex items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2">
      {TABS.map(({ label, href }) => {
        const active = pathname.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
              active
                ? 'bg-indigo-50 text-indigo-700'
                : 'text-[var(--color-fg-muted)] hover:bg-zinc-100 hover:text-[var(--color-fg)]'
            }`}
          >
            {label}
          </Link>
        )
      })}
      {unhealthy && (
        <span className="ml-auto rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-200">
          Backend unhealthy
        </span>
      )}
    </nav>
  )
}
```

- [ ] **Step 2: Create observability placeholder page**

`frontend/app/observability/page.tsx`:

```tsx
export default function ObservabilityPage() {
  return (
    <div className="p-6 text-sm text-[var(--color-fg-muted)]">
      Observability dashboard arriving in a later slice.
    </div>
  )
}
```

- [ ] **Step 3: Update test for new tabs**

Modify `frontend/__tests__/components/TopNav.test.tsx` — replace the assertion that there are 3 tabs with one that verifies 4 tab labels (`Pipeline`, `Experiments`, `Observability`, `Monitoring`). Read the existing file to see the exact assertion shape; mirror it.

- [ ] **Step 4: Run frontend tests**

```
cd frontend && npm test -- TopNav
```

Expected: pass.

- [ ] **Step 5: Commit**

```
git add frontend/components/TopNav.tsx frontend/app/observability/page.tsx frontend/__tests__/components/TopNav.test.tsx
git commit -m "feat(nav): 4 tabs (Pipeline/Experiments/Observability/Monitoring) + new token palette"
```

---

## Task 1.11 — Token migration: bulk replace `navy`/`amber`/`slate-*` in remaining components

**Files (modify all):**
- `frontend/components/pipeline/EventLog.tsx`
- `frontend/components/pipeline/HITLGate.tsx`
- `frontend/components/pipeline/ResultsDashboard.tsx`
- `frontend/components/pipeline/RunStatusBadge.tsx`
- `frontend/components/pipeline/TriggerPanel.tsx`
- `frontend/components/experiments/ChartPanel.tsx`
- `frontend/components/experiments/RunSidebar.tsx`
- `frontend/components/experiments/charts/DeploymentBarChart.tsx`
- `frontend/components/monitoring/AdHocForm.tsx`
- `frontend/components/monitoring/LatestReport.tsx`
- `frontend/app/monitoring/page.tsx`

- [ ] **Step 1: Apply this className mapping in every listed file**

| Old class | New class |
|---|---|
| `bg-navy` / `bg-navy-900` / `bg-navy-700` | `bg-indigo-600` |
| `text-navy-900` | `text-zinc-900` |
| `text-navy` | `text-indigo-700` |
| `border-navy*` | `border-indigo-600` |
| `bg-amber-50` | `bg-amber-50` (keep — semantic warning) |
| `bg-amber-600` / `bg-amber-100` (when used as a primary accent, not status) | `bg-amber-500` |
| `border-amber-600` | `border-amber-500` |
| `text-amber-800` / `text-amber-900` | `text-amber-700` |
| `bg-slate-50` | `bg-zinc-50` |
| `bg-slate-100` | `bg-zinc-100` |
| `bg-slate-200` | `bg-zinc-200` |
| `text-slate-300` | `text-zinc-300` |
| `text-slate-400` | `text-zinc-400` |
| `text-slate-500` | `text-zinc-500` |
| `text-slate-600` | `text-zinc-600` |
| `text-slate-700` | `text-zinc-700` |
| `border-slate-*` (any) | `border-zinc-200` |

Rule for ambiguity: if the old `amber` was an attention/warning color (HITL gate badge, "awaiting" pill), keep amber. If it was used as a brand accent (e.g., a deploy button), switch to `indigo-600`.

- [ ] **Step 2: Re-check that `RunStatusBadge` STYLES uses new tokens**

After migration `RunStatusBadge.tsx` should look like:

```tsx
'use client'
import { useRunStore } from '@/stores/run-store'
import type { RunStatus } from '@/types/api'

const STYLES: Record<RunStatus | 'idle', string> = {
  idle: 'bg-zinc-100 text-zinc-600',
  running: 'animate-pulse bg-indigo-50 text-indigo-700',
  awaiting_approval: 'bg-amber-50 text-amber-700',
  complete: 'bg-emerald-50 text-emerald-700',
  failed: 'bg-red-50 text-red-700',
}

export function RunStatusBadge() {
  const status = useRunStore((s) => s.status)
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${STYLES[status] ?? ''}`}>
      {status}
    </span>
  )
}
```

- [ ] **Step 3: Boot the container and visually scan every page**

```
docker compose up --build
```

Open http://localhost:3000 and click through Pipeline, Experiments, Observability, Monitoring. Expected: no raw navy/amber remains; consistent zinc/indigo look; no broken layouts.

- [ ] **Step 4: Run all frontend tests**

```
cd frontend && npm test
```

Expected: 100% pass (some snapshots may need updating — re-run with `-u` if so).

- [ ] **Step 5: Commit**

```
git add frontend/components frontend/app/monitoring/page.tsx
git commit -m "refactor(frontend): migrate 11 components from navy/slate to zinc/indigo tokens"
```

---

## Task 1.12 — Backend: rename `supervisor` → `controller` + add `problem_type` to `run_info`

**Files:**
- Modify: `api/services/pipeline.py`

- [ ] **Step 1: Replace 3 `"supervisor"` string literals with `"controller"`**

Find each occurrence of `"agent": "supervisor"` (lines ~208, ~234, ~261 — use a search) and replace each with `"agent": "controller"`.

- [ ] **Step 2: Extend `run_info` payload with problem_type**

In `pipeline_task`, the `info_event` dict currently has `"data": {"models": {...}}`. Add `problem_type` so the RunHeader can display it from event 0:

```python
    # Read problem_type from the schema JSON the caller posted
    import json as _json
    pt = ""
    try:
        pt = _json.loads(schema_json or "{}").get("problem_type", "")
    except Exception:
        pt = ""

    info_event: dict = {
        "type": "run_info",
        "agent": "system",
        "timestamp_ms": time.time() * 1000,
        "data": {
            "models": {
                "data_validator": settings.openai_model_data_validator,
                "planner":        settings.openai_model_planner,
                "report_writer":  settings.openai_model_report_writer,
            },
            "problem_type": pt,
        },
    }
```

- [ ] **Step 3: Verify nothing in `api/` still emits `"supervisor"`**

```
grep -nR "['\"]supervisor['\"]" api/
```

Expected: only docstring or comment hits, if any.

- [ ] **Step 4: Run backend tests**

```
uv run pytest -m "not integration"
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add api/services/pipeline.py
git commit -m "refactor(api): emit agent='controller'; add problem_type to run_info payload"
```

---

## Task 1.13 — Wire `RunHeader` + `PipelineStepper` into `/pipeline`

**Files:**
- Modify: `frontend/app/pipeline/page.tsx`

- [ ] **Step 1: Replace the page body**

```tsx
'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { TriggerPanel } from '@/components/pipeline/TriggerPanel'
import { HITLGate } from '@/components/pipeline/HITLGate'
import { EventLog } from '@/components/pipeline/EventLog'
import { ResultsDashboard } from '@/components/pipeline/ResultsDashboard'
import { RunHeader } from '@/components/pipeline/RunHeader'
import { PipelineStepper } from '@/components/pipeline/PipelineStepper'
import { useRunStream } from '@/hooks/use-run-stream'
import { deriveStages } from '@/lib/stage-derive'

const STAGE_LABELS: Record<string, string> = {
  data_validation: 'Data Validation',
  dataset_approval: 'Awaiting dataset approval',
  model_planning: 'Model Planning',
  training: 'Training',
  evaluation: 'Evaluation',
  audit_report: 'Generating audit report',
  deploy_approval: 'Awaiting deployment approval',
  deploy: 'Deploying',
}

export default function PipelinePage() {
  const runId = useRunStore((s) => s.runId)
  const status = useRunStore((s) => s.status)
  const events = useRunStore((s) => s.events)
  useRunStream(runId)

  const { stages, attempts, runOutcome } = useMemo(
    () => deriveStages(events, status),
    [events, status],
  )

  const activeStage = useMemo(() => {
    const order = ['deploy', 'deploy_approval', 'audit_report', 'evaluation', 'training', 'model_planning', 'dataset_approval', 'data_validation'] as const
    for (const k of order) {
      if (stages[k] === 'running' || stages[k] === 'waiting_human') return k
    }
    return null
  }, [stages])

  const problemType = useMemo(() => {
    const tp = events.find((e) => e.type === 'run_info')
    return (tp?.data as { problem_type?: string } | undefined)?.problem_type ?? ''
  }, [events])

  const llmModels = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])

  const startedMs = events[0]?.timestamp_ms ?? Date.now()

  return (
    <div className="space-y-3 p-3">
      {runId && (
        <>
          <RunHeader
            runId={runId}
            problemType={problemType}
            stageLabel={activeStage ? STAGE_LABELS[activeStage] : runOutcome}
            startedMs={startedMs}
            runOutcome={runOutcome}
            attemptCount={attempts.data_validator}
            llmModels={llmModels}
          />
          <PipelineStepper stages={stages} />
        </>
      )}
      <div className="grid grid-cols-5 gap-3">
        <div className="col-span-3 flex flex-col gap-3">
          <TriggerPanel />
          <ResultsDashboard />
          <HITLGate runId={runId} />
        </div>
        <div className="col-span-2">
          <EventLog />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Boot container and smoke-check the empty + active states**

```
docker compose up --build
```

- Open http://localhost:3000/pipeline. No run yet — RunHeader and Stepper should NOT render.
- Submit a small classification dataset via TriggerPanel.
- Expected: RunHeader appears at top, Stepper shows `Data Validation = running`, progressing through stages live as events arrive.

- [ ] **Step 3: Commit**

```
git add frontend/app/pipeline/page.tsx
git commit -m "feat(pipeline): wire RunHeader + PipelineStepper into /pipeline layout"
```

---

## Task 1.14 — Slice 1 smoke-check task

- [ ] **Step 1: Run full test suites**

```
uv run pytest -m "not integration"
cd frontend && npm test
```

Both must be green.

- [ ] **Step 2: Manual smoke**

```
docker compose down -v && docker compose up --build
```

Submit a real run end-to-end. Verify:
- Top nav shows 4 tabs.
- `Observability` tab shows the placeholder copy.
- `Monitoring` tab still works (Evidently drift unaffected).
- Pipeline page shows RunHeader + Stepper at top.
- Stepper visibly transitions stages live.
- No raw `navy` or `amber-600`-as-accent colors visible.
- Routing events in the EventLog show "controller" not "supervisor".

If anything fails, fix in a follow-up task before tagging slice 1 complete.

---

# Slice 2 — Dataset Approval Gate

## Task 2.1 — Extend `RunEntry` with `processed_dataset_path`

**Files:**
- Modify: `api/services/run_store.py`

- [ ] **Step 1: Add field**

In `RunEntry`, add a new field after `interrupt_value`:

```python
    processed_dataset_path: str | None = None
```

- [ ] **Step 2: Commit**

```
git add api/services/run_store.py
git commit -m "feat(api): add RunEntry.processed_dataset_path for dataset preview endpoints"
```

---

## Task 2.2 — Stash dataset path on HITL emission

**Files:**
- Modify: `api/services/pipeline.py`

- [ ] **Step 1: Inside `_stream`, in the `__interrupt__` handler**

Locate the block that builds `hitl_event` when `mode == "updates"`. Just before appending `hitl_event` to `entry.events`, add:

```python
                    if hitl_agent == "data_validation":
                        preview = (interrupt_val.get("dataset_preview") or {})
                        path = preview.get("path")
                        if isinstance(path, str) and path:
                            entry.processed_dataset_path = path
```

- [ ] **Step 2: Commit**

```
git add api/services/pipeline.py
git commit -m "feat(api): stash processed_dataset_path on RunEntry at HITL emission"
```

---

## Task 2.3 — Backend: build the full `dataset_preview` (key rename + shape build + tail)

**Why this is bigger than originally planned:** `dataset_approval_node` currently emits the payload key `"preview"` (not `"dataset_preview"`) and backs it with `state["dataset_summary"]`, whose shape is `{row_count, column_names, dtypes, null_counts}` — none of which matches what the frontend reads. So the dataset gate preview has been silently empty in production. This task fixes the key, builds the shape the frontend actually consumes, and adds `tail` for forecasting in one coherent change.

**Files:**
- Modify: `src/mlops_agents/graphs/approval_nodes.py`
- Create: `tests/test_graphs/test_dataset_approval_payload.py`

- [ ] **Step 1: Write failing test for the full payload shape**

```python
import pandas as pd

def _capture(monkeypatch):
    from mlops_agents.graphs import approval_nodes
    captured: dict = {}
    def fake_interrupt(payload):
        captured["payload"] = payload
        return {"approved": True, "comment": ""}
    monkeypatch.setattr(approval_nodes, "interrupt", fake_interrupt)
    return captured

def test_payload_uses_dataset_preview_key_with_correct_shape(tmp_path, monkeypatch):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": list(range(10)), "b": list("abcdefghij")}).to_csv(csv, index=False)
    captured = _capture(monkeypatch)
    from mlops_agents.graphs.approval_nodes import dataset_approval_node
    dataset_approval_node({
        "problem_type": "classification",
        "processed_dataset_path": str(csv),
        "validation_report": {"passed": True},
        "agent_attempt_counts": {},
    })
    payload = captured["payload"]
    assert "dataset_preview" in payload
    assert "preview" not in payload  # legacy key must be gone
    p = payload["dataset_preview"]
    assert p["path"] == str(csv)
    assert p["row_count"] == 10
    assert p["column_count"] == 2
    assert p["shape"] == [10, 2]
    assert {c["name"] for c in p["columns"]} == {"a", "b"}
    assert all("dtype" in c for c in p["columns"])
    assert len(p["sample_rows"]) == 5
    assert p["tail"] == []  # non-forecasting

def test_tail_populated_for_forecasting(tmp_path, monkeypatch):
    csv = tmp_path / "ts.csv"
    pd.DataFrame({"ds": list(range(10)), "y": list(range(10))}).to_csv(csv, index=False)
    captured = _capture(monkeypatch)
    from mlops_agents.graphs.approval_nodes import dataset_approval_node
    dataset_approval_node({
        "problem_type": "forecasting",
        "processed_dataset_path": str(csv),
        "validation_report": {"passed": True},
        "agent_attempt_counts": {},
    })
    p = captured["payload"]["dataset_preview"]
    assert len(p["tail"]) == 5
    assert p["tail"][-1] == {"ds": 9, "y": 9}
```

- [ ] **Step 2: Run test, expect fail**

```
uv run pytest tests/test_graphs/test_dataset_approval_payload.py -v
```

Expected: payload uses `preview` key (legacy) and lacks the new shape fields.

- [ ] **Step 3: Rewrite `dataset_approval_node` to build the full shape**

Replace the body of `dataset_approval_node` in `src/mlops_agents/graphs/approval_nodes.py`:

```python
def dataset_approval_node(state: dict[str, Any]) -> Command:
    import pandas as pd
    counts = state.get("agent_attempt_counts") or {}
    attempt = counts.get("data_validator", 1)

    path = state.get("processed_dataset_path", "")
    preview: dict[str, Any] = {
        "path": path,
        "row_count": 0,
        "column_count": 0,
        "shape": [0, 0],
        "columns": [],
        "sample_rows": [],
        "head": [],
        "tail": [],
    }
    if path:
        try:
            df = pd.read_csv(path)
            head = df.head(5).to_dict(orient="records")
            tail = (
                df.tail(5).to_dict(orient="records")
                if state.get("problem_type") == "forecasting"
                else []
            )
            preview = {
                "path": path,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "shape": [int(len(df)), int(len(df.columns))],
                "columns": [
                    {"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns
                ],
                "sample_rows": head,
                "head": head,
                "tail": tail,
            }
        except Exception as exc:
            logger.warning(f"[gate1] failed to build dataset_preview from {path}: {exc}")

    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "dataset_preview": preview,
        "validation_report": state.get("validation_report", {}),
    })
    approved = bool(approval.get("approved", False))
    comment = approval.get("comment", "")
    logger.info(f"[gate1] dataset_approved={approved} comment={comment!r}")
    return Command(
        goto="workflow_controller",
        update={
            "dataset_approved": approved,
            "dataset_rejection_comment": "" if approved else comment,
        },
    )
```

- [ ] **Step 4: Run test, expect all pass**

```
uv run pytest tests/test_graphs/test_dataset_approval_payload.py -v
```

- [ ] **Step 5: Commit**

```
git add src/mlops_agents/graphs/approval_nodes.py tests/test_graphs/test_dataset_approval_payload.py
git commit -m "fix(graph): build full dataset_preview shape in dataset_approval_node (was silently broken)"
```

---

## Task 2.4 — Backend: dataset-preview + dataset-download endpoints

**Files:**
- Modify: `api/routers/runs.py` (existing router, no `/api` prefix per repo convention)
- Create: `tests/api/test_dataset_preview.py`

- [ ] **Step 1: Test first**

```python
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.services import run_store

@pytest.fixture
def client(tmp_path):
    csv = tmp_path / "p.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]}).to_csv(csv, index=False)
    entry = run_store.create_entry("run-1", graph_config={})
    entry.processed_dataset_path = str(csv)
    yield TestClient(app)

def test_dataset_preview_pagination(client):
    r = client.get("/runs/run-1/dataset-preview?limit=2&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 5
    assert len(body["rows"]) == 2
    assert body["rows"][0]["a"] == 1
    cols = {c["name"] for c in body["columns"]}
    assert cols == {"a", "b"}

def test_dataset_preview_404_unknown_run(client):
    assert client.get("/runs/unknown/dataset-preview").status_code == 404

def test_dataset_download_streams_csv(client):
    r = client.get("/runs/run-1/dataset-download")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert b"a,b" in r.content
```

- [ ] **Step 2: Run test, expect fail**

```
uv run pytest tests/api/test_dataset_preview.py -v
```

Expected: 404 for the route — endpoints don't exist yet.

- [ ] **Step 3: Implement endpoints in `api/routers/runs.py`**

Add to the existing router (the file already exists and is registered without an `/api` prefix in `api/main.py:18` — paths stay bare):

```python
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd

@router.get("/runs/{run_id}/dataset-preview")
def dataset_preview(run_id: str, limit: int = 50, offset: int = 0):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(404, "run not found")
    path = entry.processed_dataset_path
    if not path:
        raise HTTPException(409, "no processed dataset yet")
    df = pd.read_csv(path)
    total = len(df)
    rows = df.iloc[offset : offset + limit].to_dict(orient="records")
    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "non_null_count": int(df[c].notna().sum()),
            "sample_value": None if df[c].dropna().empty else df[c].dropna().iloc[0],
        }
        for c in df.columns
    ]
    return {"columns": columns, "rows": rows, "total_rows": total}

@router.get("/runs/{run_id}/dataset-download")
def dataset_download(run_id: str):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(404, "run not found")
    path = entry.processed_dataset_path
    if not path:
        raise HTTPException(409, "no processed dataset yet")

    def iter_file():
        with open(path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.csv"'},
    )
```

- [ ] **Step 4: Run tests, expect pass**

```
uv run pytest tests/api/test_dataset_preview.py -v
```

- [ ] **Step 5: Commit**

```
git add api/routers/runs.py tests/api/test_dataset_preview.py
git commit -m "feat(api): add /runs/{id}/dataset-preview + dataset-download endpoints"
```

---

## Task 2.5 — Update frontend types

**Files:**
- Modify: `frontend/types/api.ts`

- [ ] **Step 1: Extend `DataValidationInterrupt`**

Find `DataValidationInterrupt` and ensure `dataset_preview` has these fields (adding what's missing):

```ts
export interface DataValidationInterrupt {
  type: 'data_validation'
  attempt?: number
  question?: string
  dataset_preview: {
    path: string
    shape: [number, number]
    row_count: number
    column_count: number
    columns: Array<{ name: string; dtype: string }>
    sample_rows: Record<string, unknown>[]   // existing
    head: Record<string, unknown>[]          // new alias for clarity
    tail: Record<string, unknown>[]          // NEW (forecasting only)
  }
  validation_report?: Record<string, unknown>
}
```

- [ ] **Step 2: Commit**

```
git add frontend/types/api.ts
git commit -m "feat(types): extend DataValidationInterrupt with head/tail"
```

---

## Task 2.6 — `<DatasetApprovalCard>` — test first

**Files:**
- Create: `frontend/__tests__/components/pipeline/DatasetApprovalCard.test.tsx`

- [ ] **Step 1: Write tests**

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DatasetApprovalCard } from '@/components/pipeline/DatasetApprovalCard'

const basePreview = {
  path: 'data/processed/x.csv',
  shape: [10, 2] as [number, number],
  row_count: 10,
  column_count: 2,
  columns: [{ name: 'a', dtype: 'int64' }, { name: 'b', dtype: 'object' }],
  sample_rows: [{ a: 1, b: 'x' }],
  head: [{ a: 1, b: 'x' }],
  tail: [],
}

const baseInterrupt = {
  type: 'data_validation' as const,
  attempt: 1,
  dataset_preview: basePreview,
  validation_report: { passed: true, violations: [] },
}

describe('<DatasetApprovalCard>', () => {
  it('disables Reject button until comment >= 4 chars', () => {
    const onApprove = vi.fn()
    render(<DatasetApprovalCard runId="r" interrupt={baseInterrupt} onApprove={onApprove} isPending={false} maxAttempts={3} />)
    const reject = screen.getByRole('button', { name: /reject/i }) as HTMLButtonElement
    expect(reject.disabled).toBe(true)
    fireEvent.change(screen.getByLabelText(/comment/i), { target: { value: 'bad data' } })
    expect(reject.disabled).toBe(false)
  })

  it('hides Tail tab for non-forecasting (tail is empty)', () => {
    render(<DatasetApprovalCard runId="r" interrupt={baseInterrupt} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.queryByRole('button', { name: /^tail$/i })).not.toBeInTheDocument()
  })

  it('shows Tail tab when tail rows present', () => {
    const fc = { ...baseInterrupt, dataset_preview: { ...basePreview, tail: [{ a: 10, b: 'z' }] } }
    render(<DatasetApprovalCard runId="r" interrupt={fc} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.getByRole('button', { name: /^tail$/i })).toBeInTheDocument()
  })

  it('renders attempt indicator N of M', () => {
    render(<DatasetApprovalCard runId="r" interrupt={{ ...baseInterrupt, attempt: 2 }} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.getByText(/attempt 2 of 3/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test, expect fail**

```
cd frontend && npm test -- DatasetApprovalCard
```

Expected: module not found.

---

## Task 2.7 — Implement `<DatasetApprovalCard>`

**Files:**
- Create: `frontend/components/pipeline/DatasetApprovalCard.tsx`

- [ ] **Step 1: Implement**

```tsx
'use client'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import type { DataValidationInterrupt } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

type Tab = 'head' | 'tail' | 'schema' | 'validation'

interface Props {
  runId: string | null
  interrupt: DataValidationInterrupt
  onApprove: (decision: 'approve' | 'reject', comment: string) => void
  isPending: boolean
  maxAttempts: number
}

export function DatasetApprovalCard({ runId, interrupt, onApprove, isPending, maxAttempts }: Props) {
  const [tab, setTab] = useState<Tab>('head')
  const [comment, setComment] = useState('')
  const preview = interrupt.dataset_preview
  const hasTail = (preview.tail?.length ?? 0) > 0
  const attempt = interrupt.attempt ?? 1
  const rejectDisabled = isPending || comment.trim().length < 4

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'head', label: 'Head' },
    ...(hasTail ? [{ key: 'tail' as const, label: 'Tail' }] : []),
    { key: 'schema', label: 'Schema' },
    { key: 'validation', label: 'Validation report' },
  ]

  const rows =
    tab === 'head' ? preview.head ?? preview.sample_rows ?? []
    : tab === 'tail' ? preview.tail ?? []
    : []
  const cols = preview.columns

  return (
    <Card
      title="Dataset approval required"
      actions={
        <span className="text-xs text-[var(--color-warning)]">
          Attempt {attempt} of {maxAttempts}
        </span>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-[var(--color-fg-muted)]">
        <span className="font-mono text-[var(--color-fg)]">{preview.path.split('/').pop()}</span>
        <button
          type="button"
          className="rounded border border-[var(--color-border)] px-2 py-0.5 hover:bg-zinc-50"
          onClick={() => { navigator.clipboard.writeText(preview.path); toast.success('Path copied') }}
        >
          Copy artifact path
        </button>
        <span>{preview.row_count} rows · {preview.column_count} columns</span>
        {interrupt.validation_report && (
          <Badge variant={(interrupt.validation_report as { passed?: boolean }).passed ? 'success' : 'danger'}>
            validation {(interrupt.validation_report as { passed?: boolean }).passed ? '✓ passed' : '✗ failed'}
          </Badge>
        )}
      </div>

      <div className="mb-2 flex gap-1 border-b border-[var(--color-border)]">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 text-xs font-medium ${
              tab === t.key
                ? 'border-b-2 border-indigo-600 text-indigo-700'
                : 'text-[var(--color-fg-muted)] hover:text-[var(--color-fg)]'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {(tab === 'head' || tab === 'tail') && (
        <div className="overflow-x-auto rounded border border-[var(--color-border)]">
          <table className="w-full text-xs">
            <thead className="bg-zinc-50">
              <tr>{cols.map((c) => (<th key={c.name} className="border-b border-[var(--color-border)] px-2 py-1 text-left font-medium text-zinc-500">{c.name}<span className="ml-1 text-zinc-400">{c.dtype}</span></th>))}</tr>
            </thead>
            <tbody>
              {rows.slice(0, 5).map((row, i) => (
                <tr key={i} className="border-b border-zinc-100 last:border-0">
                  {cols.map((c) => (
                    <td key={c.name} className={`px-2 py-1 ${row[c.name] == null ? 'italic text-red-400' : 'text-zinc-700'}`}>
                      {row[c.name] == null ? 'null' : String(row[c.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'schema' && (
        <table className="w-full text-xs">
          <thead className="bg-zinc-50">
            <tr>
              <th className="px-2 py-1 text-left font-medium text-zinc-500">name</th>
              <th className="px-2 py-1 text-left font-medium text-zinc-500">dtype</th>
            </tr>
          </thead>
          <tbody>
            {cols.map((c) => (
              <tr key={c.name} className="border-t border-zinc-100">
                <td className="px-2 py-1 font-mono text-zinc-700">{c.name}</td>
                <td className="px-2 py-1 text-zinc-500">{c.dtype}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {tab === 'validation' && (
        <pre className="overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
          {JSON.stringify(interrupt.validation_report ?? {}, null, 2)}
        </pre>
      )}

      <label htmlFor="reject-comment" className="mt-4 mb-1 block text-xs font-medium text-zinc-500">
        Comment <span className="text-zinc-400">(required to reject, ≥ 4 chars)</span>
      </label>
      <textarea
        id="reject-comment"
        rows={2}
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="e.g. rename column X, drop rows where value < 0…"
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-white px-2 py-1.5 text-xs text-zinc-700 placeholder-zinc-300 focus:outline-none focus:ring-1 focus:ring-indigo-300"
      />

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onApprove('approve', '')}
          disabled={isPending}
          className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
        >
          ✓ Approve dataset
        </button>
        <button
          type="button"
          onClick={() => onApprove('reject', comment)}
          disabled={rejectDisabled}
          className="rounded border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
        >
          ✗ Reject &amp; retry
        </button>
        <a
          href={`${API_BASE}/api/runs/${runId}/dataset-download`}
          className="rounded border border-[var(--color-border)] px-3 py-1.5 text-xs font-medium text-[var(--color-fg-muted)] hover:bg-zinc-50"
        >
          Download CSV ↓
        </a>
      </div>
    </Card>
  )
}
```

- [ ] **Step 2: Run tests, expect pass**

```
cd frontend && npm test -- DatasetApprovalCard
```

- [ ] **Step 3: Commit**

```
git add frontend/components/pipeline/DatasetApprovalCard.tsx frontend/__tests__/components/pipeline/DatasetApprovalCard.test.tsx
git commit -m "feat(pipeline): add DatasetApprovalCard with Head/Tail/Schema/Validation tabs"
```

---

## Task 2.8 — Swap `DatasetApprovalCard` into `ResultsDashboard`

**Files:**
- Modify: `frontend/components/pipeline/ResultsDashboard.tsx`

- [ ] **Step 1: Remove the inline `DatasetReviewPanel` function and its usage**

Delete the entire `DatasetReviewPanel` definition (lines ~317–418). Replace the conditional render block inside the Dataset tab:

```tsx
{isDataValidationHITL && (
  <DatasetReviewPanel ... />
)}
```

with:

```tsx
{isDataValidationHITL && (
  <DatasetApprovalCard
    runId={runId}
    interrupt={interruptValue as DataValidationInterrupt}
    onApprove={(decision, comment) => approve(decision, comment)}
    isPending={isPending}
    maxAttempts={3}
  />
)}
```

Add the imports at the top:

```tsx
import { DatasetApprovalCard } from '@/components/pipeline/DatasetApprovalCard'
import { useApprove } from '@/hooks/use-approve'
```

Inside `ResultsDashboard`, after pulling `runId` from the store, call:

```tsx
const { approve, isPending } = useApprove(runId)
```

- [ ] **Step 2: Run frontend tests + smoke**

```
cd frontend && npm test
docker compose up --build
```

Submit a dataset that needs HITL approval. Verify:
- New card shows path, row/col counts, validation badge.
- Copy-path button works (toast appears).
- Reject is disabled until a 4+ char comment.
- Download CSV link returns the file.

- [ ] **Step 3: Commit**

```
git add frontend/components/pipeline/ResultsDashboard.tsx
git commit -m "refactor(pipeline): swap legacy DatasetReviewPanel for DatasetApprovalCard"
```

---

## Task 2.9 — Slice 2 smoke-check

- [ ] **Step 1: Submit a forecasting CSV end-to-end**

Use the energy_forecast example. Verify:
- HITL fires with both Head AND Tail tabs.
- Tail tab shows last 5 rows.
- Schema tab lists columns + dtypes.
- Reject with `< 4 char comment` → button disabled. With `bad temporal order` → triggers retry; attempt counter increments to "Attempt 2 of 3".

- [ ] **Step 2: Run all tests**

```
uv run pytest -m "not integration"
cd frontend && npm test
```

Both must be green.

---

# Slice 3 — Audit Report + Deployment Gate + Bug Fix

## Task 3.1 — Bug-fix regression test (RED first)

**Files:**
- Create: `tests/test_graphs/test_deployment_flow.py`

- [ ] **Step 1: Write failing test**

```python
import json
from unittest.mock import patch, MagicMock
import pytest
from langgraph.types import Command
from mlops_agents.graphs.mlops_graph import graph

@pytest.mark.integration
def test_approval_routes_to_deployer():
    """After human approves at Gate 2, the deployer node MUST run."""
    cfg = {"configurable": {"thread_id": "deploy-test"}}
    state = {
        "training_run_id": "abc123",
        "validation_passed": True,
        "dataset_approved": True,
        "training_plan": {"selected_model": "seasonal_naive"},
        "evaluation_passed": True,
        "evaluation_report_audit": {"summary": "ok"},
        "evaluation_report": {},
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "deployment_decision": "pending",
        "deployment_approved": None,
        "agent_attempt_counts": {},
    }
    # Patch at the bound location (mlops_agents.deployment.deployer), not the source module
    with patch("mlops_agents.deployment.deployer.register_model") as reg, \
         patch("mlops_agents.deployment.deployer.set_model_alias") as alias:
        reg.invoke.return_value = json.dumps({"model_name": "seasonal_naive", "version": "1"})
        alias.invoke.return_value = "ok"
        # First astream pass: stops at deployment_approval HITL
        for _ in graph.stream(state, cfg, stream_mode="updates"):
            pass
        # Resume with approval
        for _ in graph.stream(Command(resume={"approved": True, "comment": ""}), cfg, stream_mode="updates"):
            pass
        final = graph.get_state(cfg).values
    assert final["deployment_status"] == "deployed", (
        "Deployer node did not run after human approval — "
        "deployment_approval_node likely overwrote deployment_decision."
    )
```

- [ ] **Step 2: Run, expect fail**

```
uv run pytest tests/test_graphs/test_deployment_flow.py -v
```

Expected: `assert "deployed" == None` or similar — the bug is real.

---

## Task 3.2 — Fix the bug in `approval_nodes.py`

**Files:**
- Modify: `src/mlops_agents/graphs/approval_nodes.py`

- [ ] **Step 1: In `deployment_approval_node`, remove the `deployment_decision` write**

Locate the return statement currently writing `"deployment_decision": "approved" if approved else "rejected"`. Replace with a payload that ONLY touches `deployment_approved`:

```python
    return Command(
        goto="workflow_controller",
        update={"deployment_approved": approved},
    )
```

Do not write `deployment_decision` here. The deployer node already flips it to `"deployed"` after running.

- [ ] **Step 2: Re-run regression test, expect pass**

```
uv run pytest tests/test_graphs/test_deployment_flow.py -v
```

- [ ] **Step 3: Commit**

```
git add src/mlops_agents/graphs/approval_nodes.py tests/test_graphs/test_deployment_flow.py
git commit -m "fix(graph): stop overwriting deployment_decision in deployment_approval_node"
```

---

## Task 3.3 — `resolve_champion_model_name` helper — test first

**Files:**
- Create: `tests/test_evaluation/test_champion.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
from mlops_agents.evaluation.champion import resolve_champion_model_name

def test_uses_audit_when_present():
    state = {"evaluation_report_audit": {"champion_model": "lightgbm"}}
    assert resolve_champion_model_name(state) == "lightgbm"

def test_falls_back_to_champion_candidate():
    state = {"champion_candidate": {"model_key": "ets"}}
    assert resolve_champion_model_name(state) == "ets"

def test_falls_back_to_training_plan():
    state = {"training_plan": {"selected_model": "auto_arima"}}
    assert resolve_champion_model_name(state) == "auto_arima"

def test_final_fallback_truncates_run_id():
    state = {"training_run_id": "abcdef1234567890"}
    assert resolve_champion_model_name(state) == "abcdef12"

def test_returns_unknown_on_total_emptiness():
    assert resolve_champion_model_name({}) == "unknown"

def test_audit_takes_precedence_over_others():
    state = {
        "evaluation_report_audit": {"champion_model": "lightgbm"},
        "champion_candidate": {"model_key": "ets"},
        "training_plan": {"selected_model": "auto_arima"},
        "training_run_id": "abcd1234",
    }
    assert resolve_champion_model_name(state) == "lightgbm"
```

- [ ] **Step 2: Run, expect fail (module not found)**

```
uv run pytest tests/test_evaluation/test_champion.py -v
```

---

## Task 3.4 — Implement `resolve_champion_model_name`

**Files:**
- Create: `src/mlops_agents/evaluation/champion.py`

- [ ] **Step 1: Implement**

```python
"""Single source of truth for the human-readable champion model name."""
from typing import Any


def resolve_champion_model_name(state: dict[str, Any]) -> str:
    """Resolve the champion model name via a 4-step fallback chain.

    1. state["evaluation_report_audit"]["champion_model"]
    2. state["champion_candidate"]["model_key"]
    3. state["training_plan"]["selected_model"]
    4. state["training_run_id"][:8]
    """
    audit = state.get("evaluation_report_audit") or {}
    if isinstance(audit, dict) and audit.get("champion_model"):
        return str(audit["champion_model"])

    candidate = state.get("champion_candidate") or {}
    if isinstance(candidate, dict) and candidate.get("model_key"):
        return str(candidate["model_key"])

    plan = state.get("training_plan") or {}
    if isinstance(plan, dict) and plan.get("selected_model"):
        return str(plan["selected_model"])

    run_id = state.get("training_run_id") or ""
    if run_id:
        return str(run_id)[:8]

    return "unknown"
```

- [ ] **Step 2: Run tests, expect all pass**

```
uv run pytest tests/test_evaluation/test_champion.py -v
```

- [ ] **Step 3: Commit**

```
git add src/mlops_agents/evaluation/champion.py tests/test_evaluation/test_champion.py
git commit -m "feat(evaluation): add resolve_champion_model_name with 4-step fallback"
```

---

## Task 3.5 — Enrich deployer HITL payload with `deployment_action`

**Files:**
- Modify: `src/mlops_agents/graphs/approval_nodes.py`

- [ ] **Step 1: Update `deployment_approval_node` payload**

Replace the interrupt call body so the payload includes the action description and all needed audit fields:

```python
    from mlops_agents.evaluation.champion import resolve_champion_model_name
    champion = resolve_champion_model_name(state)

    approval = interrupt({
        "type": "deployer",
        "evaluation_report":       state.get("evaluation_report", {}),
        "evaluation_report_audit": state.get("evaluation_report_audit", {}),
        "candidate_metrics":       state.get("candidate_metrics", {}),
        "champion_metrics":        state.get("champion_metrics", {}),
        "thresholds_applied":      state.get("thresholds_applied", {}),
        "training_plan":           state.get("training_plan", {}),
        "candidate_run_id":        state.get("training_run_id", ""),
        "deployment_action": {
            "verb":   "register_and_promote",
            "model":  champion,
            "alias":  "champion",
            "summary": "This approval will register the candidate run as a new model version and assign it the champion alias.",
        },
    })
```

- [ ] **Step 2: Run all backend tests + the regression test**

```
uv run pytest -m "not integration"
uv run pytest tests/test_graphs/test_deployment_flow.py -v
```

- [ ] **Step 3: Commit**

```
git add src/mlops_agents/graphs/approval_nodes.py
git commit -m "feat(graph): enrich deployer HITL payload with deployment_action"
```

---

## Task 3.6 — Emit `audit_report` SSE event

**Files:**
- Modify: `api/services/pipeline.py`

- [ ] **Step 1: Inside `_stream`, in the `mode == "updates"` branch, add an audit_report emission**

After the existing `planner_context` emission block, add:

```python
                if "report_writer" in data and isinstance(data["report_writer"], dict):
                    rw = data["report_writer"]
                    audit = rw.get("evaluation_report_audit")
                    if isinstance(audit, dict) and audit:
                        from mlops_agents.evaluation.champion import resolve_champion_model_name
                        # evaluation_passed lives INSIDE the audit dict
                        # (report_writer.py copies it through from prior state)
                        evaluation_passed = bool(audit.get("evaluation_passed", True))
                        champion = resolve_champion_model_name({**rw})
                        audit_event: dict = {
                            "type": "audit_report",
                            "agent": "report_writer",
                            "timestamp_ms": time.time() * 1000,
                            "data": {
                                "audit":              audit,
                                "champion_model":     audit.get("champion_model") or champion,
                                "evaluation_passed":  evaluation_passed,
                                "candidate_metrics":  rw.get("candidate_metrics", {}),
                                "champion_metrics":   rw.get("champion_metrics", {}),
                                "thresholds_applied": rw.get("thresholds_applied", {}),
                            },
                        }
                        entry.events.append(audit_event)
                        await entry.queue.put(audit_event)
```

- [ ] **Step 2: Boot the container and confirm event arrives over SSE**

Full SSE-level emission is verified by the manual docker run in task 3.11 (slice smoke). The graph state machine is already covered by `test_deployment_flow.py` from task 3.1. We deliberately do not add a new pytest here — wrapping an async streaming generator under unittest is more fragile than the smoke check it would replace.

```
docker compose up --build api
```

Trigger any pipeline that reaches `report_writer` and confirm in the EventLog Raw tab that an event of type `audit_report` appears with `agent: "report_writer"` and a populated `data.audit` block.

- [ ] **Step 3: Commit**

```
git add api/services/pipeline.py
git commit -m "feat(api): emit audit_report SSE event after report_writer runs"
```

---

## Task 3.7 — Frontend types for audit + deployer

**Files:**
- Modify: `frontend/types/api.ts`

- [ ] **Step 1: Add types**

```ts
export interface AuditReportEventData {
  audit: {
    summary?: string
    champion_model?: string
    why_champion_won?: string
    planner_alignment?: string
    deviations_from_planner_expectations?: string[]
    evidence_consistency_warnings?: string[]
    risks_and_warnings?: string[]
    promotion_decision_explanation?: string
    human_review_notes?: string[]
  }
  champion_model: string
  evaluation_passed: boolean
  candidate_metrics: Record<string, unknown>
  champion_metrics: Record<string, unknown>
  thresholds_applied: Record<string, unknown>
}

export interface DeployerInterrupt {
  type: 'deployer'
  evaluation_report?: Record<string, unknown>
  evaluation_report_audit?: AuditReportEventData['audit']
  candidate_metrics?: Record<string, unknown>
  champion_metrics?: Record<string, unknown>
  thresholds_applied?: Record<string, unknown>
  training_plan?: Record<string, unknown>
  candidate_run_id?: string
  deployment_action?: {
    verb: string
    model: string
    alias: string
    summary: string
  }
}
```

Also add `'audit_report'` to the `PipelineEventType` union and `audit_report` color to `TYPE_COLORS` in `EventLog.tsx` (use `text-violet-600`).

- [ ] **Step 2: Commit**

```
git add frontend/types/api.ts frontend/components/pipeline/EventLog.tsx
git commit -m "feat(types): add audit_report event + DeployerInterrupt + DeploymentAction"
```

---

## Task 3.8 — `<AuditReportPanel>` — test + implement

**Files:**
- Create: `frontend/components/pipeline/AuditReportPanel.tsx`
- Create: `frontend/__tests__/components/pipeline/AuditReportPanel.test.tsx`

- [ ] **Step 1: Test**

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AuditReportPanel } from '@/components/pipeline/AuditReportPanel'

const data = {
  audit: {
    summary: 's', why_champion_won: 'w',
    risks_and_warnings: ['risk one', 'risk two'],
    human_review_notes: ['note one'],
  },
  champion_model: 'seasonal_naive',
  evaluation_passed: true,
  candidate_metrics: {}, champion_metrics: {}, thresholds_applied: {},
}

describe('<AuditReportPanel>', () => {
  it('renders champion model + risks open by default', () => {
    render(<AuditReportPanel data={data} />)
    expect(screen.getByText(/seasonal_naive/)).toBeInTheDocument()
    expect(screen.getByText(/risk one/)).toBeInTheDocument()
  })
  it('renders "candidate rejected" banner when evaluation_passed false', () => {
    render(<AuditReportPanel data={{ ...data, evaluation_passed: false }} />)
    expect(screen.getByText(/candidate rejected/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Implement**

```tsx
'use client'
import type { AuditReportEventData } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

function Section({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  return (
    <details open={defaultOpen} className="border-t border-[var(--color-border)] py-2 [&_summary]:cursor-pointer">
      <summary className="text-xs font-semibold text-[var(--color-fg)]">{title}</summary>
      <div className="mt-1 text-xs text-zinc-600">{children}</div>
    </details>
  )
}

export function AuditReportPanel({ data }: { data: AuditReportEventData }) {
  const a = data.audit
  return (
    <Card
      title="Audit Report"
      actions={data.evaluation_passed
        ? <Badge variant="success">eligible</Badge>
        : <Badge variant="info">candidate rejected</Badge>}
    >
      <dl className="mb-3 grid grid-cols-[160px_1fr] gap-y-1 text-xs">
        <dt className="text-zinc-500">Champion model</dt>
        <dd className="font-mono text-zinc-800">{data.champion_model}</dd>
        <dt className="text-zinc-500">Deterministic eval</dt>
        <dd>{data.evaluation_passed
          ? <span className="text-emerald-700">✓ passed</span>
          : <span className="text-sky-700">candidate rejected</span>}
        </dd>
      </dl>

      {a.summary && <Section title="Summary">{a.summary}</Section>}
      {a.why_champion_won && <Section title="Why this model won">{a.why_champion_won}</Section>}
      {a.planner_alignment && <Section title="Planner alignment">{a.planner_alignment}</Section>}
      {(a.deviations_from_planner_expectations?.length ?? 0) > 0 && (
        <Section title="Deviations from planner expectations">
          <ul className="list-disc pl-4">{a.deviations_from_planner_expectations!.map((d, i) => <li key={i}>{d}</li>)}</ul>
        </Section>
      )}
      {(a.evidence_consistency_warnings?.length ?? 0) > 0 && (
        <Section title="Evidence consistency warnings">
          <ul className="list-disc pl-4">{a.evidence_consistency_warnings!.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </Section>
      )}
      {(a.risks_and_warnings?.length ?? 0) > 0 && (
        <Section title="Risks & warnings" defaultOpen>
          <ul className="space-y-1">{a.risks_and_warnings!.map((r, i) => (
            <li key={i} className="rounded bg-amber-50 px-2 py-1 text-amber-700">⚠ {r}</li>
          ))}</ul>
        </Section>
      )}
      {(a.human_review_notes?.length ?? 0) > 0 && (
        <Section title="Human review notes">
          <ul className="list-disc pl-4">{a.human_review_notes!.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </Section>
      )}

      <details className="mt-3 border-t border-[var(--color-border)] pt-2">
        <summary className="cursor-pointer text-xs text-zinc-500">View full audit JSON</summary>
        <pre className="mt-1 overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
          {JSON.stringify(data, null, 2)}
        </pre>
      </details>
    </Card>
  )
}
```

- [ ] **Step 3: Run tests, expect pass + commit**

```
cd frontend && npm test -- AuditReportPanel
git add frontend/components/pipeline/AuditReportPanel.tsx frontend/__tests__/components/pipeline/AuditReportPanel.test.tsx
git commit -m "feat(pipeline): add AuditReportPanel with collapsible sections"
```

---

## Task 3.9 — `<DeploymentApprovalCard>` — test + implement

**Files:**
- Create: `frontend/components/pipeline/DeploymentApprovalCard.tsx`
- Create: `frontend/__tests__/components/pipeline/DeploymentApprovalCard.test.tsx`

- [ ] **Step 1: Test**

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DeploymentApprovalCard } from '@/components/pipeline/DeploymentApprovalCard'

const interrupt = {
  type: 'deployer' as const,
  candidate_run_id: 'c5e7fe98abcd',
  deployment_action: {
    verb: 'register_and_promote',
    model: 'seasonal_naive',
    alias: 'champion',
    summary: 'register + promote',
  },
  evaluation_report_audit: { risks_and_warnings: ['risk a', 'risk b', 'risk c', 'risk d'] },
}

describe('<DeploymentApprovalCard>', () => {
  it('renders champion model name verbatim from payload', () => {
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={vi.fn()} isPending={false} />)
    expect(screen.getByText(/seasonal_naive/)).toBeInTheDocument()
  })
  it('renders top 3 risks only', () => {
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={vi.fn()} isPending={false} />)
    expect(screen.getByText(/risk a/)).toBeInTheDocument()
    expect(screen.queryByText(/risk d/)).not.toBeInTheDocument()
  })
  it('fires onApprove with approve', () => {
    const fn = vi.fn()
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={fn} isPending={false} />)
    fireEvent.click(screen.getByRole('button', { name: /approve/i }))
    expect(fn).toHaveBeenCalledWith('approve')
  })
})
```

- [ ] **Step 2: Implement**

```tsx
'use client'
import type { DeployerInterrupt } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

interface Props {
  runId: string | null
  interrupt: DeployerInterrupt
  onApprove: (decision: 'approve' | 'reject') => void
  isPending: boolean
}

export function DeploymentApprovalCard({ runId, interrupt, onApprove, isPending }: Props) {
  const action = interrupt.deployment_action
  const audit = interrupt.evaluation_report_audit ?? {}
  const risks = (audit.risks_and_warnings ?? []).slice(0, 3)
  const candidateShort = (interrupt.candidate_run_id ?? '').slice(0, 8)

  return (
    <Card
      title={action ? `Approve deployment: ${action.model}` : 'Deployment approval required'}
      actions={<Badge variant="warning">awaiting human</Badge>}
    >
      <dl className="mb-3 grid grid-cols-[140px_1fr] gap-y-1 text-xs">
        <dt className="text-zinc-500">Candidate run</dt>
        <dd className="font-mono text-zinc-800">{candidateShort}…</dd>
        <dt className="text-zinc-500">Promotion</dt>
        <dd>eligible — passed deterministic thresholds</dd>
      </dl>

      {action && (
        <div className="mb-3 rounded border border-indigo-200 bg-indigo-50 p-2 text-xs">
          <p className="font-semibold text-indigo-800">Deployment action</p>
          <p className="text-indigo-700">{action.summary}</p>
          <p className="mt-1 text-indigo-600">
            This approval will promote <span className="font-mono">{action.model}</span> as <span className="font-mono">{action.alias}</span>.
          </p>
        </div>
      )}

      {risks.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-xs font-semibold text-zinc-600">Top risks from audit</p>
          <ul className="space-y-1 text-xs">
            {risks.map((r, i) => (
              <li key={i} className="rounded bg-amber-50 px-2 py-1 text-amber-700">⚠ {r}</li>
            ))}
          </ul>
          <p className="mt-1 text-[11px] text-zinc-400">See full audit in the Audit tab above.</p>
        </div>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onApprove('approve')}
          disabled={isPending}
          className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          Approve deployment
        </button>
        <button
          type="button"
          onClick={() => onApprove('reject')}
          disabled={isPending}
          className="rounded border border-zinc-300 bg-white px-3 py-1.5 text-xs font-semibold text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
        >
          Reject deployment
        </button>
      </div>

      <details className="mt-3 border-t border-[var(--color-border)] pt-2">
        <summary className="cursor-pointer text-xs text-zinc-500">Raw payload</summary>
        <pre className="mt-1 overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
          {JSON.stringify(interrupt, null, 2)}
        </pre>
      </details>
    </Card>
  )
}
```

- [ ] **Step 3: Run tests + commit**

```
cd frontend && npm test -- DeploymentApprovalCard
git add frontend/components/pipeline/DeploymentApprovalCard.tsx frontend/__tests__/components/pipeline/DeploymentApprovalCard.test.tsx
git commit -m "feat(pipeline): add DeploymentApprovalCard replacing raw JSON dump"
```

---

## Task 3.10 — Wire Audit tab + auto-switch + replace HITLGate dump

**Files:**
- Modify: `frontend/components/pipeline/ResultsDashboard.tsx`
- Modify: `frontend/components/pipeline/HITLGate.tsx`

- [ ] **Step 1: Add Audit tab to ResultsDashboard tabs array**

In `ResultsDashboard`, extend the tabs array:

```tsx
const tabs = [
  { key: 'dataset',  label: 'Dataset', ready: hasDataset },
  { key: 'planner',  label: 'Planner', ready: hasPlanner },
  { key: 'model',    label: 'Model',   ready: hasModel },
  { key: 'audit',    label: 'Audit',   ready: !!auditData },
] as const
```

Where `auditData` is computed:

```tsx
const auditData = useMemo(() => {
  const ev = events.findLast((e) => e.type === 'audit_report')
  return ev ? (ev.data as unknown as AuditReportEventData) : null
}, [events])
```

Render the Audit tab body:

```tsx
{tab === 'audit' && auditData && <AuditReportPanel data={auditData} />}
{tab === 'audit' && !auditData && <p className="text-xs text-zinc-400">Audit not yet generated.</p>}
```

- [ ] **Step 2: Implement auto-switch with pinning**

Replace the existing `isDataValidationHITL` `useEffect` with this logic. Add state:

```tsx
const [pinnedTab, setPinnedTab] = useState(false)

function selectTab(key: typeof tab) { setTab(key); setPinnedTab(true) }
```

Replace tab onClick: `onClick={() => selectTab(key)}`.

Add per-event-type effects that respect the pin:

```tsx
useEffect(() => {
  if (pinnedTab) return
  if (isDataValidationHITL) setTab('dataset')
}, [isDataValidationHITL, pinnedTab])

useEffect(() => {
  if (pinnedTab) return
  if (plannerCtx) setTab('planner')
}, [plannerCtx, pinnedTab])

useEffect(() => {
  if (pinnedTab) return
  if (trained || tuned) setTab('model')
}, [trained, tuned, pinnedTab])

useEffect(() => {
  if (pinnedTab) return
  if (auditData) setTab('audit')
}, [auditData, pinnedTab])

useEffect(() => {
  // Reset pin when a new run begins (runId changes)
  setPinnedTab(false)
}, [runId])
```

- [ ] **Step 3: Replace HITLGate body**

```tsx
'use client'
import { useRunStore } from '@/stores/run-store'
import { useApprove } from '@/hooks/use-approve'
import { DeploymentApprovalCard } from '@/components/pipeline/DeploymentApprovalCard'
import type { DeployerInterrupt } from '@/types/api'

export function HITLGate({ runId }: { runId: string | null }) {
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const { approve, isPending } = useApprove(runId)

  if (!hitlPending) return null
  if ((interruptValue as { type?: string })?.type !== 'deployer') return null

  return (
    <DeploymentApprovalCard
      runId={runId}
      interrupt={interruptValue as unknown as DeployerInterrupt}
      onApprove={(decision) => approve(decision)}
      isPending={isPending}
    />
  )
}
```

- [ ] **Step 4: Update `HITLGate.test.tsx` to match new shape**

Read the existing test; update assertions to look for the new card title (`Approve deployment:` or `Deployment approval required`) instead of raw JSON.

- [ ] **Step 5: Smoke test + commit**

```
docker compose up --build
```

Run end-to-end. At Gate 2 verify the new card renders, then approve. Verify deployer runs and the run completes.

```
git add frontend/components/pipeline/ResultsDashboard.tsx frontend/components/pipeline/HITLGate.tsx frontend/__tests__/components/pipeline/HITLGate.test.tsx
git commit -m "feat(pipeline): Audit tab + tab pinning + DeploymentApprovalCard replaces JSON dump"
```

---

## Task 3.11 — Slice 3 smoke-check

- [ ] **Step 1: Full end-to-end with happy path**

Run a real pipeline through both gates. Verify:
- Audit tab appears after `report_writer` runs; auto-focuses if not pinned.
- Audit shows champion model, sections collapsed except risks.
- Deployment gate shows champion name + action description + top 3 risks (no raw JSON by default).
- Approving runs the deployer; MLflow shows the new model version with champion alias.

- [ ] **Step 2: Eval-rejection smoke**

Force evaluator to reject (e.g., feed a deliberately bad model via training plan override, or temporarily lower thresholds). Verify:
- Run header shows `Candidate rejected` (sky pill, NOT red).
- Stepper marks `Deploy Approval` and `Deploy` as `skipped`.
- Audit tab auto-focuses with the rejection details.

- [ ] **Step 3: Tests green**

```
uv run pytest -m "not integration"
cd frontend && npm test
```

---

# Slice 4 — Event Log Redesign

## Task 4.1 — Aggregation helpers — test first

**Files:**
- Create: `frontend/__tests__/lib/events-aggregate.test.ts`

- [ ] **Step 1: Write tests**

```ts
import { describe, it, expect } from 'vitest'
import { aggregateToolUsage, aggregateLlmNodeActivity } from '@/lib/events-aggregate'
import type { PipelineEvent } from '@/types/api'

function ev(type: string, agent: string, data: Record<string, unknown> = {}, ms = 0): PipelineEvent {
  return { type, agent, timestamp_ms: ms, data } as PipelineEvent
}

describe('aggregateToolUsage', () => {
  it('groups by agent + tool name with call counts and total ms', () => {
    const events = [
      ev('tool_call',   'data_validator', { tool_name: 'load_dataset' }, 0),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 200 }, 200),
      ev('tool_call',   'data_validator', { tool_name: 'load_dataset' }, 300),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 216 }, 516),
    ]
    const rows = aggregateToolUsage(events)
    expect(rows).toEqual([
      { agent: 'data_validator', tool_name: 'load_dataset', calls: 2, total_ms: 416 },
    ])
  })
})

describe('aggregateLlmNodeActivity', () => {
  it('measures duration between routing-in and next routing-out', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }, 1000),
      ev('routing', 'controller', { next: 'executor' }, 24400),
    ]
    const rows = aggregateLlmNodeActivity(events, ['planner'])
    expect(rows[0]).toMatchObject({ node: 'planner', activations: 1, total_ms: 23400 })
  })
})
```

- [ ] **Step 2: Run, expect fail**

```
cd frontend && npm test -- events-aggregate
```

---

## Task 4.2 — Implement helpers

**Files:**
- Create: `frontend/lib/events-aggregate.ts`

- [ ] **Step 1: Implement**

```ts
import type { PipelineEvent } from '@/types/api'

export interface ToolUsageRow {
  agent: string
  tool_name: string
  calls: number
  total_ms: number
}

export function aggregateToolUsage(events: PipelineEvent[]): ToolUsageRow[] {
  const map = new Map<string, ToolUsageRow>()
  for (const e of events) {
    if (e.type !== 'tool_result') continue
    const tool = (e.data as { tool_name?: string }).tool_name
    if (!tool) continue
    const key = `${e.agent}::${tool}`
    const dur = Number((e.data as { duration_ms?: number }).duration_ms ?? 0)
    const prev = map.get(key)
    if (prev) { prev.calls += 1; prev.total_ms += dur }
    else map.set(key, { agent: e.agent, tool_name: tool, calls: 1, total_ms: dur })
  }
  return Array.from(map.values())
}

export interface LlmNodeRow {
  node: string
  activations: number
  total_ms: number
}

export function aggregateLlmNodeActivity(events: PipelineEvent[], llmNodes: string[]): LlmNodeRow[] {
  const set = new Set(llmNodes)
  const map = new Map<string, LlmNodeRow>()
  let activeNode: string | null = null
  let activeStartMs = 0
  for (const e of events) {
    if (e.type !== 'routing') continue
    const next = (e.data as { next?: string }).next ?? ''
    if (activeNode && set.has(activeNode)) {
      const dur = e.timestamp_ms - activeStartMs
      const prev = map.get(activeNode)
      if (prev) { prev.activations += 1; prev.total_ms += dur }
      else map.set(activeNode, { node: activeNode, activations: 1, total_ms: dur })
    }
    activeNode = next
    activeStartMs = e.timestamp_ms
  }
  return Array.from(map.values())
}

export interface TimelineRow {
  ts: number
  text: string
  agent?: string
}

export function buildTimeline(events: PipelineEvent[]): TimelineRow[] {
  const rows: TimelineRow[] = []
  let lastRoutingNext = ''
  for (const e of events) {
    const t = e.timestamp_ms
    switch (e.type) {
      case 'run_info': {
        const models = Object.keys((e.data as { models?: Record<string, string> }).models ?? {})
        rows.push({ ts: t, text: `Pipeline started · LLM nodes: ${models.join(', ') || '—'}` })
        break
      }
      case 'routing': {
        const next = (e.data as { next?: string }).next ?? ''
        if (next && next !== lastRoutingNext) {
          rows.push({ ts: t, text: `Workflow moved to ${next}`, agent: 'controller' })
          lastRoutingNext = next
        }
        break
      }
      case 'tool_result': {
        const tn = (e.data as { tool_name?: string }).tool_name
        if (tn === 'load_dataset') {
          const r = (e.data as { result?: string }).result
          let summary = 'Dataset loaded'
          try { const j = JSON.parse(r ?? '{}'); summary = `Dataset loaded · ${j.row_count} rows × ${j.column_names?.length ?? '?'} cols` } catch {}
          rows.push({ ts: t, text: summary, agent: e.agent })
        }
        if (tn === 'validate_against_schema') {
          rows.push({ ts: t, text: 'Validation completed', agent: e.agent })
        }
        if (tn === 'train_model') {
          rows.push({ ts: t, text: 'Training completed', agent: e.agent })
        }
        break
      }
      case 'planner_context': {
        const cands = (e.data as { plan_summary?: { candidate_models?: string[] } }).plan_summary?.candidate_models ?? []
        rows.push({ ts: t, text: `Planner selected ${cands.length} candidates`, agent: 'planner' })
        break
      }
      case 'hitl_request': {
        const gate = (e.data as { type?: string }).type ?? ''
        rows.push({ ts: t, text: `${gate} approval requested`, agent: e.agent })
        break
      }
      case 'audit_report': {
        rows.push({ ts: t, text: 'Audit report generated', agent: 'report_writer' })
        break
      }
      case 'run_complete': {
        const err = (e.data as { error?: string }).error
        rows.push({ ts: t, text: err ? `Run failed: ${err.slice(0, 80)}` : 'Run complete', agent: e.agent })
        break
      }
    }
  }
  return rows
}
```

- [ ] **Step 2: Run tests, expect pass + commit**

```
cd frontend && npm test -- events-aggregate
git add frontend/lib/events-aggregate.ts frontend/__tests__/lib/events-aggregate.test.ts
git commit -m "feat(lib): add events aggregation helpers (tool usage, LLM nodes, timeline)"
```

---

## Task 4.3 — Refactor `<EventLog>` into 3 tabs

**Files:**
- Modify: `frontend/components/pipeline/EventLog.tsx`

- [ ] **Step 1: Replace the body**

```tsx
'use client'
import { useMemo, useState } from 'react'
import { useRunStore } from '@/stores/run-store'
import { Card } from '@/components/ui/Card'
import { displayAgentName } from '@/lib/agent-display'
import { aggregateToolUsage, aggregateLlmNodeActivity, buildTimeline } from '@/lib/events-aggregate'
import type { PipelineEvent, PipelineEventType } from '@/types/api'

type Tab = 'timeline' | 'tools' | 'raw'

const TYPE_COLORS: Record<PipelineEventType, string> = {
  run_info: 'text-zinc-400',
  routing: 'text-indigo-600',
  tool_call: 'text-zinc-500',
  tool_result: 'text-zinc-400',
  agent_reasoning: 'text-violet-500',
  planner_context: 'text-violet-600',
  hitl_request: 'font-semibold text-amber-600',
  audit_report: 'text-violet-600',
  run_complete: 'font-semibold text-emerald-600',
}

export function EventLog() {
  const events = useRunStore((s) => s.events)
  const runId = useRunStore((s) => s.runId)
  const [tab, setTab] = useState<Tab>('timeline')

  const timeline = useMemo(() => buildTimeline(events), [events])
  const tools = useMemo(() => aggregateToolUsage(events), [events])
  const llmNodes = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])
  const llmRows = useMemo(() => aggregateLlmNodeActivity(events, llmNodes), [events, llmNodes])

  function downloadRaw() {
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `run-${runId ?? 'trace'}.json`
    document.body.appendChild(a); a.click(); a.remove()
    URL.revokeObjectURL(url)
  }

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'timeline', label: 'Timeline' },
    { key: 'tools',    label: 'Tool Details' },
    { key: 'raw',      label: 'Raw Logs' },
  ]

  return (
    <Card
      title="Events"
      actions={tab === 'raw'
        ? <button onClick={downloadRaw} className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[11px] hover:bg-zinc-50">↓ Download raw trace JSON</button>
        : null}
    >
      <div className="mb-2 flex gap-1 border-b border-[var(--color-border)]">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 text-xs font-medium ${
              tab === t.key
                ? 'border-b-2 border-indigo-600 text-indigo-700'
                : 'text-zinc-500 hover:text-zinc-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="max-h-[60vh] overflow-y-auto">
        {tab === 'timeline' && (
          <ul className="space-y-0.5 text-xs">
            {timeline.length === 0 && <li className="text-zinc-400">Waiting for events…</li>}
            {timeline.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="shrink-0 font-mono text-zinc-300">
                  {new Date(r.ts).toISOString().slice(11, 19)}
                </span>
                <span className="text-zinc-700">{r.text}</span>
              </li>
            ))}
          </ul>
        )}

        {tab === 'tools' && (
          <div className="space-y-3 text-xs">
            {tools.length > 0 && (
              <div>
                <p className="mb-1 font-semibold text-zinc-600">Tool calls (agentic loops)</p>
                {tools.map((t) => (
                  <div key={`${t.agent}-${t.tool_name}`} className="grid grid-cols-[1fr_60px_80px] gap-2 border-b border-zinc-100 py-1">
                    <span><span className="text-zinc-400">{displayAgentName(t.agent)}</span> · <span className="font-mono text-zinc-700">{t.tool_name}</span></span>
                    <span className="text-right text-zinc-500">{t.calls} call{t.calls === 1 ? '' : 's'}</span>
                    <span className="text-right font-mono text-zinc-500">{t.total_ms} ms</span>
                  </div>
                ))}
              </div>
            )}
            {llmRows.length > 0 && (
              <div>
                <p className="mb-1 font-semibold text-zinc-600">LLM nodes (single-shot)</p>
                {llmRows.map((r) => (
                  <div key={r.node} className="grid grid-cols-[1fr_80px_80px] gap-2 border-b border-zinc-100 py-1">
                    <span className="font-mono text-zinc-700">{displayAgentName(r.node)}</span>
                    <span className="text-right text-zinc-500">{r.activations} activation{r.activations === 1 ? '' : 's'}</span>
                    <span className="text-right font-mono text-zinc-500">{(r.total_ms / 1000).toFixed(1)} s</span>
                  </div>
                ))}
              </div>
            )}
            {tools.length === 0 && llmRows.length === 0 && <p className="text-zinc-400">No activity yet.</p>}
          </div>
        )}

        {tab === 'raw' && (
          <div className="font-mono text-[11px]">
            {events.length === 0 && <p className="text-zinc-400">Waiting for events…</p>}
            {events.map((e: PipelineEvent, i) => (
              <div key={i} className="mb-0.5 flex gap-2">
                <span className="shrink-0 text-zinc-300">{new Date(e.timestamp_ms).toISOString().slice(11, 23)}</span>
                <span className={`shrink-0 ${TYPE_COLORS[e.type] ?? 'text-zinc-500'}`}>{e.type}</span>
                <span className="shrink-0 text-zinc-500">{displayAgentName(e.agent)}</span>
                <span className="min-w-0 break-words text-zinc-600">
                  {(() => {
                    const t = e.type
                    if (t === 'routing') return `→ ${(e.data as { next?: string }).next ?? ''}`
                    if (t === 'tool_call' || t === 'tool_result') return String((e.data as { tool_name?: string }).tool_name ?? '')
                    if (t === 'run_complete') {
                      const err = (e.data as { error?: string }).error
                      return err ? `Error: ${err.slice(0, 80)}` : 'Done'
                    }
                    return ''
                  })()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  )
}
```

- [ ] **Step 2: Update existing `EventLog.test.tsx` for new shape**

The old test asserts on the raw line shape. Update it to:
- Verify the three tab buttons render (`Timeline`, `Tool Details`, `Raw Logs`).
- Verify Timeline shows a `Workflow moved to data_validator` line after a synthetic routing event.
- Verify Download button appears only on Raw Logs tab.

- [ ] **Step 3: Run tests + commit**

```
cd frontend && npm test -- EventLog
git add frontend/components/pipeline/EventLog.tsx frontend/__tests__/components/pipeline/EventLog.test.tsx
git commit -m "feat(pipeline): tabbed EventLog (Timeline/Tool Details/Raw) + raw trace download"
```

---

## Task 4.4 — Slice 4 smoke-check

- [ ] **Step 1: Run a real pipeline**

Verify:
- Timeline shows ~10 high-level lines (no `tool_call` spam).
- Tool Details splits into "Tool calls" + "LLM nodes".
- Raw Logs has the full event stream + Download button.
- Downloading produces a valid `run-<id>.json`.

- [ ] **Step 2: Tests green**

```
cd frontend && npm test
uv run pytest -m "not integration"
```

---

# Slice 5 — Observability + Experiments Polish

## Task 5.1 — Backend: `GET /runs?limit=`

**Files:**
- Modify: `api/services/run_store.py`
- Modify: `api/routers/runs.py` (bare path — no `/api` prefix)
- Create: `tests/api/test_runs_list.py`

- [ ] **Step 1: Add `list_entries` to `run_store`**

```python
def list_entries(limit: int = 20) -> list[RunEntry]:
    items = list(_store.values())
    items.sort(key=lambda e: getattr(e, "started_at_ms", 0), reverse=True)
    return items[:limit]
```

Also add `started_at_ms: int = 0` to `RunEntry`, and set it inside `create_entry` to `int(time.time() * 1000)`.

- [ ] **Step 2: Test first**

```python
from fastapi.testclient import TestClient
from api.main import app
from api.services import run_store

def test_runs_list_returns_recent(monkeypatch):
    run_store._store.clear()
    e1 = run_store.create_entry("a", {})
    e1.status = "complete"
    e2 = run_store.create_entry("b", {})
    e2.status = "running"
    r = TestClient(app).get("/runs?limit=10")
    assert r.status_code == 200
    body = r.json()
    ids = [x["run_id"] for x in body]
    assert set(ids) == {"a", "b"}
```

- [ ] **Step 3: Implement endpoint in `api/routers/runs.py`**

Append to the existing router file:

```python
@router.get("/runs")
def list_runs(limit: int = 20):
    out = []
    for e in run_store.list_entries(limit=limit):
        out.append({
            "run_id": e.run_id,
            "status": e.status,
            "started_at_ms": getattr(e, "started_at_ms", 0),
        })
    return out
```

- [ ] **Step 4: Run tests + commit**

```
uv run pytest tests/api/test_runs_list.py -v
git add api/services/run_store.py api/routers/runs.py tests/api/test_runs_list.py
git commit -m "feat(api): add GET /runs?limit= listing + started_at_ms on RunEntry"
```

---

## Task 5.2 — Frontend: `fetchRunsList`

**Files:**
- Modify: `frontend/lib/api.ts`

- [ ] **Step 1: Add function**

```ts
export interface RunSummary {
  run_id: string
  status: 'running' | 'awaiting_approval' | 'complete' | 'failed'
  started_at_ms: number
}

export async function fetchRunsList(limit = 20): Promise<RunSummary[]> {
  const r = await fetch(`${BASE}/runs?limit=${limit}`)
  if (!r.ok) throw new Error('runs list')
  return r.json()
}
```

(Reuse the existing `BASE` constant in the file — same one used by `fetchRunStatus`.)

- [ ] **Step 2: Commit**

```
git add frontend/lib/api.ts
git commit -m "feat(api-client): add fetchRunsList"
```

---

## Task 5.3 — Three Observability cards

**Files:**
- Create: `frontend/components/observability/PipelineHealthCard.tsx`
- Create: `frontend/components/observability/LlmActivityCard.tsx`
- Create: `frontend/components/observability/ToolUsageCard.tsx`

- [ ] **Step 1: PipelineHealthCard**

```tsx
'use client'
import { useQuery } from '@tanstack/react-query'
import { fetchRunsList } from '@/lib/api'
import { Card } from '@/components/ui/Card'

export function PipelineHealthCard() {
  const { data: runs = [] } = useQuery({ queryKey: ['runs-list'], queryFn: () => fetchRunsList(20), refetchInterval: 10_000 })
  const successful = runs.filter((r) => r.status === 'complete').length
  const failed = runs.filter((r) => r.status === 'failed').length
  const awaiting = runs.filter((r) => r.status === 'awaiting_approval').length

  return (
    <Card title="Pipeline health (last 20 runs since server start)">
      <p className="text-sm text-zinc-700">
        {successful} successful · {failed} failed · {awaiting} awaiting human
      </p>
      <p className="mt-1 text-[11px] text-zinc-400">resets on container restart</p>
    </Card>
  )
}
```

- [ ] **Step 2: LlmActivityCard**

```tsx
'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateLlmNodeActivity } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { Card } from '@/components/ui/Card'

export function LlmActivityCard() {
  const events = useRunStore((s) => s.events)
  const llmNodes = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])
  const rows = useMemo(() => aggregateLlmNodeActivity(events, llmNodes), [events, llmNodes])

  return (
    <Card title="LLM activity (current run)">
      {rows.length === 0
        ? <p className="text-xs text-zinc-400">No LLM activity yet.</p>
        : (
          <table className="w-full text-xs">
            <thead><tr className="text-left text-zinc-500">
              <th className="py-1">Node</th><th>Activations</th><th>Duration</th><th>Tokens</th><th>Status</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.node} className="border-t border-zinc-100">
                  <td className="py-1 font-mono text-zinc-700">{displayAgentName(r.node)}</td>
                  <td>{r.activations}</td>
                  <td className="font-mono">{(r.total_ms / 1000).toFixed(1)} s</td>
                  <td className="text-zinc-400">—</td>
                  <td className="text-emerald-700">ok</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      <p className="mt-2 text-[11px] text-zinc-400">Token counts shown when available — none recorded in current event stream.</p>
    </Card>
  )
}
```

- [ ] **Step 3: ToolUsageCard**

```tsx
'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateToolUsage } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { Card } from '@/components/ui/Card'

export function ToolUsageCard() {
  const events = useRunStore((s) => s.events)
  const rows = useMemo(() => aggregateToolUsage(events), [events])
  if (rows.length === 0) {
    return <Card title="Tool usage (current run)"><p className="text-xs text-zinc-400">No tool calls yet.</p></Card>
  }
  return (
    <Card title="Tool usage (current run)">
      <table className="w-full text-xs">
        <thead><tr className="text-left text-zinc-500"><th>Tool</th><th>Agent</th><th>Calls</th><th>Total ms</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.agent}-${r.tool_name}`} className="border-t border-zinc-100">
              <td className="py-1 font-mono text-zinc-700">{r.tool_name}</td>
              <td className="text-zinc-500">{displayAgentName(r.agent)}</td>
              <td>{r.calls}</td>
              <td className="font-mono">{r.total_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
```

- [ ] **Step 4: Commit**

```
git add frontend/components/observability/
git commit -m "feat(observability): add PipelineHealth, LlmActivity, ToolUsage cards"
```

---

## Task 5.4 — Wire `/observability` page

**Files:**
- Modify: `frontend/app/observability/page.tsx`

- [ ] **Step 1: Replace placeholder with real content**

```tsx
'use client'
import { PipelineHealthCard } from '@/components/observability/PipelineHealthCard'
import { LlmActivityCard } from '@/components/observability/LlmActivityCard'
import { ToolUsageCard } from '@/components/observability/ToolUsageCard'

export default function ObservabilityPage() {
  return (
    <div className="space-y-3 p-3">
      <PipelineHealthCard />
      <LlmActivityCard />
      <ToolUsageCard />
    </div>
  )
}
```

- [ ] **Step 2: Smoke-check + commit**

```
docker compose up --build
```

Navigate to `/observability`. After a run, verify all three cards show data; `started_at_ms` order is most recent first; token column shows "—".

```
git add frontend/app/observability/page.tsx
git commit -m "feat(observability): real /observability page with 3 cards"
```

---

## Task 5.5 — Experiments page polish

**Files:**
- Modify: `frontend/components/experiments/RunSidebar.tsx`
- Modify: `frontend/components/experiments/ChartPanel.tsx`

- [ ] **Step 1: In `RunSidebar`, add champion badge and problem-type subtitle**

For each run row, render the existing run name, then below it a small `<span className="text-[10px] text-zinc-400">{run.problem_type}</span>`. If the run is the current champion (`run.is_champion` or similar — check the API shape; if no flag, skip this part), render `<Badge variant="success">Champion</Badge>` to the right of the run name.

Wrap the whole sidebar in a `<Card>` (no title; pass `className="h-full"`).

- [ ] **Step 2: In `ChartPanel`, wrap the chart-area outer div in `<Card>`**

Pass `title="Metrics"` so the section has a consistent header bar.

- [ ] **Step 3: Smoke + commit**

```
docker compose up --build
```

Visit `/experiments`. Verify zinc/indigo style, the sidebar reads cleanly, and the chart panel has a header bar.

```
git add frontend/components/experiments/
git commit -m "feat(experiments): adopt Card primitive + champion badge + problem-type subtitle"
```

---

## Task 5.6 — Final slice smoke + green-tests gate

- [ ] **Step 1: Full test suite**

```
uv run pytest -m "not integration"
cd frontend && npm test
```

Both must pass.

- [ ] **Step 2: Final end-to-end smoke**

```
docker compose down -v
docker compose up --build
```

Run TWO pipelines back-to-back (so Pipeline health has multiple rows). For each:
- Verify all 4 tabs render correctly.
- Verify the audit-report flow, both happy and rejected-candidate.
- Verify `/observability` updates as runs progress.
- Verify `/experiments` polished look.
- Verify raw trace JSON download works.

- [ ] **Step 3: Tag the refactor complete**

```
git tag refactor-frontend-mlops-v1
```

(no push — tag is local until the user pushes manually)

---

## Closing notes for the engineer

- **TDD discipline**: every task above pairs a test with implementation. Do not skip the failing-test step — it's how you know the test actually exercises the behavior.
- **Token vocabulary**: after slice 1 only the listed tokens are valid. If you find yourself reaching for `slate-*` or `navy-*` mid-implementation, stop and re-check task 1.11's mapping.
- **Single source of truth**: champion model name is resolved in Python by `resolve_champion_model_name`. Frontend never re-derives it. If you need it client-side, read the pre-resolved `champion_model` field from the `audit_report` event or `deployment_action.model` from the deployer HITL payload.
- **Bug-fix-first ordering**: task 3.1 writes a regression test that should fail before task 3.2 lands. If you skip the RED step, you lose the proof that the fix works.
- **Slice boundaries are real**: a slice ends with green tests and a smoke check. You can ship after any slice. If you absolutely must roll forward into a later slice mid-PR, document why in the commit message.
