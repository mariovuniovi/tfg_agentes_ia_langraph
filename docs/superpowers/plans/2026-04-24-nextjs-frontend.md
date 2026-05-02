# Next.js Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a typed Next.js 15 frontend that consumes the FastAPI MLOps backend via REST and WebSocket, replacing the Streamlit dashboard.

**Architecture:** App Router with three pages (Pipeline, Experiments, Monitoring). Live run state (WebSocket events, HITL) is owned by a Zustand store; cached server data (MLflow, drift reports) is managed by TanStack Query v5. Each page's components live in their own `components/<domain>/` folder.

**Tech Stack:** Next.js 15, TypeScript strict, Tailwind CSS, Zustand, TanStack Query v5, Recharts, Vitest + React Testing Library.

---

## File Map

| File | Responsibility |
|---|---|
| `frontend/types/api.ts` | TypeScript mirrors of all FastAPI Pydantic models |
| `frontend/lib/api.ts` | Typed fetch wrappers for every REST endpoint |
| `frontend/lib/query-client.ts` | TanStack QueryClient singleton |
| `frontend/stores/run-store.ts` | Zustand: live run state (events, status, HITL) |
| `frontend/hooks/use-run-stream.ts` | WebSocket lifecycle — writes to run-store |
| `frontend/hooks/use-approve.ts` | TanStack mutation for POST /runs/{id}/approve |
| `frontend/components/Providers.tsx` | QueryClientProvider wrapper (client component) |
| `frontend/components/TopNav.tsx` | Top tab bar + health badge |
| `frontend/components/pipeline/TriggerPanel.tsx` | Dataset input + Run button |
| `frontend/components/pipeline/RunStatusBadge.tsx` | Status pill from Zustand |
| `frontend/components/pipeline/HITLGate.tsx` | Approve/Reject panel (renders when hitlPending) |
| `frontend/components/pipeline/EventLog.tsx` | Scrollable live event list |
| `frontend/components/experiments/RunSidebar.tsx` | Experiment dropdown + run list |
| `frontend/components/experiments/ChartPanel.tsx` | Renders charts for selected run |
| `frontend/components/experiments/charts/TrainerLineChart.tsx` | Recharts Line: loss/accuracy over steps |
| `frontend/components/experiments/charts/EvaluatorRadarChart.tsx` | Recharts Radar: precision/recall/F1/AUC |
| `frontend/components/experiments/charts/EvaluatorBarChart.tsx` | Recharts Bar: a11y fallback alongside Radar |
| `frontend/components/experiments/charts/DeploymentBarChart.tsx` | Recharts horizontal Bar: model comparison + CSV export |
| `frontend/components/monitoring/DriftTable.tsx` | Column drift results table |
| `frontend/components/monitoring/LatestReport.tsx` | Drift badge + share + DriftTable |
| `frontend/components/monitoring/AdHocForm.tsx` | File upload + index selectors + Run Drift |
| `frontend/app/layout.tsx` | Root layout: fonts, Providers, TopNav |
| `frontend/app/page.tsx` | Redirect to /pipeline |
| `frontend/app/pipeline/page.tsx` | Two-column Pipeline tab |
| `frontend/app/experiments/page.tsx` | Sidebar + panel Experiments tab |
| `frontend/app/monitoring/page.tsx` | Sub-tabbed Monitoring tab |

---

### Task 1: Scaffold the project

**Files:**
- Create: `frontend/` (Next.js project root)
- Create: `frontend/vitest.config.ts`
- Create: `frontend/vitest.setup.ts`
- Modify: `frontend/tailwind.config.ts`
- Modify: `frontend/package.json`

- [ ] **Step 1: Scaffold Next.js from repo root**

```bash
npx create-next-app@latest frontend --typescript --tailwind --app --eslint
```
When prompted: use default import alias `@/*` (accept default), no src directory.

- [ ] **Step 2: Install additional dependencies**

```bash
cd frontend
npm install zustand @tanstack/react-query recharts
npm install -D vitest @vitejs/plugin-react @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @types/node
```

- [ ] **Step 3: Create `frontend/vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    globals: true,
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, '.') },
  },
})
```

- [ ] **Step 4: Create `frontend/vitest.setup.ts`**

```ts
import '@testing-library/jest-dom'
```

- [ ] **Step 5: Add test script to `frontend/package.json`**

In the `"scripts"` section, add:
```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 6: Configure Tailwind palette and fonts in `frontend/tailwind.config.ts`**

```ts
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './hooks/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#1e3a5f',
          700: '#1d4ed8',
          900: '#1e3a5f',
        },
        amber: {
          DEFAULT: '#D97706',
          600: '#D97706',
        },
      },
      fontFamily: {
        sans: ['Fira Sans', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
export default config
```

- [ ] **Step 7: Verify scaffold**

```bash
cd frontend && npm run dev
```
Expected: Next.js dev server starts on port 3000, default page loads.

```bash
npm test
```
Expected: `No test files found` (exit 0).

- [ ] **Step 8: Commit**

```bash
cd ..
git add frontend/
git commit -m "feat: scaffold Next.js 15 frontend with Tailwind, Vitest, Zustand, TanStack Query"
```

---

### Task 2: TypeScript types

**Files:**
- Create: `frontend/types/api.ts`

- [ ] **Step 1: Create `frontend/types/api.ts`**

```ts
export type RunStatus = 'running' | 'awaiting_approval' | 'complete' | 'failed'

export type PipelineEventType =
  | 'routing'
  | 'tool_call'
  | 'tool_result'
  | 'agent_reasoning'
  | 'hitl_request'
  | 'run_complete'

export interface PipelineEvent {
  type: PipelineEventType
  agent: string
  timestamp_ms: number
  data: Record<string, unknown>
}

export interface RunStatusResponse {
  run_id: string
  status: RunStatus
  interrupt_value: Record<string, unknown> | null
}

export interface HITLDecision {
  decision: 'approve' | 'reject'
  reason?: string
}

export interface ExperimentOut {
  experiment_id: string
  name: string
}

export type LineStyle = 'solid' | 'dashed' | 'dotted'

export interface MetricSeries {
  name: string
  steps: number[]
  values: number[]
  line_style: LineStyle
}

export interface RunOut {
  run_id: string
  run_name: string
  status: string
  start_time: string
  params: Record<string, string>
  metrics: Record<string, number>
  metric_series: MetricSeries[]
}

export interface ColumnDriftResult {
  column: string
  drift_detected: boolean
  score: number
  method: string
}

export interface DriftReport {
  dataset_drift: boolean
  drift_share: number
  columns: ColumnDriftResult[]
  generated_at: string
}

export interface HealthResponse {
  status: 'ok'
  mlflow: boolean
  graph: boolean
}
```

- [ ] **Step 2: Verify types compile**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/types/
git commit -m "feat: add TypeScript API types mirroring FastAPI models"
```

---

### Task 3: API client

**Files:**
- Create: `frontend/lib/api.ts`
- Create: `frontend/lib/query-client.ts`
- Create: `frontend/__tests__/lib/api.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/lib/api.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => mockFetch.mockReset())

describe('startRun', () => {
  it('POSTs dataset_paths and returns run_id', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ run_id: 'abc-123' }),
    })
    const { startRun } = await import('@/lib/api')
    const result = await startRun(['iris.csv'])
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/runs',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ dataset_paths: ['iris.csv'] }),
      })
    )
    expect(result).toEqual({ run_id: 'abc-123' })
  })
})

describe('approveRun', () => {
  it('POSTs decision to approve endpoint', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) })
    const { approveRun } = await import('@/lib/api')
    await approveRun('abc-123', { decision: 'approve' })
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/runs/abc-123/approve',
      expect.objectContaining({ method: 'POST' })
    )
  })
})

describe('fetchExperiments', () => {
  it('GETs /experiments', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{ experiment_id: '0', name: 'Default' }],
    })
    const { fetchExperiments } = await import('@/lib/api')
    const result = await fetchExperiments()
    expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/experiments')
    expect(result[0].name).toBe('Default')
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/lib/api.test.ts
```
Expected: FAIL — `@/lib/api` not found.

- [ ] **Step 3: Create `frontend/lib/api.ts`**

```ts
import type {
  DriftReport, ExperimentOut, HealthResponse,
  HITLDecision, RunOut, RunStatusResponse,
} from '@/types/api'

const BASE = 'http://localhost:8000'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export async function startRun(dataset_paths: string[]): Promise<{ run_id: string }> {
  return json(await fetch(`${BASE}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_paths }),
  }))
}

export async function fetchRunStatus(runId: string): Promise<RunStatusResponse> {
  return json(await fetch(`${BASE}/runs/${runId}`))
}

export async function approveRun(runId: string, decision: HITLDecision): Promise<{ ok: boolean }> {
  return json(await fetch(`${BASE}/runs/${runId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(decision),
  }))
}

export async function fetchExperiments(): Promise<ExperimentOut[]> {
  return json(await fetch(`${BASE}/experiments`))
}

export async function fetchExperimentRuns(expId: string): Promise<RunOut[]> {
  return json(await fetch(`${BASE}/experiments/${expId}/runs`))
}

export async function fetchLatestDrift(): Promise<DriftReport> {
  return json(await fetch(`${BASE}/monitoring/latest`))
}

export async function runAdHocDrift(
  files: File[],
  referenceIndex: number,
  currentIndex: number,
): Promise<DriftReport> {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  form.append('reference_index', String(referenceIndex))
  form.append('current_index', String(currentIndex))
  return json(await fetch(`${BASE}/monitoring/drift`, { method: 'POST', body: form }))
}

export async function fetchHealth(): Promise<HealthResponse> {
  return json(await fetch(`${BASE}/health`))
}
```

- [ ] **Step 4: Create `frontend/lib/query-client.ts`**

```ts
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
})
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/lib/api.test.ts
```
Expected: 3 passing.

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/ frontend/__tests__/
git commit -m "feat: add typed API client and TanStack QueryClient"
```

---

### Task 4: Zustand run store

**Files:**
- Create: `frontend/stores/run-store.ts`
- Create: `frontend/__tests__/stores/run-store.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/stores/run-store.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useRunStore } from '@/stores/run-store'

beforeEach(() => useRunStore.getState().reset())

describe('useRunStore', () => {
  it('starts idle with no runId', () => {
    const s = useRunStore.getState()
    expect(s.runId).toBeNull()
    expect(s.status).toBe('idle')
    expect(s.events).toHaveLength(0)
    expect(s.hitlPending).toBe(false)
  })

  it('setRunId updates runId and status to running', () => {
    useRunStore.getState().setRunId('abc-123')
    const s = useRunStore.getState()
    expect(s.runId).toBe('abc-123')
    expect(s.status).toBe('running')
  })

  it('appendEvent adds an event', () => {
    useRunStore.getState().appendEvent({
      type: 'routing', agent: 'supervisor', timestamp_ms: 1000, data: {},
    })
    expect(useRunStore.getState().events).toHaveLength(1)
  })

  it('setHITL sets hitlPending, interruptValue, and status', () => {
    useRunStore.getState().setHITL({ model: 'v1' })
    const s = useRunStore.getState()
    expect(s.hitlPending).toBe(true)
    expect(s.interruptValue).toEqual({ model: 'v1' })
    expect(s.status).toBe('awaiting_approval')
  })

  it('clearHITL clears hitlPending', () => {
    useRunStore.getState().setHITL({ model: 'v1' })
    useRunStore.getState().clearHITL()
    expect(useRunStore.getState().hitlPending).toBe(false)
  })

  it('setStatus updates status', () => {
    useRunStore.getState().setStatus('complete')
    expect(useRunStore.getState().status).toBe('complete')
  })

  it('reset returns to initial state', () => {
    useRunStore.getState().setRunId('abc-123')
    useRunStore.getState().reset()
    expect(useRunStore.getState().runId).toBeNull()
    expect(useRunStore.getState().status).toBe('idle')
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/stores/run-store.test.ts
```
Expected: FAIL — `@/stores/run-store` not found.

- [ ] **Step 3: Create `frontend/stores/run-store.ts`**

```ts
import { create } from 'zustand'
import type { PipelineEvent, RunStatus } from '@/types/api'

interface RunState {
  runId: string | null
  status: RunStatus | 'idle'
  events: PipelineEvent[]
  interruptValue: Record<string, unknown> | null
  hitlPending: boolean
  setRunId: (id: string) => void
  appendEvent: (event: PipelineEvent) => void
  setHITL: (value: Record<string, unknown>) => void
  clearHITL: () => void
  setStatus: (status: RunStatus | 'idle') => void
  reset: () => void
}

const initial = {
  runId: null,
  status: 'idle' as const,
  events: [],
  interruptValue: null,
  hitlPending: false,
}

export const useRunStore = create<RunState>((set) => ({
  ...initial,
  setRunId: (id) => set({ runId: id, status: 'running' }),
  appendEvent: (event) => set((s) => ({ events: [...s.events, event] })),
  setHITL: (value) => set({ hitlPending: true, interruptValue: value, status: 'awaiting_approval' }),
  clearHITL: () => set({ hitlPending: false }),
  setStatus: (status) => set({ status }),
  reset: () => set(initial),
}))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/stores/run-store.test.ts
```
Expected: 7 passing.

- [ ] **Step 5: Commit**

```bash
git add frontend/stores/ frontend/__tests__/stores/
git commit -m "feat: add Zustand run store for live pipeline state"
```

---

### Task 5: WebSocket hook

**Files:**
- Create: `frontend/hooks/use-run-stream.ts`
- Create: `frontend/__tests__/hooks/use-run-stream.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/hooks/use-run-stream.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useRunStore } from '@/stores/run-store'

class MockWS {
  static instance: MockWS
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  constructor() { MockWS.instance = this }
}
vi.stubGlobal('WebSocket', MockWS)

vi.mock('@/lib/api', () => ({
  fetchRunStatus: vi.fn().mockResolvedValue({
    run_id: 'abc-123', status: 'running', interrupt_value: null,
  }),
}))

beforeEach(() => useRunStore.getState().reset())

describe('useRunStream', () => {
  it('appends events from WebSocket messages', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'routing', agent: 'supervisor', timestamp_ms: 1, data: {} }),
      })
    })
    expect(useRunStore.getState().events).toHaveLength(1)
    expect(useRunStore.getState().events[0].type).toBe('routing')
  })

  it('sets hitlPending on hitl_request event', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'hitl_request', agent: 'deployer', timestamp_ms: 2, data: { model: 'v1' } }),
      })
    })
    expect(useRunStore.getState().hitlPending).toBe(true)
    expect(useRunStore.getState().status).toBe('awaiting_approval')
  })

  it('sets status complete on run_complete event', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'run_complete', agent: '', timestamp_ms: 3, data: {} }),
      })
    })
    expect(useRunStore.getState().status).toBe('complete')
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/hooks/use-run-stream.test.ts
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/hooks/use-run-stream.ts`**

```ts
import { useEffect, useRef } from 'react'
import { useRunStore } from '@/stores/run-store'
import { fetchRunStatus } from '@/lib/api'
import type { PipelineEvent } from '@/types/api'

const WS_BASE = 'ws://localhost:8000'
const MAX_RETRIES = 3

export function useRunStream(runId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const retriesRef = useRef(0)

  useEffect(() => {
    if (!runId) return

    function connect() {
      const ws = new WebSocket(`${WS_BASE}/ws/${runId}`)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const event: PipelineEvent = JSON.parse(e.data as string)
        const { appendEvent, setHITL, setStatus } = useRunStore.getState()
        appendEvent(event)
        if (event.type === 'hitl_request') {
          setHITL(event.data)
        }
        if (event.type === 'run_complete') {
          setStatus('complete')
          ws.close()
        }
      }

      ws.onerror = () => {
        if (retriesRef.current >= MAX_RETRIES) return
        retriesRef.current++
        const delay = Math.pow(2, retriesRef.current) * 200
        setTimeout(async () => {
          const state = await fetchRunStatus(runId!)
          const { setRunId, setHITL, setStatus } = useRunStore.getState()
          setRunId(state.run_id)
          if (state.status === 'awaiting_approval' && state.interrupt_value) {
            setHITL(state.interrupt_value)
          } else {
            setStatus(state.status)
          }
          connect()
        }, delay)
      }
    }

    connect()
    return () => {
      wsRef.current?.close()
      retriesRef.current = 0
    }
  }, [runId])
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/hooks/use-run-stream.test.ts
```
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add frontend/hooks/use-run-stream.ts frontend/__tests__/hooks/
git commit -m "feat: add useRunStream WebSocket hook with retry and HITL detection"
```

---

### Task 6: Approve hook

**Files:**
- Create: `frontend/hooks/use-approve.ts`
- Create: `frontend/__tests__/hooks/use-approve.test.tsx`

- [ ] **Step 1: Write failing test**

Create `frontend/__tests__/hooks/use-approve.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

vi.mock('@/lib/api', () => ({
  approveRun: vi.fn().mockResolvedValue({ ok: true }),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
)

describe('useApprove', () => {
  it('calls approveRun and clears HITL on success', async () => {
    useRunStore.getState().reset()
    useRunStore.getState().setHITL({ model: 'v1' })

    const { useApprove } = await import('@/hooks/use-approve')
    const { result } = renderHook(() => useApprove('abc-123'), { wrapper })

    await act(async () => { await result.current.approve('approve') })

    const { approveRun } = await import('@/lib/api')
    expect(approveRun).toHaveBeenCalledWith('abc-123', { decision: 'approve', reason: '' })
    expect(useRunStore.getState().hitlPending).toBe(false)
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/hooks/use-approve.test.tsx
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/hooks/use-approve.ts`**

```ts
import { useMutation } from '@tanstack/react-query'
import { approveRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'

export function useApprove(runId: string | null) {
  const clearHITL = useRunStore((s) => s.clearHITL)

  const mutation = useMutation({
    mutationFn: (decision: 'approve' | 'reject') =>
      approveRun(runId!, { decision, reason: '' }),
    onSuccess: () => clearHITL(),
  })

  return {
    approve: (decision: 'approve' | 'reject') => mutation.mutateAsync(decision),
    isPending: mutation.isPending,
    isError: mutation.isError,
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/hooks/use-approve.test.tsx
```
Expected: 1 passing.

- [ ] **Step 5: Commit**

```bash
git add frontend/hooks/use-approve.ts frontend/__tests__/hooks/use-approve.test.tsx
git commit -m "feat: add useApprove mutation hook for HITL gate"
```

---

### Task 7: Root layout and TopNav

**Files:**
- Create: `frontend/components/Providers.tsx`
- Create: `frontend/components/TopNav.tsx`
- Modify: `frontend/app/layout.tsx`
- Create: `frontend/app/page.tsx`
- Create: `frontend/__tests__/components/TopNav.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/components/TopNav.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ usePathname: () => '/pipeline' }))
vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query')
  return { ...actual }
})
vi.mock('@/lib/api', () => ({ fetchHealth: vi.fn().mockResolvedValue({ status: 'ok', mlflow: true, graph: true }) }))

describe('TopNav', () => {
  it('renders all three tabs', async () => {
    const { TopNav } = await import('@/components/TopNav')
    render(<TopNav />)
    expect(screen.getByText('Pipeline')).toBeInTheDocument()
    expect(screen.getByText('Experiments')).toBeInTheDocument()
    expect(screen.getByText('Monitoring')).toBeInTheDocument()
  })

  it('marks active tab based on pathname', async () => {
    const { TopNav } = await import('@/components/TopNav')
    render(<TopNav />)
    const active = screen.getByText('Pipeline').closest('a')
    expect(active?.className).toContain('bg-navy')
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/components/TopNav.test.tsx
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/components/Providers.tsx`**

```tsx
'use client'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from '@/lib/query-client'

export function Providers({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
```

- [ ] **Step 4: Create `frontend/components/TopNav.tsx`**

```tsx
'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '@/lib/api'

const TABS = [
  { label: 'Pipeline', href: '/pipeline' },
  { label: 'Experiments', href: '/experiments' },
  { label: 'Monitoring', href: '/monitoring' },
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
    <nav className="flex items-center gap-1 bg-navy px-4 py-2">
      {TABS.map(({ label, href }) => {
        const active = pathname.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            className={`rounded px-4 py-2 text-sm font-medium transition-colors ${
              active
                ? 'bg-navy text-white ring-1 ring-white/30'
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
            }`}
          >
            {label}
          </Link>
        )
      })}
      {unhealthy && (
        <span className="ml-auto rounded-full bg-red-500 px-3 py-1 text-xs font-medium text-white">
          Backend unhealthy
        </span>
      )}
    </nav>
  )
}
```

- [ ] **Step 5: Update `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from 'next'
import './globals.css'
import { TopNav } from '@/components/TopNav'
import { Providers } from '@/components/Providers'

export const metadata: Metadata = { title: 'MLOps Dashboard' }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-slate-50 font-sans text-slate-900">
        <Providers>
          <TopNav />
          <main className="p-6">{children}</main>
        </Providers>
      </body>
    </html>
  )
}
```

- [ ] **Step 6: Create `frontend/app/page.tsx`**

```tsx
import { redirect } from 'next/navigation'
export default function Home() {
  redirect('/pipeline')
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/components/TopNav.test.tsx
```
Expected: 2 passing.

- [ ] **Step 8: Commit**

```bash
git add frontend/components/Providers.tsx frontend/components/TopNav.tsx frontend/app/layout.tsx frontend/app/page.tsx frontend/__tests__/components/TopNav.test.tsx
git commit -m "feat: add root layout, TopNav with health badge, QueryClientProvider"
```

---

### Task 8: Pipeline tab

**Files:**
- Create: `frontend/components/pipeline/RunStatusBadge.tsx`
- Create: `frontend/components/pipeline/HITLGate.tsx`
- Create: `frontend/components/pipeline/EventLog.tsx`
- Create: `frontend/components/pipeline/TriggerPanel.tsx`
- Create: `frontend/app/pipeline/page.tsx`
- Create: `frontend/__tests__/components/pipeline/HITLGate.test.tsx`
- Create: `frontend/__tests__/components/pipeline/EventLog.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/components/pipeline/HITLGate.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

vi.mock('@/lib/api', () => ({ approveRun: vi.fn().mockResolvedValue({ ok: true }) }))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
)

describe('HITLGate', () => {
  it('renders nothing when hitlPending is false', async () => {
    useRunStore.getState().reset()
    const { HITLGate } = await import('@/components/pipeline/HITLGate')
    const { container } = render(<HITLGate runId="abc" />, { wrapper })
    expect(container.firstChild).toBeNull()
  })

  it('renders Approve and Reject when hitlPending is true', async () => {
    useRunStore.getState().reset()
    useRunStore.getState().setHITL({ model: 'v1', accuracy: 0.96 })
    const { HITLGate } = await import('@/components/pipeline/HITLGate')
    render(<HITLGate runId="abc" />, { wrapper })
    expect(screen.getByText('Approve')).toBeInTheDocument()
    expect(screen.getByText('Reject')).toBeInTheDocument()
  })
})
```

Create `frontend/__tests__/components/pipeline/EventLog.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

describe('EventLog', () => {
  it('renders event types in the log', async () => {
    useRunStore.getState().reset()
    useRunStore.getState().appendEvent({ type: 'routing', agent: 'supervisor', timestamp_ms: 1, data: {} })
    useRunStore.getState().appendEvent({ type: 'tool_call', agent: 'trainer', timestamp_ms: 2, data: {} })
    const { EventLog } = await import('@/components/pipeline/EventLog')
    render(<EventLog />)
    expect(screen.getByText('routing')).toBeInTheDocument()
    expect(screen.getByText('tool_call')).toBeInTheDocument()
  })

  it('shows placeholder when no events', async () => {
    useRunStore.getState().reset()
    const { EventLog } = await import('@/components/pipeline/EventLog')
    render(<EventLog />)
    expect(screen.getByText(/waiting/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/components/pipeline/
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/components/pipeline/RunStatusBadge.tsx`**

```tsx
'use client'
import { useRunStore } from '@/stores/run-store'

const STYLES: Record<string, string> = {
  idle: 'bg-slate-200 text-slate-600',
  running: 'animate-pulse bg-blue-100 text-blue-700',
  awaiting_approval: 'bg-amber-100 text-amber-700',
  complete: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
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

- [ ] **Step 4: Create `frontend/components/pipeline/HITLGate.tsx`**

```tsx
'use client'
import { useRunStore } from '@/stores/run-store'
import { useApprove } from '@/hooks/use-approve'

export function HITLGate({ runId }: { runId: string | null }) {
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const { approve, isPending } = useApprove(runId)

  if (!hitlPending) return null

  return (
    <div className="rounded-lg border border-amber-600 bg-amber-50 p-4">
      <p className="mb-2 font-semibold text-amber-800">⚠ Deployment Gate</p>
      {interruptValue && (
        <pre className="mb-3 overflow-auto rounded bg-amber-100 p-2 font-mono text-xs text-amber-900">
          {JSON.stringify(interruptValue, null, 2)}
        </pre>
      )}
      <div className="flex gap-2">
        <button
          onClick={() => approve('approve')}
          disabled={isPending}
          className="rounded bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
        >
          Approve
        </button>
        <button
          onClick={() => approve('reject')}
          disabled={isPending}
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/components/pipeline/EventLog.tsx`**

```tsx
'use client'
import { useEffect, useRef } from 'react'
import { useRunStore } from '@/stores/run-store'
import type { PipelineEventType } from '@/types/api'

const TYPE_COLORS: Record<PipelineEventType, string> = {
  routing: 'text-blue-600',
  tool_call: 'text-slate-500',
  tool_result: 'text-slate-400',
  agent_reasoning: 'text-indigo-600',
  hitl_request: 'font-semibold text-amber-600',
  run_complete: 'font-semibold text-green-600',
}

export function EventLog() {
  const events = useRunStore((s) => s.events)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  return (
    <div className="h-full overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-xs">
      {events.length === 0 && (
        <p className="text-slate-400">Waiting for pipeline events...</p>
      )}
      {events.map((e, i) => (
        <div key={i} className="mb-1 flex gap-2">
          <span className="shrink-0 text-slate-300">
            {new Date(e.timestamp_ms).toISOString().slice(11, 23)}
          </span>
          <span className={TYPE_COLORS[e.type]}>{e.type}</span>
          <span className="text-slate-500">{e.agent}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
```

- [ ] **Step 6: Create `frontend/components/pipeline/TriggerPanel.tsx`**

```tsx
'use client'
import { useState } from 'react'
import { startRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'
import { RunStatusBadge } from './RunStatusBadge'

export function TriggerPanel({ onRunStarted }: { onRunStarted: (id: string) => void }) {
  const [paths, setPaths] = useState('')
  const [loading, setLoading] = useState(false)
  const status = useRunStore((s) => s.status)

  async function handleRun() {
    const dataset_paths = paths.split(',').map((p) => p.trim()).filter(Boolean)
    if (!dataset_paths.length) return
    setLoading(true)
    try {
      const { run_id } = await startRun(dataset_paths)
      useRunStore.getState().setRunId(run_id)
      onRunStarted(run_id)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-navy-900">Start Pipeline Run</h2>
      <input
        type="text"
        value={paths}
        onChange={(e) => setPaths(e.target.value)}
        placeholder="data/samples/iris_measurements.csv"
        className="mb-3 w-full rounded border border-slate-300 px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-navy"
      />
      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={loading || status === 'running'}
          className="rounded bg-navy px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Starting...' : '▶ Run Pipeline'}
        </button>
        <RunStatusBadge />
      </div>
    </div>
  )
}
```

- [ ] **Step 7: Create `frontend/app/pipeline/page.tsx`**

```tsx
'use client'
import { useState } from 'react'
import { TriggerPanel } from '@/components/pipeline/TriggerPanel'
import { HITLGate } from '@/components/pipeline/HITLGate'
import { EventLog } from '@/components/pipeline/EventLog'
import { useRunStream } from '@/hooks/use-run-stream'

export default function PipelinePage() {
  const [runId, setRunId] = useState<string | null>(null)
  useRunStream(runId)

  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="flex w-2/5 flex-col gap-4">
        <TriggerPanel onRunStarted={setRunId} />
        <HITLGate runId={runId} />
      </div>
      <div className="flex-1 overflow-hidden">
        <EventLog />
      </div>
    </div>
  )
}
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/components/pipeline/
```
Expected: 4 passing.

- [ ] **Step 9: Commit**

```bash
git add frontend/components/pipeline/ frontend/app/pipeline/ frontend/__tests__/components/pipeline/
git commit -m "feat: add Pipeline tab — TriggerPanel, EventLog, HITLGate"
```

---

### Task 9: Experiments tab

**Files:**
- Create: `frontend/components/experiments/charts/TrainerLineChart.tsx`
- Create: `frontend/components/experiments/charts/EvaluatorRadarChart.tsx`
- Create: `frontend/components/experiments/charts/EvaluatorBarChart.tsx`
- Create: `frontend/components/experiments/charts/DeploymentBarChart.tsx`
- Create: `frontend/components/experiments/ChartPanel.tsx`
- Create: `frontend/components/experiments/RunSidebar.tsx`
- Create: `frontend/app/experiments/page.tsx`
- Create: `frontend/__tests__/components/experiments/ChartPanel.test.tsx`
- Create: `frontend/__tests__/components/experiments/RunSidebar.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/components/experiments/ChartPanel.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

describe('ChartPanel', () => {
  it('shows empty state when run is null', async () => {
    const { ChartPanel } = await import('@/components/experiments/ChartPanel')
    render(<ChartPanel run={null} />)
    expect(screen.getByText(/select a run/i)).toBeInTheDocument()
  })
})
```

Create `frontend/__tests__/components/experiments/RunSidebar.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'

vi.mock('@/lib/api', () => ({
  fetchExperiments: vi.fn().mockResolvedValue([{ experiment_id: '0', name: 'Default' }]),
  fetchExperimentRuns: vi.fn().mockResolvedValue([]),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

describe('RunSidebar', () => {
  it('renders experiment name after load', async () => {
    const { RunSidebar } = await import('@/components/experiments/RunSidebar')
    render(<RunSidebar selectedRunId={null} onSelectRun={() => {}} />, { wrapper })
    expect(await screen.findByText('Default')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/components/experiments/
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/components/experiments/charts/TrainerLineChart.tsx`**

```tsx
'use client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { MetricSeries } from '@/types/api'

const STROKE_DASH: Record<string, string> = { solid: '0', dashed: '6 3', dotted: '2 2' }
const COLORS = ['#1e3a5f', '#D97706', '#16a34a', '#7c3aed']

export function TrainerLineChart({ series }: { series: MetricSeries[] }) {
  if (!series.length) return <p className="text-xs text-slate-400">No training metrics</p>

  const maxLen = Math.max(...series.map((s) => s.steps.length))
  const data = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { step: i }
    series.forEach((s) => { if (s.steps[i] !== undefined) point[s.name] = s.values[i] })
    return point
  })

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="step" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s, i) => (
          <Line
            key={s.name}
            type="monotone"
            dataKey={s.name}
            stroke={COLORS[i % COLORS.length]}
            strokeDasharray={STROKE_DASH[s.line_style]}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 4: Create `frontend/components/experiments/charts/EvaluatorRadarChart.tsx`**

```tsx
'use client'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'

export function EvaluatorRadarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics).map(([metric, value]) => ({ metric, value }))
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
        <Radar dataKey="value" stroke="#1e3a5f" fill="#1e3a5f" fillOpacity={0.3} />
        <Tooltip />
      </RadarChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 5: Create `frontend/components/experiments/charts/EvaluatorBarChart.tsx`**

```tsx
'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList, ResponsiveContainer } from 'recharts'

export function EvaluatorBarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics).map(([name, value]) => ({ name, value }))
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={80} />
        <Tooltip />
        <Bar dataKey="value" fill="#D97706">
          <LabelList dataKey="value" position="right" style={{ fontSize: 10 }} formatter={(v: number) => v.toFixed(3)} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 6: Create `frontend/components/experiments/charts/DeploymentBarChart.tsx`**

```tsx
'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList, ResponsiveContainer } from 'recharts'

function downloadCSV(metrics: Record<string, number>) {
  const csv = 'metric,value\n' + Object.entries(metrics).map(([k, v]) => `${k},${v}`).join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  a.download = 'deployment-metrics.csv'
  a.click()
}

export function DeploymentBarChart({ metrics }: { metrics: Record<string, number> }) {
  const data = Object.entries(metrics)
    .sort(([, a], [, b]) => b - a)
    .map(([name, value]) => ({ name, value }))
  if (!data.length) return null
  return (
    <div>
      <div className="mb-1 flex justify-end">
        <button
          onClick={() => downloadCSV(metrics)}
          className="rounded px-2 py-1 text-xs font-medium text-amber-600 hover:bg-amber-50"
        >
          Export CSV
        </button>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
          <Tooltip />
          <Bar dataKey="value" fill="#1e3a5f">
            <LabelList dataKey="value" position="right" style={{ fontSize: 10 }} formatter={(v: number) => v.toFixed(3)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 7: Create `frontend/components/experiments/ChartPanel.tsx`**

```tsx
import type { RunOut } from '@/types/api'
import { TrainerLineChart } from './charts/TrainerLineChart'
import { EvaluatorRadarChart } from './charts/EvaluatorRadarChart'
import { EvaluatorBarChart } from './charts/EvaluatorBarChart'
import { DeploymentBarChart } from './charts/DeploymentBarChart'

export function ChartPanel({ run }: { run: RunOut | null }) {
  if (!run) {
    return (
      <div className="flex h-full items-center justify-center text-slate-400">
        Select a run to view charts
      </div>
    )
  }
  return (
    <div className="space-y-6 overflow-y-auto">
      <section>
        <h3 className="mb-2 text-sm font-semibold text-navy-900">Trainer — Loss & Accuracy</h3>
        <TrainerLineChart series={run.metric_series} />
      </section>
      <section>
        <h3 className="mb-2 text-sm font-semibold text-navy-900">Evaluator Metrics</h3>
        <div className="grid grid-cols-2 gap-4">
          <EvaluatorRadarChart metrics={run.metrics} />
          <EvaluatorBarChart metrics={run.metrics} />
        </div>
      </section>
      <section>
        <h3 className="mb-2 text-sm font-semibold text-navy-900">Deployment Comparison</h3>
        <DeploymentBarChart metrics={run.metrics} />
      </section>
    </div>
  )
}
```

- [ ] **Step 8: Create `frontend/components/experiments/RunSidebar.tsx`**

```tsx
'use client'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchExperiments, fetchExperimentRuns } from '@/lib/api'
import type { RunOut } from '@/types/api'

interface Props {
  selectedRunId: string | null
  onSelectRun: (run: RunOut) => void
}

export function RunSidebar({ selectedRunId, onSelectRun }: Props) {
  const { data: experiments } = useQuery({ queryKey: ['experiments'], queryFn: fetchExperiments })
  const [expId, setExpId] = useState<string | null>(null)

  const activeExpId = expId ?? experiments?.[0]?.experiment_id ?? null

  const { data: runs } = useQuery({
    queryKey: ['runs', activeExpId],
    queryFn: () => fetchExperimentRuns(activeExpId!),
    enabled: !!activeExpId,
  })

  return (
    <div className="flex h-full flex-col gap-3">
      <select
        value={activeExpId ?? ''}
        onChange={(e) => setExpId(e.target.value)}
        className="rounded border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-navy"
      >
        {experiments?.map((exp) => (
          <option key={exp.experiment_id} value={exp.experiment_id}>
            {exp.name}
          </option>
        ))}
      </select>
      <div className="flex-1 space-y-1 overflow-y-auto">
        {runs?.map((run) => (
          <button
            key={run.run_id}
            onClick={() => onSelectRun(run)}
            className={`w-full rounded px-3 py-2 text-left text-xs transition-colors ${
              selectedRunId === run.run_id
                ? 'bg-amber-600 text-white'
                : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
          >
            <div className="font-mono">{run.run_name || run.run_id.slice(0, 8)}</div>
            <div className="text-[10px] opacity-70">
              acc: {run.metrics.accuracy?.toFixed(3) ?? '—'}
            </div>
          </button>
        ))}
        {runs?.length === 0 && <p className="text-xs text-slate-400">No runs found</p>}
      </div>
    </div>
  )
}
```

- [ ] **Step 9: Create `frontend/app/experiments/page.tsx`**

```tsx
'use client'
import { useState } from 'react'
import { RunSidebar } from '@/components/experiments/RunSidebar'
import { ChartPanel } from '@/components/experiments/ChartPanel'
import type { RunOut } from '@/types/api'

export default function ExperimentsPage() {
  const [selectedRun, setSelectedRun] = useState<RunOut | null>(null)
  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="w-64 shrink-0 overflow-hidden">
        <RunSidebar selectedRunId={selectedRun?.run_id ?? null} onSelectRun={setSelectedRun} />
      </div>
      <div className="flex-1 overflow-hidden rounded-lg border border-slate-200 bg-white p-4">
        <ChartPanel run={selectedRun} />
      </div>
    </div>
  )
}
```

- [ ] **Step 10: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/components/experiments/
```
Expected: 2 passing.

- [ ] **Step 11: Commit**

```bash
git add frontend/components/experiments/ frontend/app/experiments/ frontend/__tests__/components/experiments/
git commit -m "feat: add Experiments tab — RunSidebar, ChartPanel, Recharts"
```

---

### Task 10: Monitoring tab

**Files:**
- Create: `frontend/components/monitoring/DriftTable.tsx`
- Create: `frontend/components/monitoring/LatestReport.tsx`
- Create: `frontend/components/monitoring/AdHocForm.tsx`
- Create: `frontend/app/monitoring/page.tsx`
- Create: `frontend/__tests__/components/monitoring/DriftTable.test.tsx`
- Create: `frontend/__tests__/components/monitoring/LatestReport.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/components/monitoring/DriftTable.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import type { ColumnDriftResult } from '@/types/api'

describe('DriftTable', () => {
  it('renders column names and drift indicators', async () => {
    const columns: ColumnDriftResult[] = [
      { column: 'sepal_length', drift_detected: false, score: 0.04, method: 'PSI' },
      { column: 'petal_width', drift_detected: true, score: 0.41, method: 'PSI' },
    ]
    const { DriftTable } = await import('@/components/monitoring/DriftTable')
    render(<DriftTable columns={columns} />)
    expect(screen.getByText('sepal_length')).toBeInTheDocument()
    expect(screen.getByText('petal_width')).toBeInTheDocument()
  })
})
```

Create `frontend/__tests__/components/monitoring/LatestReport.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'

vi.mock('@/lib/api', () => ({
  fetchLatestDrift: vi.fn().mockResolvedValue({
    dataset_drift: false,
    drift_share: 0.12,
    columns: [{ column: 'sepal_length', drift_detected: false, score: 0.04, method: 'PSI' }],
    generated_at: '2026-04-24T10:00:00',
  }),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

describe('LatestReport', () => {
  it('shows No drift badge when dataset_drift is false', async () => {
    const { LatestReport } = await import('@/components/monitoring/LatestReport')
    render(<LatestReport />, { wrapper })
    expect(await screen.findByText('No drift')).toBeInTheDocument()
  })

  it('shows drift share percentage', async () => {
    const { LatestReport } = await import('@/components/monitoring/LatestReport')
    render(<LatestReport />, { wrapper })
    expect(await screen.findByText('12.0%')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npm test -- __tests__/components/monitoring/
```
Expected: FAIL.

- [ ] **Step 3: Create `frontend/components/monitoring/DriftTable.tsx`**

```tsx
import type { ColumnDriftResult } from '@/types/api'

export function DriftTable({ columns }: { columns: ColumnDriftResult[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-slate-200 text-left text-slate-500">
          <th className="pb-1 pr-4 font-medium">Column</th>
          <th className="pb-1 pr-4 font-medium">Score</th>
          <th className="pb-1 pr-4 font-medium">Method</th>
          <th className="pb-1 font-medium">Drift</th>
        </tr>
      </thead>
      <tbody>
        {columns.map((col) => (
          <tr key={col.column} className="border-b border-slate-100">
            <td className="py-1 pr-4 font-mono">{col.column}</td>
            <td className="py-1 pr-4">{col.score.toFixed(3)}</td>
            <td className="py-1 pr-4 text-slate-500">{col.method}</td>
            <td className="py-1">
              {col.drift_detected
                ? <span className="text-red-500">✗</span>
                : <span className="text-green-600">✓</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

- [ ] **Step 4: Create `frontend/components/monitoring/LatestReport.tsx`**

```tsx
'use client'
import { useQuery } from '@tanstack/react-query'
import { fetchLatestDrift } from '@/lib/api'
import { DriftTable } from './DriftTable'

export function LatestReport() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ['monitoring', 'latest'],
    queryFn: fetchLatestDrift,
    retry: false,
  })

  if (isLoading) return <p className="text-sm text-slate-400">Loading...</p>
  if (isError || !data) {
    return (
      <p className="text-sm text-slate-400">
        No pipeline run completed yet. Run the pipeline to generate a drift report.
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${
          data.dataset_drift ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
        }`}>
          {data.dataset_drift ? 'Drift detected' : 'No drift'}
        </span>
        <span className="text-2xl font-semibold text-navy-900">
          {(data.drift_share * 100).toFixed(1)}%
        </span>
        <span className="text-sm text-slate-400">columns with drift</span>
      </div>
      <DriftTable columns={data.columns} />
      <p className="text-xs text-slate-400">
        Generated at {new Date(data.generated_at).toLocaleString()}
      </p>
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/components/monitoring/AdHocForm.tsx`**

```tsx
'use client'
import { useState, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { runAdHocDrift } from '@/lib/api'
import type { DriftReport } from '@/types/api'
import { DriftTable } from './DriftTable'

export function AdHocForm() {
  const [files, setFiles] = useState<File[]>([])
  const [refIdx, setRefIdx] = useState(0)
  const [curIdx, setCurIdx] = useState(1)
  const [result, setResult] = useState<DriftReport | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const mutation = useMutation({
    mutationFn: () => runAdHocDrift(files, refIdx, curIdx),
    onSuccess: setResult,
  })

  return (
    <div className="space-y-4">
      <div
        onClick={() => inputRef.current?.click()}
        className="cursor-pointer rounded-lg border-2 border-dashed border-slate-300 p-6 text-center hover:border-navy"
      >
        <p className="text-sm text-slate-500">
          {files.length
            ? `${files.length} file(s) selected`
            : 'Drop CSV files here or click to upload'}
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv"
          className="hidden"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </div>

      {files.length >= 2 && (
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Reference:
            <select
              value={refIdx}
              onChange={(e) => setRefIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Current:
            <select
              value={curIdx}
              onChange={(e) => setCurIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="rounded bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
          >
            {mutation.isPending ? 'Running...' : 'Run Drift'}
          </button>
        </div>
      )}

      {mutation.isError && (
        <p className="rounded bg-red-50 p-3 text-sm text-red-600">
          Drift analysis failed. Check that both files are valid CSVs with matching columns.
        </p>
      )}

      {result && (
        <div className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex items-center gap-4">
            <span className={`rounded-full px-3 py-1 text-sm font-medium ${
              result.dataset_drift ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {result.dataset_drift ? 'Drift detected' : 'No drift'}
            </span>
            <span className="text-xl font-semibold">{(result.drift_share * 100).toFixed(1)}%</span>
            <span className="text-sm text-slate-400">columns with drift</span>
          </div>
          <DriftTable columns={result.columns} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Create `frontend/app/monitoring/page.tsx`**

```tsx
'use client'
import { useState } from 'react'
import { LatestReport } from '@/components/monitoring/LatestReport'
import { AdHocForm } from '@/components/monitoring/AdHocForm'

const TABS = ['Latest Report', 'Ad-hoc Analysis'] as const
type Tab = typeof TABS[number]

export default function MonitoringPage() {
  const [tab, setTab] = useState<Tab>('Latest Report')
  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex gap-1 border-b border-slate-200">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === t
                ? 'border-b-2 border-navy text-navy-900'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        {tab === 'Latest Report' ? <LatestReport /> : <AdHocForm />}
      </div>
    </div>
  )
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd frontend && npm test -- __tests__/components/monitoring/
```
Expected: 3 passing.

- [ ] **Step 8: Run full test suite**

```bash
cd frontend && npm test
```
Expected: all tests passing (no failures).

- [ ] **Step 9: Commit**

```bash
git add frontend/components/monitoring/ frontend/app/monitoring/ frontend/__tests__/components/monitoring/
git commit -m "feat: add Monitoring tab — LatestReport, AdHocForm, DriftTable"
```

---

### Task 11: Smoke test the full UI

- [ ] **Step 1: Start the backend**

```bash
uv run uvicorn api.main:app --reload --port 8000
```
Expected: `GET http://localhost:8000/health` returns `{"status":"ok","mlflow":true,"graph":true}`.

- [ ] **Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```
Expected: Next.js starts on http://localhost:3000.

- [ ] **Step 3: Verify Pipeline tab**

Open http://localhost:3000 (redirects to /pipeline).
- Two-column layout: trigger panel on left, empty event log on right
- Enter `data/samples/iris_measurements.csv`, click Run Pipeline
- Event log fills with live colored events
- HITL gate appears (amber border) when run reaches deployment — shows interrupt payload as JSON
- Click Approve → gate disappears, events resume, status badge → complete

- [ ] **Step 4: Verify Experiments tab**

Navigate to http://localhost:3000/experiments.
- Sidebar: experiment dropdown loads, run list appears after selecting experiment
- Click a run → three chart sections render (Trainer line, Evaluator radar+bar, Deployment bar)
- Export CSV button downloads a file

- [ ] **Step 5: Verify Monitoring tab**

Navigate to http://localhost:3000/monitoring.
- Latest Report sub-tab: drift badge + percentage + column table if a run completed; empty state otherwise
- Ad-hoc Analysis sub-tab: upload two CSVs, dropdowns populate, Run Drift renders results below

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete Next.js MLOps frontend (sub-project 2)"
```
