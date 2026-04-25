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
