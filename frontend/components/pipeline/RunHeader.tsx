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
