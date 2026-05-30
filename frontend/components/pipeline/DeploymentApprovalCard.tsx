'use client'
import { useState } from 'react'
import type { DeployerInterrupt } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

interface Props {
  runId: string | null
  interrupt: DeployerInterrupt
  onApprove: (decision: 'approve' | 'reject') => void
  isPending: boolean
}

export function DeploymentApprovalCard({ runId: _runId, interrupt, onApprove, isPending }: Props) {
  const [showRaw, setShowRaw] = useState(false)
  const action = interrupt.deployment_action
  const audit = interrupt.evaluation_report_audit ?? {}
  const risks = (audit.risks_and_warnings ?? []).slice(0, 3)
  const candidateShort = (interrupt.candidate_run_id ?? '').slice(0, 8)

  return (
    <Card
      title="Deployment approval required"
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
        <button
          type="button"
          onClick={() => setShowRaw((s) => !s)}
          className="ml-auto rounded border border-[var(--color-border)] px-2 py-1.5 text-[11px] text-zinc-500 hover:bg-zinc-50"
        >
          {showRaw ? 'Hide' : 'Show'} raw payload
        </button>
      </div>

      {showRaw && (
        <pre className="mt-3 overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
          {JSON.stringify(interrupt, null, 2)}
        </pre>
      )}
    </Card>
  )
}
