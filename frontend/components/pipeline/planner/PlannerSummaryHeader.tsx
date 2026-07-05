'use client'
import type { CandidateRationale, RejectedModelRationale } from '@/types/api'

interface Props {
  candidates: CandidateRationale[]
  rejected: RejectedModelRationale[]
  status: 'ok' | 'retry_ok' | 'failed'
}

const STATUS_STYLES: Record<string, string> = {
  ok:        'bg-emerald-50 text-emerald-700 ring-emerald-200',
  retry_ok:  'bg-amber-50 text-amber-700 ring-amber-200',
  failed:    'bg-red-50 text-red-700 ring-red-200',
}

export function PlannerSummaryHeader({ candidates, rejected, status }: Props) {
  const cands = candidates ?? []
  const rejs = rejected ?? []
  const statusKey = status ?? 'ok'
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="font-semibold text-zinc-700">Selected ({cands.length}):</span>
      {cands.map((c) => (
        <span key={c.model_key} className="rounded-full bg-violet-50 px-2 py-0.5 text-violet-700">
          #{c.priority ?? '?'} {c.model_key}
        </span>
      ))}
      <span className="ml-3 font-semibold text-zinc-700">Rejected ({rejs.length}):</span>
      {rejs.map((r) => (
        <span key={r.model_key} className="rounded-full bg-red-50 px-2 py-0.5 text-red-600 line-through">
          {r.model_key}
        </span>
      ))}
      <span className={`ml-auto inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${STATUS_STYLES[statusKey] ?? STATUS_STYLES.ok}`}>
        {statusKey.replace('_', ' ')}
      </span>
    </div>
  )
}
