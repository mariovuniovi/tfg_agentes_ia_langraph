'use client'
import type { DecisionBasis, EvidenceReference } from '@/types/api'
import { Card } from '@/components/ui/Card'

function EvidenceChip({ ref }: { ref: EvidenceReference }) {
  const label = ref.source_id ? `${ref.source}:${ref.source_id}` : ref.source
  return (
    <span className="rounded bg-zinc-100 px-2 py-0.5 text-[11px] font-mono text-zinc-700">
      {label}
    </span>
  )
}

export function DecisionBasisCard({ basis }: { basis: DecisionBasis }) {
  const primary = basis.primary_evidence ?? []
  const secondary = basis.secondary_evidence ?? []
  return (
    <Card title="Decision basis">
      <p className="mb-1 text-xs font-medium text-zinc-500">Primary evidence</p>
      <div className="mb-3 flex flex-wrap gap-1">
        {primary.length > 0
          ? primary.map((e, i) => <EvidenceChip key={i} ref={e} />)
          : <span className="text-xs text-zinc-400 italic">none</span>}
      </div>
      {secondary.length > 0 && (
        <>
          <p className="mb-1 text-xs font-medium text-zinc-500">Secondary evidence</p>
          <div className="mb-3 flex flex-wrap gap-1">
            {secondary.map((e, i) => <EvidenceChip key={i} ref={e} />)}
          </div>
        </>
      )}
      <p className="text-xs leading-relaxed text-zinc-700">{basis.final_strategy || <span className="text-zinc-400 italic">no strategy</span>}</p>
    </Card>
  )
}
