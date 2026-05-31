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
  return (
    <Card title="Decision basis">
      <p className="mb-1 text-xs font-medium text-zinc-500">Primary evidence</p>
      <div className="mb-3 flex flex-wrap gap-1">
        {basis.primary_evidence.map((e, i) => <EvidenceChip key={i} ref={e} />)}
      </div>
      {basis.secondary_evidence.length > 0 && (
        <>
          <p className="mb-1 text-xs font-medium text-zinc-500">Secondary evidence</p>
          <div className="mb-3 flex flex-wrap gap-1">
            {basis.secondary_evidence.map((e, i) => <EvidenceChip key={i} ref={e} />)}
          </div>
        </>
      )}
      <p className="text-xs leading-relaxed text-zinc-700">{basis.final_strategy}</p>
    </Card>
  )
}
