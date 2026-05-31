'use client'
import { useState } from 'react'
import type { CandidateRationale, EvidenceReference } from '@/types/api'

function EvidenceChip({ ref, cited }: { ref: EvidenceReference; cited?: boolean }) {
  const label = ref.source_id ? `${ref.source}:${ref.source_id}` : ref.source
  return (
    <span className={`rounded px-2 py-0.5 text-[11px] font-mono ${cited ? 'bg-violet-100 text-violet-700' : 'bg-zinc-100 text-zinc-700'}`}>
      {label}{cited && ' (cited)'}
    </span>
  )
}

export function CandidateCard({ candidate }: { candidate: CandidateRationale }) {
  const [expanded, setExpanded] = useState(false)
  const evidence = candidate.evidence_refs ?? []
  const risks = candidate.risks ?? []
  return (
    <div className="rounded border border-zinc-200 bg-white p-3 text-xs">
      <button type="button" onClick={() => setExpanded((e) => !e)} className="flex w-full items-center gap-2 text-left">
        <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-semibold text-violet-700">#{candidate.priority ?? '?'}</span>
        <span className="font-mono font-semibold text-zinc-900">{candidate.model_key}</span>
        <span className="ml-auto text-zinc-400">{expanded ? '▾' : '▸'}</span>
      </button>
      <p className="mt-1 text-zinc-700">{candidate.reason || <span className="text-zinc-400 italic">no reason given</span>}</p>
      {expanded && (
        <>
          <p className="mt-2 text-[11px] font-medium text-zinc-500">Evidence</p>
          <div className="mt-1 flex flex-wrap gap-1">
            {evidence.length > 0
              ? evidence.map((r, i) => <EvidenceChip key={i} ref={r} />)
              : <span className="text-zinc-400 italic">none</span>}
          </div>
          {risks.length > 0 && (
            <>
              <p className="mt-2 text-[11px] font-medium text-zinc-500">Risks</p>
              <ul className="ml-4 list-disc text-zinc-600">
                {risks.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  )
}
