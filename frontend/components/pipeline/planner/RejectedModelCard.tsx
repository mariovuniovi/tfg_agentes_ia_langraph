'use client'
import { useState } from 'react'
import type { RejectedModelRationale } from '@/types/api'

export function RejectedModelCard({ rejected }: { rejected: RejectedModelRationale }) {
  const [expanded, setExpanded] = useState(false)
  const evidence = rejected.evidence_refs ?? []
  return (
    <div className="rounded border border-zinc-200 bg-white p-2 text-xs">
      <button type="button" onClick={() => setExpanded((e) => !e)} className="flex w-full items-center gap-2 text-left">
        <span className="font-mono font-semibold text-red-600 line-through">{rejected.model_key}</span>
        <span className="ml-auto text-zinc-400">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <>
          <p className="mt-1 text-zinc-700">{rejected.reason || <span className="text-zinc-400 italic">no reason given</span>}</p>
          {rejected.reconsider_if && (
            <p className="mt-1 italic text-zinc-500">Reconsider if: {rejected.reconsider_if}</p>
          )}
          {evidence.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {evidence.map((r, i) => (
                <span key={i} className="rounded bg-zinc-100 px-2 py-0.5 text-[11px] font-mono text-zinc-700">
                  {r.source_id ? `${r.source}:${r.source_id}` : r.source}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
