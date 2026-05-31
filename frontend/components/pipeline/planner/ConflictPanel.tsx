'use client'
import type { EvidenceConflict, SoftConflict } from '@/types/api'
import { Card } from '@/components/ui/Card'

interface Props {
  hard: EvidenceConflict[]
  soft: SoftConflict[]
}

export function ConflictPanel({ hard, soft }: Props) {
  const hardList = hard ?? []
  const softList = soft ?? []
  if (hardList.length === 0 && softList.length === 0) return null
  return (
    <Card title={hardList.length > 0 ? '⚠ Evidence conflict' : 'ℹ Retrieved but not cited'}
          className={hardList.length > 0 ? 'border-amber-300' : ''}>
      {hardList.length > 0 && (
        <div className="space-y-2">
          {hardList.map((c, i) => (
            <div key={i} className="rounded border border-amber-200 bg-amber-50 p-2 text-xs">
              <p className="font-semibold text-amber-900">{c.summary}</p>
              <p className="mt-0.5 text-amber-800">Affected: {(c.affected_models ?? []).join(', ')}</p>
              <p className="mt-1 italic text-amber-700">Resolution: {c.resolution}</p>
            </div>
          ))}
        </div>
      )}
      {softList.length > 0 && (
        <div className={`text-xs text-zinc-600 ${hardList.length > 0 ? 'mt-3 border-t border-zinc-200 pt-2' : ''}`}>
          {softList.map((s, i) => (
            <p key={i} className="italic">{s.summary}</p>
          ))}
        </div>
      )}
    </Card>
  )
}
