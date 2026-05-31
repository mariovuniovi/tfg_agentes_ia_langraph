'use client'
import type { ExperienceSummary } from '@/types/api'
import { Card } from '@/components/ui/Card'

interface Props {
  experiences: ExperienceSummary[]
  citedIds: string[]
}

export function EvidenceQualityCard({ experiences, citedIds }: Props) {
  const exps = experiences ?? []
  const citedSet = new Set(citedIds ?? [])
  const tiers = { high: 0, medium: 0, low: 0 }
  let scaleMismatchCount = 0
  for (const e of exps) {
    const t = e.relevance_tier ?? 'low'
    if (t === 'high' || t === 'medium' || t === 'low') tiers[t]++
    if (e.target_scale_note) scaleMismatchCount++
  }
  return (
    <Card title="Evidence quality">
      <p className="text-xs text-zinc-700">
        Available experiences: {exps.length}   ·   Cited: {citedSet.size}
      </p>
      <p className="mt-1 text-xs text-zinc-700">
        Relevance:   high: {tiers.high}   ·   medium: {tiers.medium}   ·   low: {tiers.low}
      </p>
      {scaleMismatchCount > 0 && (
        <p className="mt-2 text-[11px] italic text-amber-700">
          ⚠ {scaleMismatchCount} experience(s) have target-scale mismatches — raw metric values
          may not be directly comparable.
        </p>
      )}
    </Card>
  )
}
