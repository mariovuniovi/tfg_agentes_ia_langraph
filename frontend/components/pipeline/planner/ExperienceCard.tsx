'use client'
import type { ExperienceSummary } from '@/types/api'

const TIER_STYLES = {
  high:   'bg-emerald-50 text-emerald-700 ring-emerald-200',
  medium: 'bg-amber-50 text-amber-700 ring-amber-200',
  low:    'bg-zinc-100 text-zinc-600 ring-zinc-200',
}

export function ExperienceCard({ exp, cited }: { exp: ExperienceSummary; cited: boolean }) {
  const tier = exp.relevance_tier ?? 'low'
  const matched = exp.matched_buckets ?? []
  const mismatched = exp.mismatched_buckets ?? []
  return (
    <div className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-200 bg-white'}`}>
      <div className="flex items-center gap-2">
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${TIER_STYLES[tier]}`}>
          {tier} · similarity {(exp.similarity_score ?? 0).toFixed(2)}
        </span>
        {cited && (
          <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] text-violet-700">cited</span>
        )}
      </div>
      <div className="mt-1 text-zinc-700">
        Best model: <span className="font-mono font-semibold">{exp.best_model ?? '—'}</span>
        {' · '}
        <span className="uppercase">{exp.metric_name ?? 'metric'}</span>: <span className="font-mono">{(exp.validation_score ?? 0).toFixed(4)}</span>
      </div>
      {exp.target_scale_note && (
        <p className="mt-1 italic text-amber-700">⚠ {exp.target_scale_note}</p>
      )}
      {matched.length > 0 && (
        <p className="mt-1 text-[11px] text-zinc-500">Matched: {matched.join(', ')}</p>
      )}
      {mismatched.length > 0 && (
        <p className="text-[11px] text-zinc-500">Mismatched: {mismatched.join(', ')}</p>
      )}
    </div>
  )
}
