'use client'

import type { JoinPlan, JoinCandidateEvaluation } from '@/types/api'

interface JoinPlanPanelProps {
  joinPlan?: JoinPlan | null
  joinBaseNrows?: number | null
}

function formatPct(x: number): string {
  return `${(x * 100).toFixed(1)}%`
}

function formatMultiplier(x: number): string {
  return `${x.toFixed(2)}×`
}

function RiskBadge({ risk }: { risk: 'low' | 'medium' | 'high' }) {
  const colors = {
    low: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    medium: 'bg-amber-50 text-amber-700 border-amber-200',
    high: 'bg-red-50 text-red-700 border-red-200',
  }
  return (
    <span className={`rounded border px-1.5 py-0.5 text-xs font-medium ${colors[risk]}`}>
      {risk} risk
    </span>
  )
}

export function JoinPlanPanel({ joinPlan, joinBaseNrows }: JoinPlanPanelProps) {
  if (!joinPlan) {
    return (
      <div className="mt-4 rounded border border-zinc-100 bg-zinc-50 px-3 py-2 text-xs text-zinc-500">
        No inferred join plan — target dataset built from a single source.
      </div>
    )
  }

  return (
    <div className="mt-4 space-y-3">
      <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wide">
        Join Plan ({joinPlan.mode})
      </div>

      {/* Base dataset */}
      <div className="rounded border border-zinc-200 p-3 text-xs">
        <div className="font-medium text-zinc-700">
          Base dataset: <span className="font-mono">{joinPlan.base_dataset.dataset_name}</span>
          {joinBaseNrows != null && (
            <span className="ml-2 text-zinc-400">({joinBaseNrows.toLocaleString()} rows preserved)</span>
          )}
          <span className={`ml-2 rounded px-1.5 py-0.5 text-[10px] font-medium ${
            joinPlan.base_dataset.confidence === 'high' ? 'bg-emerald-50 text-emerald-700' :
            joinPlan.base_dataset.confidence === 'medium' ? 'bg-amber-50 text-amber-700' :
            'bg-red-50 text-red-700'
          }`}>{joinPlan.base_dataset.confidence} confidence</span>
        </div>
        <div className="mt-1 text-zinc-500">{joinPlan.base_dataset.reason}</div>
        {joinPlan.base_dataset.covered_target_columns.length > 0 && (
          <div className="mt-1 text-zinc-400">
            Covers: {joinPlan.base_dataset.covered_target_columns.join(', ')}
          </div>
        )}
        {joinPlan.base_dataset.missing_target_columns.length > 0 && (
          <div className="mt-1 text-amber-600">
            Missing before joins: {joinPlan.base_dataset.missing_target_columns.join(', ')}
          </div>
        )}
      </div>

      {/* Selected joins */}
      {joinPlan.selected_joins.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-zinc-500">Selected joins</div>
          {joinPlan.selected_joins.map((join) => (
            <div key={join.step_id} className="rounded border border-zinc-200 p-3 text-xs">
              <div className="font-mono text-zinc-700">
                {join.left_dataset}.{join.left_column}
                <span className="mx-1 text-zinc-400">{join.join_type.toUpperCase()} JOIN</span>
                {join.right_dataset}.{join.right_column}
              </div>
              {join.columns_added.length > 0 && (
                <div className="mt-1 text-zinc-500">Adds: {join.columns_added.join(', ')}</div>
              )}
              <div className="mt-1 flex flex-wrap gap-3 text-zinc-500">
                <span>left coverage: {formatPct(join.evaluation.left_coverage)}</span>
                <span>right coverage: {formatPct(join.evaluation.right_coverage)}</span>
                <span>containment: {formatPct(join.evaluation.containment)}</span>
                <span>row multiplier: {formatMultiplier(join.evaluation.row_multiplier_left)}</span>
                <span>relationship: {join.evaluation.inferred_relationship}</span>
                <RiskBadge risk={join.evaluation.join_explosion_risk} />
              </div>
              <div className="mt-1 text-zinc-400">{join.reason}</div>
              {join.warnings.length > 0 && (
                <div className="mt-1 text-amber-600">⚠ {join.warnings.join(' · ')}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Rejected candidates */}
      {joinPlan.rejected_candidates.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-zinc-400 hover:text-zinc-600">
            {joinPlan.rejected_candidates.length} rejected candidate(s)
          </summary>
          <div className="mt-1 space-y-1 pl-2">
            {joinPlan.rejected_candidates.map((r) => (
              <div key={r.candidate_id} className="text-zinc-500">
                <span className="font-mono">{r.left_dataset}.{r.left_column} → {r.right_dataset}.{r.right_column}</span>
                {' — '}{r.reason}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Unresolved ambiguities */}
      {joinPlan.unresolved_ambiguities.length > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          ⚠ Unresolved ambiguities: {joinPlan.unresolved_ambiguities.join(' · ')}
        </div>
      )}

      {/* Global warnings */}
      {joinPlan.warnings.length > 0 && (
        <div className="rounded border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-600">
          {joinPlan.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}
    </div>
  )
}
