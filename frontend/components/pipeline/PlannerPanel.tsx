'use client'
import { Component, type ReactNode, type ErrorInfo } from 'react'
import type { PlannerContextData } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { PlannerSummaryHeader } from './planner/PlannerSummaryHeader'
import { DecisionBasisCard } from './planner/DecisionBasisCard'
import { ConflictPanel } from './planner/ConflictPanel'
import { CandidateCard } from './planner/CandidateCard'
import { RejectedModelCard } from './planner/RejectedModelCard'
import { EvidenceQualityCard } from './planner/EvidenceQualityCard'
import { ExperienceCard } from './planner/ExperienceCard'

export class PlannerErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  constructor(props: { children: ReactNode }) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('[PlannerPanel] render error:', error, info) }
  render() {
    if (this.state.error) {
      return (
        <div className="rounded border border-red-200 bg-red-50 p-4 text-xs text-red-700">
          <p className="font-semibold">Planner panel failed to render</p>
          <p className="mt-1 font-mono opacity-70">{this.state.error.message}</p>
        </div>
      )
    }
    return this.props.children
  }
}

interface Props {
  ctx: PlannerContextData | null
  running: boolean
}

export function PlannerPanel({ ctx, running }: Props) {
  if (!ctx && running) {
    return <p className="text-xs text-zinc-400">Planner is running…</p>
  }
  if (!ctx) {
    return <p className="text-xs text-zinc-400">Planner has not run yet.</p>
  }

  const candidates = ctx.plan_summary.candidate_rationales ?? []
  const rejected = ctx.plan_summary.rejected_model_rationales ?? []
  const sortedCandidates = [...candidates].sort((a, b) => a.priority - b.priority)
  const sortedExperiences = [...ctx.retrieved_experiences].sort((a, b) => b.similarity_score - a.similarity_score)
  const citedSet = new Set(ctx.cited_experience_ids ?? [])
  const citedRules = new Set(ctx.cited_rule_ids ?? [])

  return (
    <div className="space-y-4">
      <PlannerSummaryHeader candidates={sortedCandidates} rejected={rejected} status={ctx.planner_status} />
      {ctx.decision_basis && <DecisionBasisCard basis={ctx.decision_basis} />}
      <ConflictPanel hard={ctx.evidence_conflicts ?? []} soft={ctx.soft_conflicts ?? []} />

      {sortedCandidates.length > 0 && (
        <Card title={`Candidate rationale (${sortedCandidates.length})`}>
          <div className="space-y-2">
            {sortedCandidates.map((c) => <CandidateCard key={c.model_key} candidate={c} />)}
          </div>
        </Card>
      )}

      {rejected.length > 0 && (
        <Card title={`Rejected models (${rejected.length})`}>
          <div className="space-y-1">
            {rejected.map((r) => <RejectedModelCard key={r.model_key} rejected={r} />)}
          </div>
        </Card>
      )}

      <EvidenceQualityCard experiences={ctx.retrieved_experiences} citedIds={ctx.cited_experience_ids ?? []} />

      {sortedExperiences.length > 0 && (
        <Card title={`Similar past runs (${sortedExperiences.length})`}>
          <div className="space-y-2">
            {sortedExperiences.map((e) => (
              <ExperienceCard key={e.experience_id} exp={e} cited={citedSet.has(e.experience_id)} />
            ))}
          </div>
        </Card>
      )}

      {ctx.matched_rules.length > 0 && (
        <Card title={`ML rules matched (${ctx.matched_rules.length})`}>
          <div className="space-y-1.5">
            {ctx.matched_rules.map((rule) => {
              const cited = citedRules.has(rule.rule_id)
              return (
                <div key={rule.rule_id} className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-200 bg-white'}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-zinc-500">{rule.rule_id}</span>
                    {cited && <span className="ml-auto rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">cited</span>}
                  </div>
                  <p className="mt-0.5 text-zinc-700">{rule.summary}</p>
                  {rule.prefer && rule.prefer.length > 0 && (
                    <p className="mt-0.5 text-emerald-700">↑ prefer: {rule.prefer.join(', ')}</p>
                  )}
                  {rule.avoid_or_deprioritize && rule.avoid_or_deprioritize.length > 0 && (
                    <p className="mt-0.5 text-red-600">↓ avoid: {rule.avoid_or_deprioritize.join(', ')}</p>
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {ctx.warnings.length > 0 && (
        <Card title="Planner warnings">
          <ul className="space-y-0.5 text-xs">
            {ctx.warnings.map((w, i) => (
              <li key={i} className="text-amber-700">⚠ {w}</li>
            ))}
          </ul>
        </Card>
      )}

      <details className="rounded border border-[var(--color-border)] bg-white p-3">
        <summary className="cursor-pointer text-xs text-zinc-500">View full planning analysis</summary>
        <pre className="mt-2 whitespace-pre-wrap text-xs text-zinc-700">{ctx.planning_analysis}</pre>
      </details>
    </div>
  )
}
