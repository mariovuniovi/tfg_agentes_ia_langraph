'use client'
import { Component, type ReactNode, type ErrorInfo } from 'react'
import type { PlannerContextData } from '@/types/api'

// Legacy v1 shape still used by PlannerPanel — Phase 8 will migrate to EvidenceReference
interface EvidenceRef {
  evidence_type: 'experience' | 'rule'
  experience_id?: string
  rule_id?: string
  relevance_note: string
}

function PulseRow() {
  return (
    <div className="animate-pulse space-y-2">
      <div className="h-3 w-3/4 rounded bg-zinc-200" />
      <div className="h-3 w-1/2 rounded bg-zinc-200" />
      <div className="h-3 w-2/3 rounded bg-zinc-200" />
    </div>
  )
}

export class PlannerErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[PlannerPanel] render error:', error, info)
  }
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

export function PlannerPanel({ ctx, running }: { ctx: PlannerContextData | null; running: boolean }) {
  if (!ctx && running) return <PulseRow />
  if (!ctx) return <p className="text-xs text-zinc-400">Planner has not run yet.</p>

  // evidence_used is EvidenceReference[] (v2); cast to v1 shape for legacy rendering — Phase 8 will migrate
  const evidenceV1 = (ctx.evidence_used as unknown as EvidenceRef[])
  const citedExpIds = new Set(evidenceV1.filter(e => e.evidence_type === 'experience').map(e => e.experience_id))
  const citedRuleIds = new Set(evidenceV1.filter(e => e.evidence_type === 'rule').map(e => e.rule_id))

  return (
    <div className="space-y-5">
      {/* Planning analysis */}
      {ctx.planning_analysis && (
        <div>
          <p className="mb-1.5 text-xs font-semibold text-zinc-600">Planning Analysis</p>
          <p className="text-xs leading-relaxed text-zinc-600">{ctx.planning_analysis}</p>
        </div>
      )}

      {/* Candidate models */}
      {ctx.plan_summary?.candidate_models?.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold text-zinc-600">Selected Candidates</p>
          <div className="flex flex-wrap gap-1">
            {ctx.plan_summary.candidate_models.map((m) => (
              <span key={m} className="rounded-full bg-violet-50 px-2.5 py-0.5 text-xs font-medium text-violet-700">{m}</span>
            ))}
          </div>
          {ctx.plan_summary.models_not_recommended?.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              {ctx.plan_summary.models_not_recommended.map((m) => (
                <span key={m} className="rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-500 line-through">{m}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Similar experiences */}
      <div>
        <p className="mb-1.5 text-xs font-semibold text-zinc-600">
          Similar Past Runs <span className="font-normal text-zinc-400">({ctx.retrieved_experiences.length} retrieved)</span>
        </p>
        {ctx.retrieved_experiences.length === 0
          ? <p className="text-xs text-zinc-400">No similar experiences in the pool.</p>
          : (
            <div className="space-y-1.5">
              {ctx.retrieved_experiences.map((exp) => {
                const cited = citedExpIds.has(exp.experience_id)
                return (
                  <div key={exp.experience_id} className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-100 bg-zinc-50'}`}>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-700">{exp.dataset_name}</span>
                      <span className="text-zinc-400">{exp.problem_type}</span>
                      {cited && <span className="ml-auto rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">cited</span>}
                    </div>
                    <div className="mt-0.5 text-zinc-500">
                      Best model: <span className="font-medium text-zinc-700">{exp.best_model}</span>
                      {' · '}<span className="uppercase">{exp.metric_name ?? 'score'}</span>: <span className="font-medium text-zinc-700">{(exp.validation_score ?? 0).toFixed(4)}</span>
                    </div>
                    {cited && evidenceV1.find(e => e.experience_id === exp.experience_id)?.relevance_note && (
                      <div className="mt-1 italic text-violet-600">
                        {evidenceV1.find(e => e.experience_id === exp.experience_id)!.relevance_note}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )
        }
      </div>

      {/* Matched rules */}
      <div>
        <p className="mb-1.5 text-xs font-semibold text-zinc-600">
          ML Rules Matched <span className="font-normal text-zinc-400">({ctx.matched_rules.length} rules)</span>
        </p>
        {ctx.matched_rules.length === 0
          ? <p className="text-xs text-zinc-400">No rules matched.</p>
          : (
            <div className="space-y-1.5">
              {ctx.matched_rules.map((rule) => {
                const cited = citedRuleIds.has(rule.rule_id)
                return (
                  <div key={rule.rule_id} className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-100 bg-zinc-50'}`}>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-zinc-400">{rule.rule_id}</span>
                      {cited && <span className="ml-auto rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">cited</span>}
                    </div>
                    <p className="mt-0.5 text-zinc-600">{rule.summary}</p>
                    {rule.prefer && rule.prefer.length > 0 && (
                      <p className="mt-0.5 text-emerald-700">↑ prefer: {rule.prefer.join(', ')}</p>
                    )}
                    {rule.avoid_or_deprioritize && rule.avoid_or_deprioritize.length > 0 && (
                      <p className="mt-0.5 text-red-600">↓ avoid: {rule.avoid_or_deprioritize.join(', ')}</p>
                    )}
                    {rule.recommend && (
                      <p className="mt-0.5 text-zinc-500 italic">{rule.recommend}</p>
                    )}
                  </div>
                )
              })}
            </div>
          )
        }
      </div>

      {/* Warnings */}
      {ctx.warnings?.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold text-amber-700">Planner Warnings</p>
          <ul className="space-y-0.5">
            {ctx.warnings.map((w, i) => (
              <li key={i} className="text-xs text-amber-700">⚠ {w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
