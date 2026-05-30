'use client'
import { useEffect, useMemo, useRef, useState, Component, type ReactNode, type ErrorInfo } from 'react'
import { useRunStore } from '@/stores/run-store'
import type { DataValidationInterrupt } from '@/types/api'
import { useApprove } from '@/hooks/use-approve'

interface DatasetSummary {
  row_count: number
  column_count: number
  column_names: string[]
  dtypes: Record<string, string>
  head: Record<string, unknown>[]
}

interface MergedSummary {
  row_count: number
  columns: string[]
}

interface MissingSummary {
  total_rows: number
  columns_with_missing: number
  max_missing_pct: number
  per_column: Record<string, { count: number; pct: number }>
  passed_threshold: boolean
}

interface ValidationResult {
  passed: boolean
  violations?: Array<{ column: string; rule: string; detail: string }>
}

interface TuneResult {
  model_type: string
  best_params: Record<string, number | string>
  best_cv_f1: number
  n_trials: number
}

interface TrainResult {
  model_type: string
  train_accuracy: number
  val_accuracy: number
  classification_report: Record<string, unknown>
}

type ClassRow = { label: string; precision: number; recall: number; f1: number; support: number }

function parseClassReport(report: Record<string, unknown>): ClassRow[] {
  const skipKeys = new Set(['accuracy', 'macro avg', 'weighted avg'])
  return Object.entries(report)
    .filter(([k]) => !skipKeys.has(k))
    .map(([k, v]) => {
      const m = v as Record<string, number>
      return { label: k, precision: m.precision, recall: m.recall, f1: m['f1-score'], support: m.support }
    })
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded bg-zinc-50 p-3 text-center">
      <div className="text-xl font-semibold text-zinc-900">{value}</div>
      <div className="text-xs text-zinc-500">{label}</div>
      {sub && <div className="mt-0.5 text-xs text-zinc-400">{sub}</div>}
    </div>
  )
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

function DatasetPanel({
  dataset,
  merged,
  missing,
  validation,
  running,
  hideRawSample,
  merging,
}: {
  dataset?: DatasetSummary
  merged?: MergedSummary
  missing?: MissingSummary
  validation?: ValidationResult
  running: boolean
  hideRawSample?: boolean
  merging?: boolean
}) {
  if (merging && !merged) {
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-400">
        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-200 border-t-zinc-500" />
        Merging datasets…
      </div>
    )
  }
  if (!dataset && !merged && running) return <PulseRow />
  if (!dataset && !merged) return <p className="text-xs text-zinc-400">No dataset results yet.</p>

  const rows = dataset?.row_count ?? merged?.row_count ?? 0
  const cols = dataset ? dataset.column_names : merged?.columns ?? []
  const dtypes = dataset?.dtypes ?? {}
  const head = dataset?.head ?? []

  return (
    <div className="space-y-4">
      {/* Shape pill */}
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-indigo-600 px-3 py-0.5 text-xs font-semibold text-white">
          {rows.toLocaleString()} rows
        </span>
        <span className="rounded-full bg-zinc-100 px-3 py-0.5 text-xs font-semibold text-zinc-600">
          {cols.length} columns
        </span>
        {validation && (
          <span
            className={`ml-auto rounded-full px-3 py-0.5 text-xs font-semibold ${
              validation.passed
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-red-50 text-red-600'
            }`}
          >
            {validation.passed ? '✓ Valid' : `✗ ${(validation.violations ?? []).length} violation${(validation.violations ?? []).length !== 1 ? 's' : ''}`}
          </span>
        )}
      </div>

      {/* Column list */}
      <div>
        <p className="mb-1.5 text-xs font-medium text-zinc-500">Columns</p>
        <div className="flex flex-wrap gap-1">
          {cols.map((col) => (
            <span
              key={col}
              className="inline-flex items-center gap-1 rounded border border-zinc-200 bg-zinc-50 px-2 py-0.5 text-xs text-zinc-700"
            >
              {col}
              {dtypes[col] && (
                <span className="text-zinc-400">{dtypes[col].replace('object', 'str').replace('int64', 'int').replace('float64', 'float')}</span>
              )}
            </span>
          ))}
        </div>
      </div>

      {/* Missing values */}
      {missing && missing.columns_with_missing > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">
            Missing values — {missing.columns_with_missing} column{missing.columns_with_missing !== 1 ? 's' : ''} affected
          </p>
          <div className="space-y-1">
            {Object.entries(missing.per_column).map(([col, { pct }]) => (
              <div key={col} className="flex items-center gap-2">
                <span className="w-24 truncate text-xs text-zinc-600">{col}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-200">
                  <div
                    className={`h-full rounded-full ${pct > 20 ? 'bg-red-400' : 'bg-amber-400'}`}
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs text-zinc-400">{pct}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {missing && missing.columns_with_missing === 0 && (
        <p className="text-xs text-emerald-600">No missing values</p>
      )}

      {/* Sample rows — hidden when merged data is present (raw file is not the processed result) */}
      {head.length > 0 && !hideRawSample && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Sample (3 rows)</p>
          <div className="overflow-x-auto rounded border border-zinc-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-zinc-50">
                  {cols.map((col) => (
                    <th key={col} className="border-b border-zinc-200 px-2 py-1 text-left font-medium text-zinc-500">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {head.map((row, i) => (
                  <tr key={i} className="border-b border-zinc-100 last:border-0">
                    {cols.map((col) => (
                      <td key={col} className="px-2 py-1 text-zinc-600">
                        {String(row[col] ?? '—')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Validation violations */}
      {validation && !validation.passed && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-red-600">Violations</p>
          <ul className="space-y-1">
            {(validation.violations ?? []).map((v, i) => (
              <li key={i} className="rounded bg-red-50 px-2 py-1 text-xs text-red-700">
                <span className="font-medium">{v.column}</span> — {v.detail}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function ModelPanel({
  trained,
  tuned,
  running,
}: {
  trained?: TrainResult
  tuned?: TuneResult
  running: boolean
}) {
  if (!trained && !tuned && running) return <PulseRow />
  if (!trained && !tuned) return <p className="text-xs text-zinc-400">No model results yet.</p>

  const modelType = trained?.model_type ?? tuned?.model_type ?? ''
  const classRows = trained ? parseClassReport(trained.classification_report as Record<string, unknown>) : []

  return (
    <div className="space-y-4">
      {/* Model type badge */}
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-indigo-600 px-3 py-0.5 text-xs font-semibold text-white">
          {modelType.replace(/_/g, ' ')}
        </span>
        {tuned && (
          <span className="text-xs text-zinc-400">
            {tuned.n_trials} Optuna trials
          </span>
        )}
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-2">
        {tuned && (
          <StatCard label="CV F1" value={tuned.best_cv_f1.toFixed(3)} sub="cross-val" />
        )}
        {trained && (
          <>
            <StatCard label="Train Acc" value={`${(trained.train_accuracy * 100).toFixed(1)}%`} />
            <StatCard label="Val Acc" value={`${(trained.val_accuracy * 100).toFixed(1)}%`} />
          </>
        )}
      </div>

      {/* Best hyperparams */}
      {tuned && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Best hyperparameters</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(tuned.best_params).map(([k, v]) => (
              <span key={k} className="rounded border border-zinc-200 bg-zinc-50 px-2 py-0.5 text-xs text-zinc-700">
                <span className="text-zinc-400">{k}:</span> {String(v)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Classification report */}
      {classRows.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Per-class metrics</p>
          <div className="overflow-x-auto rounded border border-zinc-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-zinc-50">
                  <th className="border-b border-zinc-200 px-2 py-1 text-left font-medium text-zinc-500">Class</th>
                  <th className="border-b border-zinc-200 px-2 py-1 text-right font-medium text-zinc-500">Prec</th>
                  <th className="border-b border-zinc-200 px-2 py-1 text-right font-medium text-zinc-500">Recall</th>
                  <th className="border-b border-zinc-200 px-2 py-1 text-right font-medium text-zinc-500">F1</th>
                  <th className="border-b border-zinc-200 px-2 py-1 text-right font-medium text-zinc-500">N</th>
                </tr>
              </thead>
              <tbody>
                {classRows.map((r) => (
                  <tr key={r.label} className="border-b border-zinc-100 last:border-0">
                    <td className="px-2 py-1 font-medium text-zinc-700">{r.label}</td>
                    <td className="px-2 py-1 text-right text-zinc-600">{r.precision.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right text-zinc-600">{r.recall.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right font-medium text-zinc-900">{r.f1.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right text-zinc-400">{r.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function DatasetReviewPanel({
  runId,
  interruptValue,
  panelRef,
}: {
  runId: string | null
  interruptValue: DataValidationInterrupt
  panelRef?: React.RefObject<HTMLDivElement | null>
}) {
  const commentRef = useRef<HTMLTextAreaElement>(null)
  const { approve, isPending } = useApprove(runId)
  const maxAttempts = 3
  const attempt = interruptValue.attempt ?? 1
  const preview = interruptValue.dataset_preview
  const sampleRows = preview?.sample_rows ?? []
  const sampleCols = preview?.columns ?? []

  return (
    <div ref={panelRef} className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-semibold text-blue-900">Dataset Review</span>
        <span className="rounded bg-blue-100 px-2 py-0.5 text-xs text-blue-700">
          awaiting approval
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-xs text-zinc-400">
          Attempt {attempt} of {maxAttempts}
          {Array.from({ length: maxAttempts }).map((_, i) => (
            <span
              key={i}
              className={`inline-block h-2 w-2 rounded-full ${
                i < attempt ? 'bg-amber-400' : 'bg-zinc-200'
              }`}
            />
          ))}
        </span>
      </div>
      <p className="mb-3 text-xs text-zinc-500">
        Approve to proceed to training, or reject with a comment so the data agent can fix
        the issue and reprocess.
      </p>

      {sampleRows.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-xs font-medium text-zinc-500">
            Processed data preview ({preview.shape[0]} rows × {preview.shape[1]} cols)
          </p>
          <div className="overflow-x-auto rounded border border-blue-100">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-blue-100">
                  {sampleCols.map((c) => (
                    <th key={c.name} className="border-b border-blue-200 px-2 py-1 text-left font-medium text-zinc-600">
                      {c.name}
                      <span className="ml-1 text-zinc-400">{c.dtype}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sampleRows.slice(0, 5).map((row, i) => (
                  <tr key={i} className="border-b border-blue-100 last:border-0">
                    {sampleCols.map((c) => (
                      <td key={c.name} className={`px-2 py-1 ${row[c.name] == null ? 'text-red-400 italic' : 'text-zinc-700'}`}>
                        {row[c.name] == null ? 'null' : String(row[c.name])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <label className="mb-1 block text-xs font-medium text-zinc-500">
        Comment (optional)
      </label>
      <textarea
        ref={commentRef}
        rows={2}
        placeholder="e.g. rename column X, drop rows where value < 0…"
        className="mb-3 w-full rounded border border-zinc-200 bg-white px-2 py-1.5 text-xs text-zinc-700 placeholder-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-300"
      />
      <div className="flex gap-2">
        <button
          onClick={() => approve('approve', '')}
          disabled={isPending}
          className="rounded bg-emerald-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
        >
          ✓ Approve dataset
        </button>
        <button
          onClick={() => approve('reject', commentRef.current?.value ?? '')}
          disabled={isPending}
          className="rounded border border-red-200 bg-red-50 px-4 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
        >
          ✗ Reject &amp; retry
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Planner evidence types
// ---------------------------------------------------------------------------

interface ExperienceSummary {
  experience_id: string
  dataset_name: string
  problem_type: string
  best_model: string
  validation_score: number
  metric_name?: string
}

interface MatchedRule {
  rule_id: string
  prefer?: string[]
  avoid_or_deprioritize?: string[]
  recommend?: string
  summary: string
}

interface EvidenceRef {
  evidence_type: 'experience' | 'rule'
  experience_id?: string
  rule_id?: string
  relevance_note: string
}

interface PlannerContextData {
  retrieved_experiences: ExperienceSummary[]
  matched_rules: MatchedRule[]
  evidence_used: EvidenceRef[]
  planning_analysis: string
  plan_summary: { candidate_models: string[]; models_not_recommended: string[] }
  warnings: string[]
}

class PlannerErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
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

function PlannerPanel({ ctx, running }: { ctx: PlannerContextData | null; running: boolean }) {
  if (!ctx && running) return <PulseRow />
  if (!ctx) return <p className="text-xs text-zinc-400">Planner has not run yet.</p>

  const citedExpIds = new Set(ctx.evidence_used.filter(e => e.evidence_type === 'experience').map(e => e.experience_id))
  const citedRuleIds = new Set(ctx.evidence_used.filter(e => e.evidence_type === 'rule').map(e => e.rule_id))

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
                    {cited && ctx.evidence_used.find(e => e.experience_id === exp.experience_id)?.relevance_note && (
                      <div className="mt-1 italic text-violet-600">
                        {ctx.evidence_used.find(e => e.experience_id === exp.experience_id)!.relevance_note}
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

export function ResultsDashboard() {
  const events = useRunStore((s) => s.events)
  const status = useRunStore((s) => s.status)
  const runId = useRunStore((s) => s.runId)
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const [tab, setTab] = useState<'dataset' | 'model' | 'planner'>('dataset')
  const reviewPanelRef = useRef<HTMLDivElement>(null)

  const isDataValidationHITL = hitlPending && (interruptValue as { type?: string })?.type === 'data_validation'

  useEffect(() => {
    if (isDataValidationHITL) {
      setTab('dataset')
      setTimeout(() => reviewPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100)
    }
  }, [isDataValidationHITL])

  const toolResults = useMemo(() => {
    const results: Record<string, unknown> = {}
    for (const event of events) {
      if (event.type === 'tool_result' && typeof event.data.result === 'string') {
        try {
          results[event.data.tool_name as string] = JSON.parse(event.data.result as string)
        } catch {
          // ignore malformed JSON
        }
      }
    }
    return results
  }, [events])

  const plannerCtx = useMemo(() => {
    const ev = events.findLast((e) => e.type === 'planner_context')
    return ev ? (ev.data as unknown as PlannerContextData) : null
  }, [events])

  const dataset = toolResults['load_dataset'] as DatasetSummary | undefined
  const merged = toolResults['merge_datasets'] as MergedSummary | undefined
  const missing = toolResults['check_missing_values'] as MissingSummary | undefined
  const validation = toolResults['validate_against_schema'] as ValidationResult | undefined
  const trained = toolResults['train_model'] as TrainResult | undefined
  const tuned = toolResults['tune_hyperparameters'] as TuneResult | undefined

  const mergeWasStarted = useMemo(
    () => events.some((e) => e.type === 'tool_call' && e.data.tool_name === 'merge_datasets'),
    [events],
  )
  const isMerging = mergeWasStarted && !merged

  const hasDataset = !!dataset || !!merged || isMerging
  const hasModel = !!trained || !!tuned
  const hasPlanner = !!plannerCtx
  const running = status === 'running' || status === 'awaiting_approval'
  const active = running || hasDataset || hasModel

  if (!active) return null

  return (
    <div className="rounded-lg border border-zinc-200 bg-white">
      <div className="flex border-b border-zinc-200">
        {([
          { key: 'dataset', label: 'Dataset', ready: hasDataset },
          { key: 'planner', label: 'Planner', ready: hasPlanner },
          { key: 'model', label: 'Model', ready: hasModel },
        ] as const).map(({ key, label, ready }) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
              tab === key
                ? 'border-b-2 border-indigo-600 text-zinc-900'
                : 'text-zinc-400 hover:text-zinc-600'
            }`}
          >
            {label}
            {ready && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />}
            {!ready && running && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-zinc-300" />}
          </button>
        ))}
      </div>
      <div className="p-4">
        {tab === 'dataset' && (
          <>
            <DatasetPanel
              dataset={mergeWasStarted ? undefined : dataset}
              merged={merged}
              missing={missing}
              validation={validation}
              running={running}
              hideRawSample={isDataValidationHITL && !!merged}
              merging={isMerging}
            />
            {isDataValidationHITL && (
              <DatasetReviewPanel
                runId={runId}
                interruptValue={interruptValue as unknown as DataValidationInterrupt}
                panelRef={reviewPanelRef}
              />
            )}
          </>
        )}
        {tab === 'planner' && (
          <PlannerErrorBoundary>
            <PlannerPanel ctx={plannerCtx} running={running} />
          </PlannerErrorBoundary>
        )}
        {tab === 'model' && (
          <ModelPanel trained={trained} tuned={tuned} running={running} />
        )}
      </div>
    </div>
  )
}
