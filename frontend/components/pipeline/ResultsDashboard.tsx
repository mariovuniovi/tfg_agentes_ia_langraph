'use client'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useRunStore } from '@/stores/run-store'
import type { DataValidationInterrupt, AuditReportEventData, ExperienceSummary, MatchedRule, PlannerContextData, EvidenceReference } from '@/types/api'
import { useApprove } from '@/hooks/use-approve'
import { DatasetApprovalCard } from '@/components/pipeline/DatasetApprovalCard'
import { AuditReportPanel } from '@/components/pipeline/AuditReportPanel'
import { PlannerPanel, PlannerErrorBoundary } from '@/components/pipeline/PlannerPanel'

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


function TrainingCompletePanel({ data }: { data: { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null; selection_score?: number | null; validation_strategy?: { type?: string; n_folds?: number } | null } }) {
  const champ = data.champion_candidate as { model_key?: string; primary_metric?: string; primary_score?: number }
  const metricEntries = Object.entries(data.training_metrics ?? {})
  const vs = data.validation_strategy
  const vsLabel = vs?.type ? ` · via ${vs.type}${typeof vs.n_folds === 'number' ? `, ${vs.n_folds} fold${vs.n_folds === 1 ? '' : 's'}` : ''}` : ''
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-indigo-600 px-3 py-0.5 text-xs font-semibold text-white">
          {champ?.model_key ?? 'champion'}
        </span>
        {champ?.primary_metric && (
          <span className="text-xs text-zinc-500">
            {champ.primary_metric}: <span className="font-mono text-zinc-800">{Number(champ.primary_score).toFixed(4)}</span>
          </span>
        )}
        {data.training_run_id && (
          <span className="ml-auto font-mono text-[11px] text-zinc-400">{data.training_run_id.slice(0, 8)}…</span>
        )}
      </div>
      {metricEntries.length > 0 ? (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Test metrics (held-out)</p>
          <div className="grid grid-cols-2 gap-2">
            {metricEntries.map(([k, v]) => (
              <div key={k} className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs">
                <div className="text-zinc-500">{k}</div>
                <div className="font-mono text-zinc-800">{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
              </div>
            ))}
          </div>
          {typeof data.selection_score === 'number' && (
            <p className="mt-1.5 text-[11px] text-zinc-400">
              Selected on validation: <span className="font-mono">{data.selection_score.toFixed(4)}</span>{vsLabel}
            </p>
          )}
        </div>
      ) : (
        <div>
          <p className="text-xs text-amber-600">Test evaluation unavailable.</p>
          {typeof data.selection_score === 'number' && (
            <p className="mt-1 text-[11px] text-zinc-400">
              Selected on validation: <span className="font-mono">{data.selection_score.toFixed(4)}</span>{vsLabel}
            </p>
          )}
        </div>
      )}
      {data.forecast_chart_png && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Test forecast vs actuals</p>
          <img
            src={`data:image/png;base64,${data.forecast_chart_png}`}
            alt="Forecast chart"
            className="w-full rounded border border-zinc-200"
          />
        </div>
      )}
      {data.trained_model_path && (
        <p className="text-[11px] text-zinc-400">Artifact: <span className="font-mono">{data.trained_model_path}</span></p>
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
  const [tab, setTab] = useState<'dataset' | 'model' | 'planner' | 'audit'>('dataset')
  const [pinnedTab, setPinnedTab] = useState(false)
  const reviewPanelRef = useRef<HTMLDivElement>(null)
  const { approve, isPending } = useApprove(runId)

  const isDataValidationHITL = hitlPending && (interruptValue as { type?: string })?.type === 'data_validation'

  function selectTab(key: typeof tab) {
    setTab(key)
    setPinnedTab(true)
  }

  useEffect(() => {
    if (pinnedTab) return
    if (isDataValidationHITL) {
      setTab('dataset')
      setTimeout(() => reviewPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100)
    }
  }, [isDataValidationHITL, pinnedTab])

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

  const auditData = useMemo(() => {
    const ev = events.findLast((e) => e.type === 'audit_report')
    return ev ? (ev.data as unknown as AuditReportEventData) : null
  }, [events])

  const trainingData = useMemo(() => {
    const ev = events.findLast((e) => e.type === 'training_complete')
    return ev ? (ev.data as { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null; selection_score?: number | null; validation_strategy?: { type?: string; n_folds?: number } | null }) : null
  }, [events])

  const loadDatasetCallCount = useMemo(
    () => events.filter((e) => e.type === 'tool_call' && e.data.tool_name === 'load_dataset').length,
    [events],
  )
  // Only show the load_dataset result when exactly one file was loaded (single-file shortcut).
  // With multiple files, the last load_dataset result is just one raw source — not the final merged table.
  const dataset = loadDatasetCallCount === 1 ? toolResults['load_dataset'] as DatasetSummary | undefined : undefined
  // merge_datasets (classic) OR execute_join_plan (join discovery) both produce a merged summary
  const merged = (toolResults['merge_datasets'] ?? toolResults['execute_join_plan']) as MergedSummary | undefined
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
  const hasModel = !!(trained || tuned || trainingData)
  const hasPlanner = !!plannerCtx
  const running = status === 'running' || status === 'awaiting_approval'
  const active = running || hasDataset || hasModel

  useEffect(() => {
    if (pinnedTab) return
    if (plannerCtx) setTab('planner')
  }, [plannerCtx, pinnedTab])

  useEffect(() => {
    if (pinnedTab) return
    if (trained || tuned || trainingData) setTab('model')
  }, [trained, tuned, trainingData, pinnedTab])

  useEffect(() => {
    if (pinnedTab) return
    if (auditData) setTab('audit')
  }, [auditData, pinnedTab])

  useEffect(() => {
    setPinnedTab(false)
  }, [runId])

  if (!active) return null

  return (
    <div className="rounded-lg border border-zinc-200 bg-white">
      <div className="flex border-b border-zinc-200">
        {([
          { key: 'dataset', label: 'Dataset', ready: hasDataset },
          { key: 'planner', label: 'Planner', ready: hasPlanner },
          { key: 'model', label: 'Model', ready: hasModel },
          { key: 'audit', label: 'Audit', ready: !!auditData },
        ] as const).map(({ key, label, ready }) => (
          <button
            key={key}
            type="button"
            onClick={() => selectTab(key)}
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
              <DatasetApprovalCard
                runId={runId}
                interrupt={interruptValue as unknown as DataValidationInterrupt}
                onApprove={(decision, comment) => approve(decision, comment)}
                isPending={isPending}
                maxAttempts={3}
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
          <>
            {trained || tuned ? <ModelPanel trained={trained} tuned={tuned} running={running} /> : null}
            {trainingData && !trained && !tuned ? <TrainingCompletePanel data={trainingData} /> : null}
            {!trained && !tuned && !trainingData && <p className="text-xs text-zinc-400">No model results yet.</p>}
          </>
        )}
        {tab === 'audit' && auditData && <AuditReportPanel data={auditData} />}
        {tab === 'audit' && !auditData && <p className="text-xs text-zinc-400">Audit not yet generated.</p>}
      </div>
    </div>
  )
}
