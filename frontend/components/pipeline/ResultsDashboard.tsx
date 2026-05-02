'use client'
import { useEffect, useMemo, useRef, useState } from 'react'
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
  violations: Array<{ column: string; rule: string; detail: string }>
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
    <div className="rounded bg-slate-50 p-3 text-center">
      <div className="text-xl font-semibold text-navy-900">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-400">{sub}</div>}
    </div>
  )
}

function PulseRow() {
  return (
    <div className="animate-pulse space-y-2">
      <div className="h-3 w-3/4 rounded bg-slate-200" />
      <div className="h-3 w-1/2 rounded bg-slate-200" />
      <div className="h-3 w-2/3 rounded bg-slate-200" />
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
}: {
  dataset?: DatasetSummary
  merged?: MergedSummary
  missing?: MissingSummary
  validation?: ValidationResult
  running: boolean
  hideRawSample?: boolean
}) {
  if (!dataset && !merged && running) return <PulseRow />
  if (!dataset && !merged) return <p className="text-xs text-slate-400">No dataset results yet.</p>

  const rows = dataset?.row_count ?? merged?.row_count ?? 0
  const cols = dataset ? dataset.column_names : merged?.columns ?? []
  const dtypes = dataset?.dtypes ?? {}
  const head = dataset?.head ?? []

  return (
    <div className="space-y-4">
      {/* Shape pill */}
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-navy px-3 py-0.5 text-xs font-semibold text-white">
          {rows.toLocaleString()} rows
        </span>
        <span className="rounded-full bg-slate-100 px-3 py-0.5 text-xs font-semibold text-slate-600">
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
            {validation.passed ? '✓ Valid' : `✗ ${validation.violations.length} violation${validation.violations.length !== 1 ? 's' : ''}`}
          </span>
        )}
      </div>

      {/* Column list */}
      <div>
        <p className="mb-1.5 text-xs font-medium text-slate-500">Columns</p>
        <div className="flex flex-wrap gap-1">
          {cols.map((col) => (
            <span
              key={col}
              className="inline-flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-700"
            >
              {col}
              {dtypes[col] && (
                <span className="text-slate-400">{dtypes[col].replace('object', 'str').replace('int64', 'int').replace('float64', 'float')}</span>
              )}
            </span>
          ))}
        </div>
      </div>

      {/* Missing values */}
      {missing && missing.columns_with_missing > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-slate-500">
            Missing values — {missing.columns_with_missing} column{missing.columns_with_missing !== 1 ? 's' : ''} affected
          </p>
          <div className="space-y-1">
            {Object.entries(missing.per_column).map(([col, { pct }]) => (
              <div key={col} className="flex items-center gap-2">
                <span className="w-24 truncate text-xs text-slate-600">{col}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className={`h-full rounded-full ${pct > 20 ? 'bg-red-400' : 'bg-amber-400'}`}
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs text-slate-400">{pct}%</span>
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
          <p className="mb-1.5 text-xs font-medium text-slate-500">Sample (3 rows)</p>
          <div className="overflow-x-auto rounded border border-slate-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50">
                  {cols.map((col) => (
                    <th key={col} className="border-b border-slate-200 px-2 py-1 text-left font-medium text-slate-500">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {head.map((row, i) => (
                  <tr key={i} className="border-b border-slate-100 last:border-0">
                    {cols.map((col) => (
                      <td key={col} className="px-2 py-1 text-slate-600">
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
            {validation.violations.map((v, i) => (
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
  if (!trained && !tuned) return <p className="text-xs text-slate-400">No model results yet.</p>

  const modelType = trained?.model_type ?? tuned?.model_type ?? ''
  const classRows = trained ? parseClassReport(trained.classification_report as Record<string, unknown>) : []

  return (
    <div className="space-y-4">
      {/* Model type badge */}
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-navy px-3 py-0.5 text-xs font-semibold text-white">
          {modelType.replace(/_/g, ' ')}
        </span>
        {tuned && (
          <span className="text-xs text-slate-400">
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
          <p className="mb-1.5 text-xs font-medium text-slate-500">Best hyperparameters</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(tuned.best_params).map(([k, v]) => (
              <span key={k} className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-700">
                <span className="text-slate-400">{k}:</span> {String(v)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Classification report */}
      {classRows.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-slate-500">Per-class metrics</p>
          <div className="overflow-x-auto rounded border border-slate-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50">
                  <th className="border-b border-slate-200 px-2 py-1 text-left font-medium text-slate-500">Class</th>
                  <th className="border-b border-slate-200 px-2 py-1 text-right font-medium text-slate-500">Prec</th>
                  <th className="border-b border-slate-200 px-2 py-1 text-right font-medium text-slate-500">Recall</th>
                  <th className="border-b border-slate-200 px-2 py-1 text-right font-medium text-slate-500">F1</th>
                  <th className="border-b border-slate-200 px-2 py-1 text-right font-medium text-slate-500">N</th>
                </tr>
              </thead>
              <tbody>
                {classRows.map((r) => (
                  <tr key={r.label} className="border-b border-slate-100 last:border-0">
                    <td className="px-2 py-1 font-medium text-slate-700">{r.label}</td>
                    <td className="px-2 py-1 text-right text-slate-600">{r.precision.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right text-slate-600">{r.recall.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right font-medium text-navy-900">{r.f1.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right text-slate-400">{r.support}</td>
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
        <span className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
          Attempt {attempt} of {maxAttempts}
          {Array.from({ length: maxAttempts }).map((_, i) => (
            <span
              key={i}
              className={`inline-block h-2 w-2 rounded-full ${
                i < attempt ? 'bg-amber-400' : 'bg-slate-200'
              }`}
            />
          ))}
        </span>
      </div>
      <p className="mb-3 text-xs text-slate-500">
        Approve to proceed to training, or reject with a comment so the data agent can fix
        the issue and reprocess.
      </p>

      {sampleRows.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-xs font-medium text-slate-500">
            Processed data preview ({preview.shape[0]} rows × {preview.shape[1]} cols)
          </p>
          <div className="overflow-x-auto rounded border border-blue-100">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-blue-100">
                  {sampleCols.map((c) => (
                    <th key={c.name} className="border-b border-blue-200 px-2 py-1 text-left font-medium text-slate-600">
                      {c.name}
                      <span className="ml-1 text-slate-400">{c.dtype}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sampleRows.slice(0, 5).map((row, i) => (
                  <tr key={i} className="border-b border-blue-100 last:border-0">
                    {sampleCols.map((c) => (
                      <td key={c.name} className={`px-2 py-1 ${row[c.name] == null ? 'text-red-400 italic' : 'text-slate-700'}`}>
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

      <label className="mb-1 block text-xs font-medium text-slate-500">
        Comment (optional)
      </label>
      <textarea
        ref={commentRef}
        rows={2}
        placeholder="e.g. rename column X, drop rows where value < 0…"
        className="mb-3 w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-1 focus:ring-blue-300"
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

export function ResultsDashboard() {
  const events = useRunStore((s) => s.events)
  const status = useRunStore((s) => s.status)
  const runId = useRunStore((s) => s.runId)
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const [tab, setTab] = useState<'dataset' | 'model'>('dataset')
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

  const dataset = toolResults['load_dataset'] as DatasetSummary | undefined
  const merged = toolResults['merge_datasets'] as MergedSummary | undefined
  const missing = toolResults['check_missing_values'] as MissingSummary | undefined
  const validation = toolResults['validate_against_schema'] as ValidationResult | undefined
  const trained = toolResults['train_model'] as TrainResult | undefined
  const tuned = toolResults['tune_hyperparameters'] as TuneResult | undefined

  const hasDataset = !!dataset || !!merged
  const hasModel = !!trained || !!tuned
  const running = status === 'running' || status === 'awaiting_approval'
  const active = running || hasDataset || hasModel

  if (!active) return null

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="flex border-b border-slate-200">
        {(['dataset', 'model'] as const).map((t) => {
          const ready = t === 'dataset' ? hasDataset : hasModel
          return (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors ${
                tab === t
                  ? 'border-b-2 border-navy text-navy-900'
                  : 'text-slate-400 hover:text-slate-600'
              }`}
            >
              {t === 'dataset' ? 'Dataset' : 'Model'}
              {ready && (
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              )}
              {!ready && running && (
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-300" />
              )}
            </button>
          )
        })}
      </div>
      <div className="p-4">
        {tab === 'dataset' && (
          <>
            <DatasetPanel
              dataset={dataset}
              merged={merged}
              missing={missing}
              validation={validation}
              running={running}
              hideRawSample={isDataValidationHITL && !!merged}
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
        {tab === 'model' && (
          <ModelPanel trained={trained} tuned={tuned} running={running} />
        )}
      </div>
    </div>
  )
}
