'use client'
import { useState } from 'react'
import { toast } from 'sonner'
import type { DataValidationInterrupt } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { JoinPlanPanel } from './JoinPlanPanel'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

type Tab = 'head' | 'tail' | 'schema' | 'validation' | 'join_plan'

interface Props {
  runId: string | null
  interrupt: DataValidationInterrupt
  onApprove: (decision: 'approve' | 'reject', comment: string) => void
  isPending: boolean
  maxAttempts: number
}

export function DatasetApprovalCard({ runId, interrupt, onApprove, isPending, maxAttempts }: Props) {
  const [tab, setTab] = useState<Tab>('head')
  const [comment, setComment] = useState('')
  const preview = interrupt.dataset_preview
  const hasTail = (preview.tail?.length ?? 0) > 0
  const attempt = interrupt.attempt ?? 1
  const rejectDisabled = isPending || comment.trim().length < 4

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'head', label: 'Head' },
    ...(hasTail ? [{ key: 'tail' as const, label: 'Tail' }] : []),
    { key: 'schema', label: 'Schema' },
    { key: 'validation', label: 'Validation report' },
    ...(interrupt.join_plan !== undefined ? [{ key: 'join_plan' as const, label: 'Join plan' }] : []),
  ]

  const rows =
    tab === 'head' ? preview.head ?? preview.sample_rows ?? []
    : tab === 'tail' ? preview.tail ?? []
    : []
  const cols = preview.columns

  return (
    <Card
      title="Dataset approval required"
      actions={
        <span className="text-xs text-[var(--color-warning)]">
          Attempt {attempt} of {maxAttempts}
        </span>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-[var(--color-fg-muted)]">
        <span className="font-mono text-[var(--color-fg)]">{preview.path.split('/').pop()}</span>
        <button
          type="button"
          className="rounded border border-[var(--color-border)] px-2 py-0.5 hover:bg-zinc-50"
          onClick={() => { navigator.clipboard.writeText(preview.path); toast.success('Path copied') }}
        >
          Copy artifact path
        </button>
        <span>{preview.row_count} rows · {preview.column_count} columns</span>
        {interrupt.validation_report && (
          <Badge variant={(interrupt.validation_report as { passed?: boolean }).passed ? 'success' : 'danger'}>
            validation {(interrupt.validation_report as { passed?: boolean }).passed ? '✓ passed' : '✗ failed'}
          </Badge>
        )}
      </div>

      <div className="mb-2 flex gap-1 border-b border-[var(--color-border)]">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 text-xs font-medium ${
              tab === t.key
                ? 'border-b-2 border-indigo-600 text-indigo-700'
                : 'text-[var(--color-fg-muted)] hover:text-[var(--color-fg)]'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {(tab === 'head' || tab === 'tail') && (
        <div className="overflow-x-auto rounded border border-[var(--color-border)]">
          <table className="w-full text-xs">
            <thead className="bg-zinc-50">
              <tr>{cols.map((c) => (<th key={c.name} className="border-b border-[var(--color-border)] px-2 py-1 text-left font-medium text-zinc-500">{c.name}<span className="ml-1 text-zinc-400">{c.dtype}</span></th>))}</tr>
            </thead>
            <tbody>
              {rows.slice(0, 5).map((row, i) => (
                <tr key={i} className="border-b border-zinc-100 last:border-0">
                  {cols.map((c) => (
                    <td key={c.name} className={`px-2 py-1 ${(row as Record<string, unknown>)[c.name] == null ? 'italic text-red-400' : 'text-zinc-700'}`}>
                      {(row as Record<string, unknown>)[c.name] == null ? 'null' : String((row as Record<string, unknown>)[c.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'schema' && (
        <table className="w-full text-xs">
          <thead className="bg-zinc-50">
            <tr>
              <th className="px-2 py-1 text-left font-medium text-zinc-500">name</th>
              <th className="px-2 py-1 text-left font-medium text-zinc-500">dtype</th>
            </tr>
          </thead>
          <tbody>
            {cols.map((c) => (
              <tr key={c.name} className="border-t border-zinc-100">
                <td className="px-2 py-1 font-mono text-zinc-700">{c.name}</td>
                <td className="px-2 py-1 text-zinc-500">{c.dtype}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {tab === 'validation' && (
        <pre className="overflow-x-auto rounded bg-zinc-50 p-2 font-mono text-xs text-zinc-700">
          {JSON.stringify(interrupt.validation_report ?? {}, null, 2)}
        </pre>
      )}

      {tab === 'join_plan' && (
        <JoinPlanPanel joinPlan={interrupt.join_plan} joinBaseNrows={interrupt.join_base_nrows} />
      )}

      <label htmlFor="reject-comment" className="mt-4 mb-1 block text-xs font-medium text-zinc-500">
        Comment <span className="text-zinc-400">(required to reject, ≥ 4 chars)</span>
      </label>
      <textarea
        id="reject-comment"
        rows={2}
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="e.g. rename column X, drop rows where value < 0…"
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-white px-2 py-1.5 text-xs text-zinc-700 placeholder-zinc-300 focus:outline-none focus:ring-1 focus:ring-indigo-300"
      />

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onApprove('approve', '')}
          disabled={isPending}
          className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
        >
          ✓ Approve dataset
        </button>
        <button
          type="button"
          onClick={() => onApprove('reject', comment)}
          disabled={rejectDisabled}
          className="rounded border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
        >
          ✗ Reject &amp; retry
        </button>
        <a
          href={`${API_BASE}/runs/${runId}/dataset-download`}
          className="rounded border border-[var(--color-border)] px-3 py-1.5 text-xs font-medium text-[var(--color-fg-muted)] hover:bg-zinc-50"
        >
          Download CSV ↓
        </a>
      </div>
    </Card>
  )
}
