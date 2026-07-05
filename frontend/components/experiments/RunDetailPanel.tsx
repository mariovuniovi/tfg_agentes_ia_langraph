'use client'
import type { RunOut } from '@/types/api'
import { formatMetricValue, formatRunTime, buildRunCsv } from '@/lib/format'

const STATUS_BADGE: Record<string, { label: string; className: string }> = {
  FINISHED: { label: 'complete', className: 'bg-emerald-50 text-emerald-700' },
  FAILED: { label: 'failed', className: 'bg-red-50 text-red-600' },
  KILLED: { label: 'killed', className: 'bg-red-50 text-red-600' },
  RUNNING: { label: 'running', className: 'bg-amber-50 text-amber-700' },
  SCHEDULED: { label: 'scheduled', className: 'bg-amber-50 text-amber-700' },
}

function statusBadge(status: string): { label: string; className: string } {
  return STATUS_BADGE[status] ?? { label: status.toLowerCase(), className: 'bg-zinc-100 text-zinc-600' }
}

function MetricCard({ name, value }: { name: string; value: number }) {
  return (
    <div className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs">
      <div className="text-zinc-500">{name}</div>
      <div className="font-mono text-zinc-800">{formatMetricValue(value)}</div>
    </div>
  )
}

function downloadCsv(run: RunOut) {
  const csv = buildRunCsv(run.metrics, run.params)
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `run-${run.run_id.slice(0, 8)}-metrics.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function RunDetailPanel({ run }: { run: RunOut | null }) {
  if (!run) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-400">
        Select a run to view its metrics
      </div>
    )
  }

  const metrics = Object.entries(run.metrics).sort(([a], [b]) => a.localeCompare(b))
  const params = Object.entries(run.params).sort(([a], [b]) => a.localeCompare(b))
  const badge = statusBadge(run.status)
  const canExport = metrics.length > 0 || params.length > 0

  return (
    <div className="h-full overflow-y-auto">
      <div className="space-y-6">
        {/* Header */}
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-zinc-900">{run.run_name}</h3>
            <span className={`rounded-full px-3 py-0.5 text-xs font-semibold ${badge.className}`}>
              {badge.label}
            </span>
            <span className="text-xs text-zinc-400">{formatRunTime(run.start_time)}</span>
            {canExport && (
              <button
                onClick={() => downloadCsv(run)}
                className="ml-auto rounded px-2 py-1 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
              >
                Export CSV
              </button>
            )}
          </div>
          <div className="font-mono text-[11px] text-zinc-400">run_id: {run.run_id.slice(0, 8)}…</div>
        </div>

        {/* Metrics */}
        <section>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Metrics</p>
          {metrics.length > 0 ? (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
              {metrics.map(([name, value]) => (
                <MetricCard key={name} name={name} value={value} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-400">No metrics logged</p>
          )}
        </section>

        {/* Configuration */}
        <section>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Configuration</p>
          {params.length > 0 ? (
            <table className="w-full text-xs">
              <tbody>
                {params.map(([k, v]) => (
                  <tr key={k} className="border-b border-zinc-100 last:border-0">
                    <td className="py-1 pr-4 align-top font-medium whitespace-nowrap text-zinc-600">{k}</td>
                    <td className="py-1 font-mono break-all text-zinc-800">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-zinc-400">No parameters logged</p>
          )}
        </section>
      </div>
    </div>
  )
}
