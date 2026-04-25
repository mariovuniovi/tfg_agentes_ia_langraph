'use client'
import { useState } from 'react'
import { startRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'
import { RunStatusBadge } from './RunStatusBadge'

export function TriggerPanel({ onRunStarted }: { onRunStarted: (id: string) => void }) {
  const [paths, setPaths] = useState('')
  const [loading, setLoading] = useState(false)
  const status = useRunStore((s) => s.status)

  async function handleRun() {
    const dataset_paths = paths.split(',').map((p) => p.trim()).filter(Boolean)
    if (!dataset_paths.length) return
    setLoading(true)
    try {
      const { run_id } = await startRun(dataset_paths)
      useRunStore.getState().setRunId(run_id)
      onRunStarted(run_id)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-navy-900">Start Pipeline Run</h2>
      <input
        type="text"
        value={paths}
        onChange={(e) => setPaths(e.target.value)}
        placeholder="data/samples/iris_measurements.csv"
        className="mb-3 w-full rounded border border-slate-300 px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-navy"
      />
      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={loading || status === 'running'}
          className="rounded bg-navy px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Starting...' : '▶ Run Pipeline'}
        </button>
        <RunStatusBadge />
      </div>
    </div>
  )
}
