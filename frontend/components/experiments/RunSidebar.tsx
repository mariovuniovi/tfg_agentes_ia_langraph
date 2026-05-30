'use client'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchExperiments, fetchExperimentRuns } from '@/lib/api'
import type { RunOut } from '@/types/api'

interface Props {
  selectedRunId: string | null
  onSelectRun: (run: RunOut) => void
}

export function RunSidebar({ selectedRunId, onSelectRun }: Props) {
  const { data: experiments } = useQuery({ queryKey: ['experiments'], queryFn: fetchExperiments })
  const [expId, setExpId] = useState<string | null>(null)

  const activeExpId = expId ?? experiments?.[0]?.experiment_id ?? null

  const { data: runs } = useQuery({
    queryKey: ['runs', activeExpId],
    queryFn: () => fetchExperimentRuns(activeExpId!),
    enabled: !!activeExpId,
  })

  return (
    <div className="flex h-full flex-col gap-3">
      <select
        value={activeExpId ?? ''}
        onChange={(e) => setExpId(e.target.value)}
        className="rounded border border-zinc-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-600"
      >
        {experiments?.map((exp) => (
          <option key={exp.experiment_id} value={exp.experiment_id}>
            {exp.name}
          </option>
        ))}
      </select>
      <div className="flex-1 space-y-1 overflow-y-auto">
        {runs?.map((run) => (
          <button
            key={run.run_id}
            onClick={() => onSelectRun(run)}
            className={`w-full rounded px-3 py-2 text-left text-xs transition-colors ${
              selectedRunId === run.run_id
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-zinc-700 hover:bg-zinc-100'
            }`}
          >
            <div className="font-mono">{run.run_name || run.run_id.slice(0, 8)}</div>
            <div className="text-[10px] opacity-70">
              acc: {run.metrics.accuracy?.toFixed(3) ?? '—'}
            </div>
          </button>
        ))}
        {runs?.length === 0 && <p className="text-xs text-zinc-400">No runs found</p>}
      </div>
    </div>
  )
}
