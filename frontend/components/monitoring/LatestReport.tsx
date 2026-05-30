'use client'
import { useQuery } from '@tanstack/react-query'
import { fetchLatestDrift } from '@/lib/api'
import { DriftTable } from './DriftTable'

export function LatestReport() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ['monitoring', 'latest'],
    queryFn: fetchLatestDrift,
    retry: false,
  })

  if (isLoading) return <p className="text-sm text-zinc-400">Loading...</p>
  if (isError || !data) {
    return (
      <p className="text-sm text-zinc-400">
        No pipeline run completed yet. Run the pipeline to generate a drift report.
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${
          data.dataset_drift ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
        }`}>
          {data.dataset_drift ? 'Drift detected' : 'No drift'}
        </span>
        <span className="text-2xl font-semibold text-zinc-900">
          {(data.drift_share * 100).toFixed(1)}%
        </span>
        <span className="text-sm text-zinc-400">columns with drift</span>
      </div>
      <DriftTable columns={data.columns} />
      <p className="text-xs text-zinc-400">
        Generated at {new Date(data.generated_at).toLocaleString()}
      </p>
    </div>
  )
}
