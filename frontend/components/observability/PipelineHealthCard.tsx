'use client'
import { useQuery } from '@tanstack/react-query'
import { fetchRunsList } from '@/lib/api'
import { Card } from '@/components/ui/Card'

export function PipelineHealthCard() {
  const { data: runs = [] } = useQuery({ queryKey: ['runs-list'], queryFn: () => fetchRunsList(20), refetchInterval: 10_000 })
  const successful = runs.filter((r) => r.status === 'complete').length
  const failed = runs.filter((r) => r.status === 'failed').length
  const awaiting = runs.filter((r) => r.status === 'awaiting_approval').length

  return (
    <Card title="Pipeline health (last 20 runs since server start)">
      <p className="text-sm text-zinc-700">
        {successful} successful · {failed} failed · {awaiting} awaiting human
      </p>
      <p className="mt-1 text-[11px] text-zinc-400">resets on container restart</p>
    </Card>
  )
}
