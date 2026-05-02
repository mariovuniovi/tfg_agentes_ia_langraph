'use client'
import { useRunStore } from '@/stores/run-store'
import { TriggerPanel } from '@/components/pipeline/TriggerPanel'
import { HITLGate } from '@/components/pipeline/HITLGate'
import { EventLog } from '@/components/pipeline/EventLog'
import { ResultsDashboard } from '@/components/pipeline/ResultsDashboard'
import { useRunStream } from '@/hooks/use-run-stream'

export default function PipelinePage() {
  const runId = useRunStore((s) => s.runId)
  useRunStream(runId)

  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="flex w-2/5 flex-col gap-4 overflow-y-auto pb-2 pr-1">
        <TriggerPanel />
        <ResultsDashboard />
        <HITLGate runId={runId} />
      </div>
      <div className="flex-1 overflow-hidden">
        <EventLog />
      </div>
    </div>
  )
}
