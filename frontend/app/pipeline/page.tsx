'use client'
import { useState } from 'react'
import { TriggerPanel } from '@/components/pipeline/TriggerPanel'
import { HITLGate } from '@/components/pipeline/HITLGate'
import { EventLog } from '@/components/pipeline/EventLog'
import { useRunStream } from '@/hooks/use-run-stream'

export default function PipelinePage() {
  const [runId, setRunId] = useState<string | null>(null)
  useRunStream(runId)

  return (
    <div className="flex h-[calc(100vh-80px)] gap-4">
      <div className="flex w-2/5 flex-col gap-4">
        <TriggerPanel onRunStarted={setRunId} />
        <HITLGate runId={runId} />
      </div>
      <div className="flex-1 overflow-hidden">
        <EventLog />
      </div>
    </div>
  )
}
