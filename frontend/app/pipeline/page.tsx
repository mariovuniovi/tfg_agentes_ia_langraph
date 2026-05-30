'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { TriggerPanel } from '@/components/pipeline/TriggerPanel'
import { HITLGate } from '@/components/pipeline/HITLGate'
import { EventLog } from '@/components/pipeline/EventLog'
import { ResultsDashboard } from '@/components/pipeline/ResultsDashboard'
import { RunHeader } from '@/components/pipeline/RunHeader'
import { PipelineStepper } from '@/components/pipeline/PipelineStepper'
import { useRunStream } from '@/hooks/use-run-stream'
import { deriveStages } from '@/lib/stage-derive'

const STAGE_LABELS: Record<string, string> = {
  data_validation: 'Data Validation',
  dataset_approval: 'Awaiting dataset approval',
  model_planning: 'Model Planning',
  training: 'Training',
  evaluation: 'Evaluation',
  audit_report: 'Generating audit report',
  deploy_approval: 'Awaiting deployment approval',
  deploy: 'Deploying',
}

export default function PipelinePage() {
  const runId = useRunStore((s) => s.runId)
  const status = useRunStore((s) => s.status)
  const events = useRunStore((s) => s.events)
  useRunStream(runId)

  const { stages, attempts, runOutcome } = useMemo(
    () => deriveStages(events, status),
    [events, status],
  )

  const activeStage = useMemo(() => {
    const order = ['deploy', 'deploy_approval', 'audit_report', 'evaluation', 'training', 'model_planning', 'dataset_approval', 'data_validation'] as const
    for (const k of order) {
      if (stages[k] === 'running' || stages[k] === 'waiting_human') return k
    }
    return null
  }, [stages])

  const problemType = useMemo(() => {
    const tp = events.find((e) => e.type === 'run_info')
    return (tp?.data as { problem_type?: string } | undefined)?.problem_type ?? ''
  }, [events])

  const llmModels = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])

  const startedMs = events[0]?.timestamp_ms ?? Date.now()

  return (
    <div className="space-y-3 p-3">
      {runId && (
        <>
          <RunHeader
            runId={runId}
            problemType={problemType}
            stageLabel={activeStage ? STAGE_LABELS[activeStage] : runOutcome}
            startedMs={startedMs}
            runOutcome={runOutcome}
            attemptCount={attempts.data_validator}
            llmModels={llmModels}
          />
          <PipelineStepper stages={stages} />
        </>
      )}
      <div className="grid grid-cols-5 gap-3">
        <div className="col-span-3 flex flex-col gap-3">
          <TriggerPanel />
          <ResultsDashboard />
          <HITLGate runId={runId} />
        </div>
        <div className="col-span-2">
          <EventLog />
        </div>
      </div>
    </div>
  )
}
