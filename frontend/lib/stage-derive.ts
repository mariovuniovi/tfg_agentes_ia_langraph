import type { PipelineEvent, RunStatus } from '@/types/api'

export type StageKey =
  | 'data_validation' | 'dataset_approval' | 'model_planning'
  | 'training' | 'evaluation' | 'audit_report'
  | 'deploy_approval' | 'deploy'

export type StageStatus =
  | 'pending' | 'running' | 'completed'
  | 'waiting_human' | 'failed' | 'skipped'

export type RunOutcome = 'running' | 'complete' | 'failed' | 'candidate_rejected'

const INITIAL: Record<StageKey, StageStatus> = {
  data_validation: 'pending',
  dataset_approval: 'pending',
  model_planning: 'pending',
  training: 'pending',
  evaluation: 'pending',
  audit_report: 'pending',
  deploy_approval: 'pending',
  deploy: 'pending',
}

const STAGE_ORDER: StageKey[] = [
  'data_validation', 'dataset_approval', 'model_planning',
  'training', 'evaluation', 'audit_report', 'deploy_approval', 'deploy',
]

export function deriveStages(
  events: PipelineEvent[],
  runStatus: RunStatus | 'idle',
): {
  stages: Record<StageKey, StageStatus>
  attempts: { data_validator: number }
  runOutcome: RunOutcome
} {
  const stages = { ...INITIAL }
  const attempts = { data_validator: 0 }
  let currentStage: StageKey | null = null
  let lastDeployDecision: 'approve' | 'reject' | null = null
  let evaluationPassed: boolean | null = null

  for (const e of events) {
    const next = (e.data as { next?: string }).next
    const tool = (e.data as { tool_name?: string }).tool_name
    const hitlType = (e.data as { type?: string }).type
    const decision = (e.data as { decision?: string }).decision

    if (e.type === 'routing' && next === 'data_validator') {
      // retry: rewind dataset_approval if already completed
      if (stages.dataset_approval === 'completed') {
        stages.dataset_approval = 'pending'
      }
      stages.data_validation = 'running'
      currentStage = 'data_validation'
      attempts.data_validator += 1
    }
    if (e.type === 'tool_result' && tool === 'validate_against_schema') {
      stages.data_validation = 'completed'
    }
    if (e.type === 'hitl_request' && hitlType === 'data_validation') {
      stages.dataset_approval = 'waiting_human'
      currentStage = 'dataset_approval'
    }
    if ((e.type as string) === 'approval_received' && e.agent === 'dataset_approval') {
      stages.dataset_approval = decision === 'approve' ? 'completed' : 'pending'
    }
    if (e.type === 'routing' && next === 'planner') {
      stages.model_planning = 'running'
      currentStage = 'model_planning'
    }
    if (e.type === 'planner_context') {
      stages.model_planning = 'completed'
    }
    if (e.type === 'routing' && next === 'executor') {
      stages.training = 'running'
      currentStage = 'training'
    }
    if (e.type === 'tool_result' && (tool === 'train_model' || tool === 'tune_hyperparameters')) {
      stages.training = 'completed'
    }
    if (e.type === 'routing' && next === 'evaluation') {
      stages.evaluation = 'running'
      currentStage = 'evaluation'
    }
    if (e.type === 'routing' && next === 'report_writer') {
      if (stages.evaluation === 'running') stages.evaluation = 'completed'
      stages.audit_report = 'running'
      currentStage = 'audit_report'
    }
    if ((e.type as string) === 'audit_report') {
      stages.audit_report = 'completed'
      const passed = (e.data as { evaluation_passed?: boolean }).evaluation_passed
      if (typeof passed === 'boolean') evaluationPassed = passed
    }
    if (e.type === 'hitl_request' && hitlType === 'deployer') {
      stages.deploy_approval = 'waiting_human'
      currentStage = 'deploy_approval'
    }
    if ((e.type as string) === 'approval_received' && e.agent === 'deployer') {
      stages.deploy_approval = 'completed'
      lastDeployDecision = decision === 'approve' ? 'approve' : 'reject'
    }
    if (e.type === 'routing' && next === 'deployer') {
      stages.deploy = 'running'
      currentStage = 'deploy'
    }
    if ((e.type as string) === 'deployment_complete') {
      stages.deploy = 'completed'
      if (stages.deploy_approval === 'waiting_human') stages.deploy_approval = 'completed'
    }
    if (e.type === 'run_complete') {
      const errorMsg = (e.data as { error?: string }).error
      if (errorMsg && currentStage) {
        stages[currentStage] = 'failed'
      }
    }
  }

  // run_complete without error means any still-waiting_human gate was actually approved
  const lastEvent = events[events.length - 1]
  const completedWithoutError = lastEvent
    && lastEvent.type === 'run_complete'
    && !(lastEvent.data as { error?: string }).error
  if (completedWithoutError) {
    for (const k of STAGE_ORDER) {
      if (stages[k] === 'waiting_human') stages[k] = 'completed'
    }
  }

  let runOutcome: RunOutcome = 'running'
  if (runStatus === 'failed') {
    runOutcome = 'failed'
  } else if (runStatus === 'complete') {
    if (evaluationPassed === false) {
      runOutcome = 'candidate_rejected'
      stages.deploy_approval = 'skipped'
      stages.deploy = 'skipped'
    } else if (lastDeployDecision === 'reject') {
      runOutcome = 'complete'
      stages.deploy = 'skipped'
    } else if (lastDeployDecision === 'approve' || stages.deploy === 'pending') {
      runOutcome = 'complete'
      if (lastDeployDecision === 'approve') stages.deploy = 'completed'
    } else {
      runOutcome = 'complete'
    }
    // mark any still-pending downstream as skipped on terminal
    let foundActive = false
    for (let i = STAGE_ORDER.length - 1; i >= 0; i--) {
      const k = STAGE_ORDER[i]
      if (stages[k] === 'pending' && !foundActive) stages[k] = 'skipped'
      else foundActive = true
    }
  }

  return { stages, attempts, runOutcome }
}
