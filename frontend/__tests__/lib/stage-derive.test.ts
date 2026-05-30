import { describe, it, expect } from 'vitest'
import { deriveStages, type StageKey } from '@/lib/stage-derive'
import type { PipelineEvent } from '@/types/api'

function ev(type: string, agent: string, data: Record<string, unknown> = {}): PipelineEvent {
  return { type, agent, timestamp_ms: Date.now(), data } as PipelineEvent
}

describe('deriveStages', () => {
  it('returns all stages pending for an empty event list', () => {
    const { stages, attempts, runOutcome } = deriveStages([], 'running')
    const keys: StageKey[] = [
      'data_validation', 'dataset_approval', 'model_planning',
      'training', 'evaluation', 'audit_report', 'deploy_approval', 'deploy',
    ]
    for (const k of keys) expect(stages[k]).toBe('pending')
    expect(attempts.data_validator).toBe(0)
    expect(runOutcome).toBe('running')
  })

  it('marks data_validation as running on routing event', () => {
    const events = [ev('routing', 'controller', { next: 'data_validator' })]
    const { stages, attempts } = deriveStages(events, 'running')
    expect(stages.data_validation).toBe('running')
    expect(attempts.data_validator).toBe(1)
  })

  it('marks data_validation as completed on validate_against_schema result', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
    ]
    expect(deriveStages(events, 'running').stages.data_validation).toBe('completed')
  })

  it('marks dataset_approval as waiting_human on hitl_request', () => {
    const events = [
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
    ]
    expect(deriveStages(events, 'awaiting_approval').stages.dataset_approval).toBe('waiting_human')
  })

  it('handles full happy path through deploy', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
      ev('approval_received', 'dataset_approval', { decision: 'approve' }),
      ev('routing', 'controller', { next: 'planner' }),
      ev('planner_context', 'planner'),
      ev('routing', 'controller', { next: 'executor' }),
      ev('tool_result', 'executor', { tool_name: 'train_model' }),
      ev('routing', 'controller', { next: 'evaluation' }),
      ev('routing', 'controller', { next: 'report_writer' }),
      ev('audit_report', 'report_writer', { evaluation_passed: true }),
      ev('hitl_request', 'deployer', { type: 'deployer' }),
      ev('approval_received', 'deployer', { decision: 'approve' }),
      ev('run_complete', 'controller'),
    ]
    const { stages, runOutcome } = deriveStages(events, 'complete')
    expect(stages.deploy).toBe('completed')
    expect(runOutcome).toBe('complete')
  })

  it('eval-rejection path produces candidate_rejected outcome', () => {
    const events = [
      ev('routing', 'controller', { next: 'evaluation' }),
      ev('routing', 'controller', { next: 'report_writer' }),
      ev('audit_report', 'report_writer', { evaluation_passed: false }),
      ev('run_complete', 'controller'),
    ]
    const { stages, runOutcome } = deriveStages(events, 'complete')
    expect(stages.evaluation).toBe('completed')
    expect(stages.audit_report).toBe('completed')
    expect(stages.deploy_approval).toBe('skipped')
    expect(stages.deploy).toBe('skipped')
    expect(runOutcome).toBe('candidate_rejected')
  })

  it('retry increments attempts and rewinds dataset_approval', () => {
    const events = [
      ev('routing', 'controller', { next: 'data_validator' }),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema' }),
      ev('hitl_request', 'dataset_approval', { type: 'data_validation' }),
      ev('approval_received', 'dataset_approval', { decision: 'reject' }),
      ev('routing', 'controller', { next: 'data_validator' }),
    ]
    const { stages, attempts } = deriveStages(events, 'running')
    expect(stages.data_validation).toBe('running')
    expect(stages.dataset_approval).toBe('pending')
    expect(attempts.data_validator).toBe(2)
  })

  it('run_complete with error marks current stage failed', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }),
      ev('run_complete', 'controller', { error: 'boom' }),
    ]
    const { stages, runOutcome } = deriveStages(events, 'failed')
    expect(stages.model_planning).toBe('failed')
    expect(runOutcome).toBe('failed')
  })

  it('deploy rejection skips deploy stage', () => {
    const events = [
      ev('hitl_request', 'deployer', { type: 'deployer' }),
      ev('approval_received', 'deployer', { decision: 'reject' }),
      ev('run_complete', 'controller'),
    ]
    expect(deriveStages(events, 'complete').stages.deploy).toBe('skipped')
  })

  it('marks waiting_human stages as completed when run_complete arrives without error', () => {
    const events = [
      ev('hitl_request', 'deployer', { type: 'deployer' }),
      ev('run_complete', 'controller'),
    ]
    const { stages } = deriveStages(events, 'complete')
    expect(stages.deploy_approval).toBe('completed')
  })
})
