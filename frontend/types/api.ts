export type RunStatus = 'running' | 'awaiting_approval' | 'complete' | 'failed'

export type PipelineEventType =
  | 'run_info'
  | 'routing'
  | 'tool_call'
  | 'tool_result'
  | 'agent_reasoning'
  | 'planner_context'
  | 'hitl_request'
  | 'audit_report'
  | 'run_complete'

export interface PipelineEvent {
  type: PipelineEventType
  agent: string
  timestamp_ms: number
  data: Record<string, unknown>
}

export interface RunStatusResponse {
  run_id: string
  status: RunStatus
  interrupt_value: Record<string, unknown> | null
}

export interface HITLDecision {
  decision: 'approve' | 'reject'
  reason?: string
  comment?: string
}

export interface DataValidationInterrupt {
  type: 'data_validation'
  attempt?: number
  question?: string
  dataset_preview: {
    path: string
    shape: [number, number]
    row_count: number
    column_count: number
    columns: Array<{ name: string; dtype: string }>
    sample_rows: Record<string, unknown>[]
    head: Record<string, unknown>[]
    tail: Record<string, unknown>[]
  }
  validation_report?: Record<string, unknown>
  validation_summary?: {
    passed: boolean
    missing_values: Record<string, number>
    schema_validated: boolean
  }
}

export interface AuditReportEventData {
  audit: {
    summary?: string
    champion_model?: string
    why_champion_won?: string
    planner_alignment?: string
    deviations_from_planner_expectations?: string[]
    evidence_consistency_warnings?: string[]
    risks_and_warnings?: string[]
    promotion_decision_explanation?: string
    human_review_notes?: string[]
  }
  champion_model: string
  evaluation_passed: boolean
  candidate_metrics: Record<string, unknown>
  champion_metrics: Record<string, unknown>
  thresholds_applied: Record<string, unknown>
}

export interface DeployerInterrupt {
  type: 'deployer'
  evaluation_report?: Record<string, unknown>
  evaluation_report_audit?: AuditReportEventData['audit']
  candidate_metrics?: Record<string, unknown>
  champion_metrics?: Record<string, unknown>
  thresholds_applied?: Record<string, unknown>
  training_plan?: Record<string, unknown>
  candidate_run_id?: string
  deployment_action?: {
    verb: string
    model: string
    alias: string
    summary: string
  }
}

export interface ExperimentOut {
  experiment_id: string
  name: string
}

export type LineStyle = 'solid' | 'dashed' | 'dotted'

export interface MetricSeries {
  name: string
  steps: number[]
  values: number[]
  line_style: LineStyle
}

export interface RunOut {
  run_id: string
  run_name: string
  status: string
  start_time: string
  params: Record<string, string>
  metrics: Record<string, number>
  metric_series: MetricSeries[]
}

export interface ColumnDriftResult {
  column: string
  drift_detected: boolean
  score: number
  method: string
}

export interface DriftReport {
  dataset_drift: boolean
  drift_share: number
  columns: ColumnDriftResult[]
  generated_at: string
}

export interface HealthResponse {
  status: 'ok'
  mlflow: boolean
  graph: boolean
}
