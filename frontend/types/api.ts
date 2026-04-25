export type RunStatus = 'running' | 'awaiting_approval' | 'complete' | 'failed'

export type PipelineEventType =
  | 'routing'
  | 'tool_call'
  | 'tool_result'
  | 'agent_reasoning'
  | 'hitl_request'
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
