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
  | 'training_complete'
  | 'deployment_complete'
  | 'token_usage'
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

export interface HealthResponse {
  status: 'ok'
  mlflow: boolean
  graph: boolean
}

// ---------------------------------------------------------------------------
// Planner v2 contracts
// ---------------------------------------------------------------------------

export interface EvidenceReference {
  source: 'dataset_profile' | 'task_metadata' | 'registry' | 'experience' | 'rule'
  source_id?: string | null
  relevance_note?: string
}

export interface CandidateRationale {
  model_key: string
  priority: number
  reason: string
  evidence_refs: EvidenceReference[]
  risks: string[]
}

export interface RejectedModelRationale {
  model_key: string
  reason: string
  evidence_refs: EvidenceReference[]
  reconsider_if?: string | null
}

export interface DecisionBasis {
  primary_evidence: EvidenceReference[]
  secondary_evidence: EvidenceReference[]
  final_strategy: string
}

export interface EvidenceConflict {
  summary: string
  affected_models: string[]
  conflicting_evidence_refs: EvidenceReference[]
  resolution: string
}

export interface SoftConflict {
  type: string
  models: string[]
  summary: string
}

export interface MatchedRule {
  rule_id: string
  prefer?: string[]
  avoid_or_deprioritize?: string[]
  recommend?: string
  summary: string
}

export interface ExperienceSummary {
  experience_id: string
  similarity_score: number
  relevance_tier: 'high' | 'medium' | 'low'
  matched_buckets: string[]
  mismatched_buckets: string[]
  target_scale_note: string | null
  dataset_name: string
  problem_type: string
  best_model: string
  validation_score: number
  metric_name?: string
}

export interface PlannerContextData {
  retrieved_experiences: ExperienceSummary[]
  matched_rules: MatchedRule[]
  evidence_used: EvidenceReference[]
  planning_analysis: string
  plan_summary: {
    candidate_rationales: CandidateRationale[]
    rejected_model_rationales: RejectedModelRationale[]
    candidate_models: string[]          // legacy
    models_not_recommended: string[]    // legacy
  }
  warnings: string[]
  decision_basis: DecisionBasis
  evidence_conflicts: EvidenceConflict[]
  soft_conflicts: SoftConflict[]
  cited_experience_ids: string[]
  cited_rule_ids: string[]
  planner_status: 'ok' | 'retry_ok' | 'failed'
}
