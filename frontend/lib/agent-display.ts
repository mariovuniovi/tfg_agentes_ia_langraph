export const DISPLAY_AGENT: Record<string, string> = {
  supervisor:          'Controller',      // historical events only
  controller:          'Controller',
  workflow_controller: 'Controller',
  data_validator:      'Data Validator',
  dataset_approval:    'Dataset Approval',
  planner:             'Model Planner',
  executor:            'Training Executor',
  evaluation:          'Evaluation',
  report_writer:       'Audit Report',
  deployment_approval: 'Deployment Approval',
  deployer:            'Deployer',
  system:              'System',
}

export function displayAgentName(raw: string): string {
  return DISPLAY_AGENT[raw] ?? raw
}
