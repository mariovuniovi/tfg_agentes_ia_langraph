import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PlannerPanel } from '@/components/pipeline/PlannerPanel'
import type { PlannerContextData } from '@/types/api'

const minimalCtx: PlannerContextData = {
  retrieved_experiences: [],
  matched_rules: [],
  evidence_used: [],
  planning_analysis: 'long analysis text here',
  plan_summary: {
    candidate_rationales: [{
      model_key: 'ets', priority: 1, reason: 'short history',
      evidence_refs: [{ source: 'registry', source_id: 'ets' }],
      risks: ['risk1'],
    }],
    rejected_model_rationales: [],
    candidate_models: ['ets'],
    models_not_recommended: [],
  },
  warnings: [],
  decision_basis: {
    primary_evidence: [{ source: 'dataset_profile' }],
    secondary_evidence: [],
    final_strategy: 'prefer statistical',
  },
  evidence_conflicts: [],
  soft_conflicts: [],
  cited_experience_ids: [],
  cited_rule_ids: [],
  planner_status: 'ok',
}

describe('<PlannerPanel>', () => {
  it('renders placeholder when ctx is null and not running', () => {
    render(<PlannerPanel ctx={null} running={false} />)
    expect(screen.getByText(/Planner has not run yet/)).toBeInTheDocument()
  })
  it('renders all sections with minimal valid ctx', () => {
    render(<PlannerPanel ctx={minimalCtx} running={false} />)
    expect(screen.getByText(/Selected \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Decision basis/)).toBeInTheDocument()
    expect(screen.getByText(/Candidate rationale \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Evidence quality/)).toBeInTheDocument()
    expect(screen.getByText(/View full planning analysis/)).toBeInTheDocument()
  })
  it('renders ConflictPanel only when conflicts present', () => {
    const ctxWithConflict = {
      ...minimalCtx,
      evidence_conflicts: [{
        summary: 'a', affected_models: ['ets'],
        conflicting_evidence_refs: [], resolution: 'r',
      }],
    }
    render(<PlannerPanel ctx={ctxWithConflict} running={false} />)
    expect(screen.getByText(/Evidence conflict/)).toBeInTheDocument()
  })
})
