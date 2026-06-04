import { render, screen } from '@testing-library/react'
import { JoinPlanPanel } from '@/components/pipeline/JoinPlanPanel'
import type { JoinPlan } from '@/types/api'

const mockPlan: JoinPlan = {
  mode: 'inferred',
  base_dataset: {
    dataset_name: 'energy',
    confidence: 'high',
    covered_target_columns: ['week_date', 'kwh_consumed'],
    missing_target_columns: ['avg_temp_c'],
    reason: 'contains target column',
    warnings: [],
  },
  selected_joins: [{
    step_id: 1,
    candidate_id: 'join_001',
    left_dataset: 'energy',
    left_column: 'week_date',
    right_dataset: 'weather',
    right_column: 'week_date',
    join_type: 'left',
    columns_added: ['avg_temp_c'],
    evaluation: {
      candidate_id: 'join_001',
      left_dataset: 'energy', left_column: 'week_date',
      right_dataset: 'weather', right_column: 'week_date',
      left_distinct: 52, right_distinct: 52, intersection_count: 52,
      left_coverage: 1.0, right_coverage: 1.0, jaccard: 1.0, containment: 1.0,
      left_unique_ratio: 1.0, right_unique_ratio: 1.0,
      inferred_relationship: 'one_to_one',
      estimated_inner_rows: 52, estimated_left_rows: 52,
      row_multiplier_left: 1.0, join_explosion_risk: 'low',
      warnings: [],
    },
    confidence_after_evaluation: 'high',
    reason: 'perfect overlap on week_date',
    warnings: [],
  }],
  rejected_candidates: [],
  unresolved_ambiguities: [],
  warnings: [],
}

test('renders empty state when no plan', () => {
  render(<JoinPlanPanel joinPlan={null} />)
  expect(screen.getByText(/No inferred join plan/)).toBeInTheDocument()
})

test('renders base dataset selection with row count', () => {
  render(<JoinPlanPanel joinPlan={mockPlan} joinBaseNrows={52} />)
  expect(screen.getByText('energy')).toBeInTheDocument()
  expect(screen.getByText(/contains target column/)).toBeInTheDocument()
  expect(screen.getByText(/52/)).toBeInTheDocument()
})

test('renders selected join metrics', () => {
  render(<JoinPlanPanel joinPlan={mockPlan} />)
  expect(screen.getByText(/LEFT JOIN/)).toBeInTheDocument()
  expect(screen.getByText(/low risk/)).toBeInTheDocument()
})
