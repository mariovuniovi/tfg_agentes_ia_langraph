import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PipelineStepper } from '@/components/pipeline/PipelineStepper'

const allPending = {
  data_validation: 'pending', dataset_approval: 'pending', model_planning: 'pending',
  training: 'pending', evaluation: 'pending', audit_report: 'pending',
  deploy_approval: 'pending', deploy: 'pending',
} as const

describe('<PipelineStepper>', () => {
  it('renders 8 named stages', () => {
    render(<PipelineStepper stages={allPending} />)
    expect(screen.getByText('Data Validation')).toBeInTheDocument()
    expect(screen.getByText('Dataset Approval')).toBeInTheDocument()
    expect(screen.getByText('Model Planning')).toBeInTheDocument()
    expect(screen.getByText('Training')).toBeInTheDocument()
    expect(screen.getByText('Evaluation')).toBeInTheDocument()
    expect(screen.getByText('Audit Report')).toBeInTheDocument()
    expect(screen.getByText('Deploy Approval')).toBeInTheDocument()
    expect(screen.getByText('Deploy')).toBeInTheDocument()
  })

  it('marks a completed stage with a check', () => {
    render(<PipelineStepper stages={{ ...allPending, training: 'completed' }} />)
    expect(screen.getByTestId('stage-training').textContent).toContain('✓')
  })

  it('renders waiting_human stage with a clock', () => {
    render(<PipelineStepper stages={{ ...allPending, dataset_approval: 'waiting_human' }} />)
    expect(screen.getByTestId('stage-dataset_approval').textContent).toContain('⏱')
  })
})
