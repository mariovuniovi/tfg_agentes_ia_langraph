import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RunHeader } from '@/components/pipeline/RunHeader'

describe('<RunHeader>', () => {
  it('renders truncated run id and problem type', () => {
    render(
      <RunHeader
        runId="37c17107abcd"
        problemType="forecasting"
        stageLabel="Waiting for human"
        startedMs={Date.now() - 5000}
        runOutcome="running"
        attemptCount={1}
        llmModels={['data_validator', 'planner', 'report_writer']}
      />,
    )
    expect(screen.getByText(/37c17107/)).toBeInTheDocument()
    expect(screen.getByText(/forecasting/i)).toBeInTheDocument()
    expect(screen.getByText(/Waiting for human/)).toBeInTheDocument()
  })

  it('renders sky pill for candidate_rejected outcome', () => {
    render(
      <RunHeader
        runId="abc"
        problemType="classification"
        stageLabel="Candidate rejected"
        startedMs={Date.now()}
        runOutcome="candidate_rejected"
        attemptCount={1}
        llmModels={[]}
      />,
    )
    const pill = screen.getByTestId('run-status-pill')
    expect(pill.className).toMatch(/sky/)
  })

  it('shows attempt counter when > 1', () => {
    render(
      <RunHeader
        runId="abc"
        problemType="forecasting"
        stageLabel="Data Validation"
        startedMs={Date.now()}
        runOutcome="running"
        attemptCount={2}
        llmModels={[]}
      />,
    )
    expect(screen.getByText(/attempt 2/i)).toBeInTheDocument()
  })
})
