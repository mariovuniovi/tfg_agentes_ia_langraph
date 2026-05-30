import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DeploymentApprovalCard } from '@/components/pipeline/DeploymentApprovalCard'

const interrupt = {
  type: 'deployer' as const,
  candidate_run_id: 'c5e7fe98abcd',
  deployment_action: {
    verb: 'register_and_promote',
    model: 'seasonal_naive',
    alias: 'champion',
    summary: 'register + promote',
  },
  evaluation_report_audit: { risks_and_warnings: ['risk a', 'risk b', 'risk c', 'risk d'] },
}

describe('<DeploymentApprovalCard>', () => {
  it('renders champion model name verbatim from payload', () => {
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={vi.fn()} isPending={false} />)
    expect(screen.getByText(/seasonal_naive/)).toBeInTheDocument()
  })
  it('renders top 3 risks only', () => {
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={vi.fn()} isPending={false} />)
    expect(screen.getByText(/risk a/)).toBeInTheDocument()
    expect(screen.queryByText(/risk d/)).not.toBeInTheDocument()
  })
  it('fires onApprove with approve', () => {
    const fn = vi.fn()
    render(<DeploymentApprovalCard runId="r" interrupt={interrupt} onApprove={fn} isPending={false} />)
    fireEvent.click(screen.getByRole('button', { name: /^approve deployment$/i }))
    expect(fn).toHaveBeenCalledWith('approve')
  })
})
