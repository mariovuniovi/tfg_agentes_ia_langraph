import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { useRunStore } from '@/stores/run-store'
import { HITLGate } from '@/components/pipeline/HITLGate'

vi.mock('@/lib/api', () => ({ approveRun: vi.fn().mockResolvedValue({ ok: true }) }))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
)

describe('HITLGate', () => {
  beforeEach(() => useRunStore.getState().reset())

  it('renders nothing when hitlPending is false', () => {
    const { container } = render(<HITLGate runId="abc" />, { wrapper })
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when interrupt type is not deployer', () => {
    useRunStore.getState().setHITL({ type: 'data_validation', model: 'v1' })
    const { container } = render(<HITLGate runId="abc" />, { wrapper })
    expect(container.firstChild).toBeNull()
  })

  it('renders DeploymentApprovalCard when type is deployer', () => {
    useRunStore.getState().setHITL({
      type: 'deployer',
      candidate_run_id: 'run-abc123',
      deployment_action: null,
      evaluation_report_audit: null,
    })
    render(<HITLGate runId="abc" />, { wrapper })
    expect(screen.getByText(/awaiting human/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /approve deployment/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /reject deployment/i })).toBeInTheDocument()
  })
})
