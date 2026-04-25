import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

vi.mock('@/lib/api', () => ({ approveRun: vi.fn().mockResolvedValue({ ok: true }) }))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
)

describe('HITLGate', () => {
  beforeEach(() => useRunStore.getState().reset())

  it('renders nothing when hitlPending is false', async () => {
    const { HITLGate } = await import('@/components/pipeline/HITLGate')
    const { container } = render(<HITLGate runId="abc" />, { wrapper })
    expect(container.firstChild).toBeNull()
  })

  it('renders Approve and Reject when hitlPending is true', async () => {
    useRunStore.getState().setHITL({ model: 'v1', accuracy: 0.96 })
    const { HITLGate } = await import('@/components/pipeline/HITLGate')
    render(<HITLGate runId="abc" />, { wrapper })
    expect(screen.getByText('Approve')).toBeInTheDocument()
    expect(screen.getByText('Reject')).toBeInTheDocument()
  })
})
