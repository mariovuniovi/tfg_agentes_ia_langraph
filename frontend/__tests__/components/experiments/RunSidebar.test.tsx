import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { RunSidebar } from '@/components/experiments/RunSidebar'

vi.mock('@/lib/api', () => ({
  fetchExperiments: vi.fn().mockResolvedValue([{ experiment_id: '0', name: 'Default' }]),
  fetchExperimentRuns: vi.fn().mockResolvedValue([]),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

describe('RunSidebar', () => {
  it('renders experiment name after load', async () => {
    render(<RunSidebar selectedRunId={null} onSelectRun={() => {}} />, { wrapper })
    expect(await screen.findByText('Default')).toBeInTheDocument()
  })
})
