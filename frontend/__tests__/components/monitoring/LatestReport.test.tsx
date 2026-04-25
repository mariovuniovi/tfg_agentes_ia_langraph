import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { LatestReport } from '@/components/monitoring/LatestReport'

vi.mock('@/lib/api', () => ({
  fetchLatestDrift: vi.fn().mockResolvedValue({
    dataset_drift: false,
    drift_share: 0.12,
    columns: [{ column: 'sepal_length', drift_detected: false, score: 0.04, method: 'PSI' }],
    generated_at: '2026-04-24T10:00:00',
  }),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

describe('LatestReport', () => {
  it('shows No drift badge when dataset_drift is false', async () => {
    render(<LatestReport />, { wrapper })
    expect(await screen.findByText('No drift')).toBeInTheDocument()
  })

  it('shows drift share percentage', async () => {
    render(<LatestReport />, { wrapper })
    expect(await screen.findByText('12.0%')).toBeInTheDocument()
  })
})
