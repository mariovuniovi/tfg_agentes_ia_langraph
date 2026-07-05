import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'

vi.mock('next/navigation', () => ({ usePathname: () => '/pipeline' }))
vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query')
  return { ...actual }
})
vi.mock('@/lib/api', () => ({ fetchHealth: vi.fn().mockResolvedValue({ status: 'ok', mlflow: true, graph: true }) }))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

describe('TopNav', () => {
  it('renders the three primary tabs (Monitoring deliberately excluded)', async () => {
    const { TopNav } = await import('@/components/TopNav')
    render(<TopNav />, { wrapper })
    expect(screen.getByText('Pipeline')).toBeInTheDocument()
    expect(screen.getByText('Experiments')).toBeInTheDocument()
    expect(screen.getByText('Observability')).toBeInTheDocument()
    expect(screen.queryByText('Monitoring')).not.toBeInTheDocument()
  })

  it('marks active tab based on pathname', async () => {
    const { TopNav } = await import('@/components/TopNav')
    render(<TopNav />, { wrapper })
    const active = screen.getByText('Pipeline').closest('a')
    expect(active?.className).toContain('bg-indigo-50')
  })
})
