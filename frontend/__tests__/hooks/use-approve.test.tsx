import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

vi.mock('@/lib/api', () => ({
  approveRun: vi.fn().mockResolvedValue({ ok: true }),
}))

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
)

beforeEach(() => useRunStore.getState().reset())

describe('useApprove', () => {
  it('calls approveRun and clears HITL on success', async () => {
    useRunStore.getState().setHITL({ model: 'v1' })

    const { useApprove } = await import('@/hooks/use-approve')
    const { result } = renderHook(() => useApprove('abc-123'), { wrapper })

    await act(async () => { await result.current.approve('approve') })

    const { approveRun } = await import('@/lib/api')
    expect(approveRun).toHaveBeenCalledWith('abc-123', { decision: 'approve', reason: '' })
    expect(useRunStore.getState().hitlPending).toBe(false)
  })
})
