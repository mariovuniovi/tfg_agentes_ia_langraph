import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useRunStore } from '@/stores/run-store'

class MockWS {
  static instance: MockWS
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  constructor() { MockWS.instance = this }
}
vi.stubGlobal('WebSocket', MockWS)

vi.mock('@/lib/api', () => ({
  fetchRunStatus: vi.fn().mockResolvedValue({
    run_id: 'abc-123', status: 'running', interrupt_value: null,
  }),
}))

beforeEach(() => useRunStore.getState().reset())

describe('useRunStream', () => {
  it('appends events from WebSocket messages', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'routing', agent: 'supervisor', timestamp_ms: 1, data: {} }),
      })
    })
    expect(useRunStore.getState().events).toHaveLength(1)
    expect(useRunStore.getState().events[0].type).toBe('routing')
  })

  it('sets hitlPending on hitl_request event', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'hitl_request', agent: 'deployer', timestamp_ms: 2, data: { model: 'v1' } }),
      })
    })
    expect(useRunStore.getState().hitlPending).toBe(true)
    expect(useRunStore.getState().status).toBe('awaiting_approval')
  })

  it('sets status complete on run_complete event', async () => {
    const { useRunStream } = await import('@/hooks/use-run-stream')
    renderHook(() => useRunStream('abc-123'))
    act(() => {
      MockWS.instance.onmessage?.({
        data: JSON.stringify({ type: 'run_complete', agent: '', timestamp_ms: 3, data: {} }),
      })
    })
    expect(useRunStore.getState().status).toBe('complete')
  })
})
