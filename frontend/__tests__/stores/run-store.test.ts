import { describe, it, expect, beforeEach } from 'vitest'
import { useRunStore } from '@/stores/run-store'

beforeEach(() => useRunStore.getState().reset())

describe('useRunStore', () => {
  it('starts idle with no runId', () => {
    const s = useRunStore.getState()
    expect(s.runId).toBeNull()
    expect(s.status).toBe('idle')
    expect(s.events).toHaveLength(0)
    expect(s.hitlPending).toBe(false)
  })

  it('setRunId updates runId and status to running', () => {
    useRunStore.getState().setRunId('abc-123')
    const s = useRunStore.getState()
    expect(s.runId).toBe('abc-123')
    expect(s.status).toBe('running')
  })

  it('appendEvent adds an event', () => {
    useRunStore.getState().appendEvent({
      type: 'routing', agent: 'supervisor', timestamp_ms: 1000, data: {},
    })
    expect(useRunStore.getState().events).toHaveLength(1)
  })

  it('setHITL sets hitlPending, interruptValue, and status', () => {
    useRunStore.getState().setHITL({ model: 'v1' })
    const s = useRunStore.getState()
    expect(s.hitlPending).toBe(true)
    expect(s.interruptValue).toEqual({ model: 'v1' })
    expect(s.status).toBe('awaiting_approval')
  })

  it('clearHITL clears hitlPending and resumes running status', () => {
    useRunStore.getState().setHITL({ model: 'v1' })
    useRunStore.getState().clearHITL()
    const s = useRunStore.getState()
    expect(s.hitlPending).toBe(false)
    expect(s.interruptValue).toBeNull()
    expect(s.status).toBe('running')
  })

  it('setStatus updates status', () => {
    useRunStore.getState().setStatus('complete')
    expect(useRunStore.getState().status).toBe('complete')
  })

  it('reset returns to initial state', () => {
    useRunStore.getState().setRunId('abc-123')
    useRunStore.getState().reset()
    expect(useRunStore.getState().runId).toBeNull()
    expect(useRunStore.getState().status).toBe('idle')
  })
})
