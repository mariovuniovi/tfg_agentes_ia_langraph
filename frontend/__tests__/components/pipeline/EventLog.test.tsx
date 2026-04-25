import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { useRunStore } from '@/stores/run-store'

describe('EventLog', () => {
  beforeEach(() => useRunStore.getState().reset())

  it('renders event types in the log', async () => {
    useRunStore.getState().appendEvent({ type: 'routing', agent: 'supervisor', timestamp_ms: 1, data: {} })
    useRunStore.getState().appendEvent({ type: 'tool_call', agent: 'trainer', timestamp_ms: 2, data: {} })
    const { EventLog } = await import('@/components/pipeline/EventLog')
    render(<EventLog />)
    expect(screen.getByText('routing')).toBeInTheDocument()
    expect(screen.getByText('tool_call')).toBeInTheDocument()
  })

  it('shows placeholder when no events', async () => {
    const { EventLog } = await import('@/components/pipeline/EventLog')
    render(<EventLog />)
    expect(screen.getByText(/waiting/i)).toBeInTheDocument()
  })
})
