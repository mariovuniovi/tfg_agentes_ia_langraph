import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { useRunStore } from '@/stores/run-store'
import { EventLog } from '@/components/pipeline/EventLog'

describe('EventLog', () => {
  beforeEach(() => useRunStore.getState().reset())

  it('renders event types in the log', () => {
    useRunStore.getState().appendEvent({ type: 'routing', agent: 'supervisor', timestamp_ms: 1, data: {} })
    useRunStore.getState().appendEvent({ type: 'tool_call', agent: 'trainer', timestamp_ms: 2, data: {} })
    render(<EventLog />)
    expect(screen.getByText('routing')).toBeInTheDocument()
    expect(screen.getByText('tool_call')).toBeInTheDocument()
  })

  it('shows placeholder when no events', () => {
    render(<EventLog />)
    expect(screen.getByText(/waiting/i)).toBeInTheDocument()
  })
})
