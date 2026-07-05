import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import { useRunStore } from '@/stores/run-store'
import { EventLog } from '@/components/pipeline/EventLog'

// Stub URL APIs not available in jsdom
vi.stubGlobal('URL', {
  createObjectURL: vi.fn(() => 'blob:mock'),
  revokeObjectURL: vi.fn(),
})

describe('EventLog', () => {
  beforeEach(() => useRunStore.getState().reset())

  it('renders all 3 tab labels', () => {
    render(<EventLog />)
    expect(screen.getByText('Timeline')).toBeInTheDocument()
    expect(screen.getByText('Tool Details')).toBeInTheDocument()
    expect(screen.getByText('Raw Logs')).toBeInTheDocument()
  })

  it('Timeline tab shows "Workflow moved to <agent>" after a routing event', () => {
    useRunStore.getState().appendEvent({
      type: 'routing',
      agent: 'controller',
      timestamp_ms: 1000,
      data: { next: 'data_validator' },
    })
    render(<EventLog />)
    // Timeline is the default tab
    expect(screen.getByText(/Workflow moved to data_validator/i)).toBeInTheDocument()
  })

  it('Download button appears only on Raw Logs tab', () => {
    render(<EventLog />)
    // On Timeline tab (default), button should not be visible
    expect(screen.queryByText(/Download raw trace JSON/i)).not.toBeInTheDocument()

    // Switch to Raw Logs tab
    fireEvent.click(screen.getByText('Raw Logs'))
    expect(screen.getByText(/Download raw trace JSON/i)).toBeInTheDocument()

    // Switch back to Timeline, button disappears
    fireEvent.click(screen.getByText('Timeline'))
    expect(screen.queryByText(/Download raw trace JSON/i)).not.toBeInTheDocument()
  })

  it('shows placeholder on Timeline when no events', () => {
    render(<EventLog />)
    expect(screen.getByText(/Waiting for events/i)).toBeInTheDocument()
  })

  it('Raw Logs tab shows event type and agent', () => {
    useRunStore.getState().appendEvent({
      type: 'routing',
      agent: 'supervisor',
      timestamp_ms: 1,
      data: { next: 'trainer' },
    })
    useRunStore.getState().appendEvent({
      type: 'tool_call',
      agent: 'trainer',
      timestamp_ms: 2,
      data: { tool_name: 'train_model' },
    })
    render(<EventLog />)
    fireEvent.click(screen.getByText('Raw Logs'))
    expect(screen.getByText('routing')).toBeInTheDocument()
    expect(screen.getByText('tool_call')).toBeInTheDocument()
  })
})
