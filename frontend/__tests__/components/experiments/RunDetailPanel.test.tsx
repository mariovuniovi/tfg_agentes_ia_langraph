import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { RunDetailPanel } from '@/components/experiments/RunDetailPanel'
import type { RunOut } from '@/types/api'

const run: RunOut = {
  run_id: 'a1b2c3d4e5f6',
  run_name: 'extra_trees_forecaster',
  status: 'FINISHED',
  start_time: '2026-06-26T06:27:05+00:00',
  params: { season_length: '12', model_type: 'extra_trees_forecaster' },
  metrics: { rmse: 18.5, mae: 12.3 },
  metric_series: [],
}

describe('RunDetailPanel', () => {
  it('shows empty state when run is null', () => {
    render(<RunDetailPanel run={null} />)
    expect(screen.getByText(/select a run/i)).toBeInTheDocument()
  })

  it('renders a formatted metric value', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('18.500')).toBeInTheDocument()
  })

  it('renders a param key', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('season_length')).toBeInTheDocument()
  })

  it('renders a status badge label', () => {
    render(<RunDetailPanel run={run} />)
    expect(screen.getByText('complete')).toBeInTheDocument()
  })

  it('hides Export CSV when there are no metrics and no params', () => {
    render(<RunDetailPanel run={{ ...run, metrics: {}, params: {} }} />)
    expect(screen.queryByText(/export csv/i)).not.toBeInTheDocument()
  })
})
