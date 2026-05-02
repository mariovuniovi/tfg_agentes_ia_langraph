import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import { ChartPanel } from '@/components/experiments/ChartPanel'

describe('ChartPanel', () => {
  it('shows empty state when run is null', () => {
    render(<ChartPanel run={null} />)
    expect(screen.getByText(/select a run/i)).toBeInTheDocument()
  })
})
