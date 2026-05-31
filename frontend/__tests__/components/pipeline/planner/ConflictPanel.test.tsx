import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConflictPanel } from '@/components/pipeline/planner/ConflictPanel'

describe('<ConflictPanel>', () => {
  it('renders nothing when both hard and soft are empty', () => {
    const { container } = render(<ConflictPanel hard={[]} soft={[]} />)
    expect(container.firstChild).toBeNull()
  })
  it('renders hard conflicts with amber border', () => {
    render(<ConflictPanel hard={[{
      summary: 'extra_trees won but not selected',
      affected_models: ['extra_trees'],
      conflicting_evidence_refs: [],
      resolution: 'short history; conservative choice',
    }]} soft={[]} />)
    expect(screen.getByText(/extra_trees won/)).toBeInTheDocument()
    expect(screen.getByText(/Resolution: short history/)).toBeInTheDocument()
  })
  it('renders soft conflicts as info', () => {
    render(<ConflictPanel hard={[]} soft={[{
      type: 'retrieved_experience_winner_not_selected',
      models: ['extra_trees'],
      summary: '1 model won in retrieved experiences but was not cited or selected',
    }]} />)
    expect(screen.getByText(/1 model won/)).toBeInTheDocument()
  })
})
