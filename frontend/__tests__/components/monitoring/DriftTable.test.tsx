import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'
import type { ColumnDriftResult } from '@/types/api'
import { DriftTable } from '@/components/monitoring/DriftTable'

describe('DriftTable', () => {
  it('renders column names and drift indicators', () => {
    const columns: ColumnDriftResult[] = [
      { column: 'sepal_length', drift_detected: false, score: 0.04, method: 'PSI' },
      { column: 'petal_width', drift_detected: true, score: 0.41, method: 'PSI' },
    ]
    render(<DriftTable columns={columns} />)
    expect(screen.getByText('sepal_length')).toBeInTheDocument()
    expect(screen.getByText('petal_width')).toBeInTheDocument()
  })
})
