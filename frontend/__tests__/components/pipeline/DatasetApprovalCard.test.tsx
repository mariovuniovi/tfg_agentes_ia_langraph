import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { DatasetApprovalCard } from '@/components/pipeline/DatasetApprovalCard'

const basePreview = {
  path: 'data/processed/x.csv',
  shape: [10, 2] as [number, number],
  row_count: 10,
  column_count: 2,
  columns: [{ name: 'a', dtype: 'int64' }, { name: 'b', dtype: 'object' }],
  sample_rows: [{ a: 1, b: 'x' }],
  head: [{ a: 1, b: 'x' }],
  tail: [],
}

const baseInterrupt = {
  type: 'data_validation' as const,
  attempt: 1,
  dataset_preview: basePreview,
  validation_report: { passed: true, violations: [] },
}

describe('<DatasetApprovalCard>', () => {
  it('disables Reject button until comment >= 4 chars', () => {
    const onApprove = vi.fn()
    render(<DatasetApprovalCard runId="r" interrupt={baseInterrupt} onApprove={onApprove} isPending={false} maxAttempts={3} />)
    const reject = screen.getByRole('button', { name: /reject/i }) as HTMLButtonElement
    expect(reject.disabled).toBe(true)
    fireEvent.change(screen.getByLabelText(/comment/i), { target: { value: 'bad data' } })
    expect(reject.disabled).toBe(false)
  })

  it('hides Tail tab for non-forecasting (tail is empty)', () => {
    render(<DatasetApprovalCard runId="r" interrupt={baseInterrupt} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.queryByRole('button', { name: /^tail$/i })).not.toBeInTheDocument()
  })

  it('shows Tail tab when tail rows present', () => {
    const fc = { ...baseInterrupt, dataset_preview: { ...basePreview, tail: [{ a: 10, b: 'z' }] } }
    render(<DatasetApprovalCard runId="r" interrupt={fc} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.getByRole('button', { name: /^tail$/i })).toBeInTheDocument()
  })

  it('renders attempt indicator N of M', () => {
    render(<DatasetApprovalCard runId="r" interrupt={{ ...baseInterrupt, attempt: 2 }} onApprove={vi.fn()} isPending={false} maxAttempts={3} />)
    expect(screen.getByText(/attempt 2 of 3/i)).toBeInTheDocument()
  })
})
