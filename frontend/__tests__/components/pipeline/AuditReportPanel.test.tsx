import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AuditReportPanel } from '@/components/pipeline/AuditReportPanel'

const data = {
  audit: {
    summary: 's', why_champion_won: 'w',
    risks_and_warnings: ['risk one', 'risk two'],
    human_review_notes: ['note one'],
  },
  champion_model: 'seasonal_naive',
  evaluation_passed: true,
  candidate_metrics: {}, champion_metrics: {}, thresholds_applied: {},
}

describe('<AuditReportPanel>', () => {
  it('renders champion model + risks open by default', () => {
    render(<AuditReportPanel data={data} />)
    expect(screen.getByText(/seasonal_naive/)).toBeInTheDocument()
    expect(screen.getByText(/risk one/)).toBeInTheDocument()
  })
  it('renders "candidate rejected" banner when evaluation_passed false', () => {
    render(<AuditReportPanel data={{ ...data, evaluation_passed: false }} />)
    expect(screen.getByText(/candidate rejected/i)).toBeInTheDocument()
  })
})
