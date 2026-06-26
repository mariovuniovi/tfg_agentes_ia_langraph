import { describe, it, expect } from 'vitest'
import { formatK, formatCost, formatMetricValue, formatRunTime, buildRunCsv } from '@/lib/format'

describe('formatK', () => {
  it('returns plain string for values below 1000', () => {
    expect(formatK(0)).toBe('0')
    expect(formatK(999)).toBe('999')
  })

  it('formats exactly 1000 as 1.0k', () => {
    expect(formatK(1000)).toBe('1.0k')
  })

  it('formats thousands with one decimal place', () => {
    expect(formatK(1500)).toBe('1.5k')
    expect(formatK(8200)).toBe('8.2k')
    expect(formatK(12345)).toBe('12.3k')
  })
})

describe('formatCost', () => {
  it('returns Unknown for null or undefined', () => {
    expect(formatCost(null)).toBe('Unknown')
    expect(formatCost(undefined)).toBe('Unknown')
  })

  it('returns $0.00000 for exactly zero', () => {
    expect(formatCost(0)).toBe('$0.00000')
  })

  it('uses 5 decimal places for values below $0.01', () => {
    expect(formatCost(0.00341)).toBe('$0.00341')
    expect(formatCost(0.00003)).toBe('$0.00003')
    expect(formatCost(0.00999)).toBe('$0.00999')
  })

  it('uses 4 decimal places for values at $0.01 or above', () => {
    expect(formatCost(0.01512)).toBe('$0.0151')
    expect(formatCost(0.01115)).toBe('$0.0112')
    expect(formatCost(1.23456)).toBe('$1.2346')
  })
})

describe('formatMetricValue', () => {
  it('uses 3 decimals for magnitude >= 1', () => {
    expect(formatMetricValue(18.5)).toBe('18.500')
    expect(formatMetricValue(-2)).toBe('-2.000')
  })

  it('uses 4 decimals for magnitude < 1', () => {
    expect(formatMetricValue(0.0412)).toBe('0.0412')
    expect(formatMetricValue(0)).toBe('0.0000')
  })

  it('renders an em dash for non-finite values', () => {
    expect(formatMetricValue(NaN)).toBe('—')
    expect(formatMetricValue(Infinity)).toBe('—')
  })
})

describe('formatRunTime', () => {
  it('formats a valid ISO string as YYYY-MM-DD HH:mm', () => {
    expect(formatRunTime('2026-06-26T06:27:05+00:00')).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/)
  })

  it('falls back to the raw string when unparseable', () => {
    expect(formatRunTime('not-a-date')).toBe('not-a-date')
  })
})

describe('buildRunCsv', () => {
  it('quotes every field and tags rows by type', () => {
    const csv = buildRunCsv({ rmse: 18.5 }, { model_type: 'ets' })
    expect(csv).toBe('type,key,value\n"metric","rmse","18.5"\n"param","model_type","ets"')
  })

  it('escapes commas and quotes in param values', () => {
    const csv = buildRunCsv({}, { lags: '[1, 2, 3, 12]' })
    expect(csv).toContain('"param","lags","[1, 2, 3, 12]"')
  })

  it('sorts metrics and params alphabetically by key', () => {
    const csv = buildRunCsv({ rmse: 1, mae: 2 }, { z: '1', a: '2' })
    const lines = csv.split('\n')
    expect(lines[1]).toContain('"mae"')
    expect(lines[2]).toContain('"rmse"')
    expect(lines[3]).toContain('"a"')
    expect(lines[4]).toContain('"z"')
  })
})
