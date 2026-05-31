import { describe, it, expect } from 'vitest'
import { formatK, formatCost } from '@/lib/format'

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
