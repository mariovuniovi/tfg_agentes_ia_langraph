import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => mockFetch.mockReset())

describe('startRun', () => {
  it('POSTs dataset_paths and returns run_id', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ run_id: 'abc-123' }),
    })
    const { startRun } = await import('@/lib/api')
    const result = await startRun(['iris.csv'])
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/runs',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ dataset_paths: ['iris.csv'] }),
      })
    )
    expect(result).toEqual({ run_id: 'abc-123' })
  })
})

describe('approveRun', () => {
  it('POSTs decision to approve endpoint', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) })
    const { approveRun } = await import('@/lib/api')
    await approveRun('abc-123', { decision: 'approve' })
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/runs/abc-123/approve',
      expect.objectContaining({ method: 'POST' })
    )
  })
})

describe('fetchExperiments', () => {
  it('GETs /experiments', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{ experiment_id: '0', name: 'Default' }],
    })
    const { fetchExperiments } = await import('@/lib/api')
    const result = await fetchExperiments()
    expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/experiments')
    expect(result[0].name).toBe('Default')
  })
})

describe('uploadFiles', () => {
  it('uploadFiles POSTs FormData to /uploads and returns paths', async () => {
    const mockPaths = ['data/uploads/abc_iris.csv']
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ paths: mockPaths }),
    } as unknown as Response)

    const file = new File(['col1\n1'], 'iris.csv', { type: 'text/csv' })
    const { uploadFiles } = await import('@/lib/api')
    const result = await uploadFiles([file])

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/uploads',
      expect.objectContaining({ method: 'POST' })
    )
    expect(result).toEqual({ paths: mockPaths })
  })
})
