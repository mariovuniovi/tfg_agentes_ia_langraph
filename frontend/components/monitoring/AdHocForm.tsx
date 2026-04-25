'use client'
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import { runAdHocDrift } from '@/lib/api'
import type { DriftReport } from '@/types/api'
import { DriftTable } from './DriftTable'

function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function AdHocForm() {
  const [files, setFiles] = useState<File[]>([])
  const [refIdx, setRefIdx] = useState(0)
  const [curIdx, setCurIdx] = useState(1)
  const [result, setResult] = useState<DriftReport | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function removeFile(index: number) {
    setFiles((prev) => {
      const next = prev.filter((_, i) => i !== index)
      if (refIdx >= next.length) setRefIdx(0)
      if (curIdx >= next.length) setCurIdx(Math.min(1, next.length - 1))
      return next
    })
  }

  const mutation = useMutation({
    mutationFn: () => runAdHocDrift(files, refIdx, curIdx),
    onSuccess: (data) => {
      setResult(data)
      toast.success('Analysis complete', {
        description: data.dataset_drift ? 'Drift detected in dataset' : 'No drift detected',
      })
    },
    onError: (err) => {
      toast.error('Analysis failed', {
        description: err instanceof Error ? err.message : 'Check that both files are valid CSVs with matching columns',
      })
    },
  })

  return (
    <div className="space-y-4">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="flex w-full items-center justify-center gap-2 rounded bg-navy px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
      >
        ↑ Upload CSV files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv"
        className="hidden"
        onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
      />

      {files.length > 0 && (
        <ul className="space-y-1.5">
          {files.map((f, i) => (
            <li
              key={i}
              className="flex items-center gap-2 rounded bg-slate-100 px-3 py-2 text-sm text-slate-700"
            >
              <span className="truncate flex-1">{f.name}</span>
              <span className="shrink-0 text-xs text-slate-400">{formatBytes(f.size)}</span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                className="shrink-0 text-slate-300 hover:text-red-500"
                aria-label={`Remove ${f.name}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {files.length >= 2 && (
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Reference:
            <select
              value={refIdx}
              onChange={(e) => setRefIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Current:
            <select
              value={curIdx}
              onChange={(e) => setCurIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="rounded bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
          >
            {mutation.isPending ? 'Running...' : 'Run Drift'}
          </button>
        </div>
      )}

      {result && (
        <div className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex items-center gap-4">
            <span className={`rounded-full px-3 py-1 text-sm font-medium ${
              result.dataset_drift ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {result.dataset_drift ? 'Drift detected' : 'No drift'}
            </span>
            <span className="text-xl font-semibold">{(result.drift_share * 100).toFixed(1)}%</span>
            <span className="text-sm text-slate-400">columns with drift</span>
          </div>
          <DriftTable columns={result.columns} />
        </div>
      )}
    </div>
  )
}
