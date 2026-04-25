'use client'
import { useState, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { runAdHocDrift } from '@/lib/api'
import type { DriftReport } from '@/types/api'
import { DriftTable } from './DriftTable'

export function AdHocForm() {
  const [files, setFiles] = useState<File[]>([])
  const [refIdx, setRefIdx] = useState(0)
  const [curIdx, setCurIdx] = useState(1)
  const [result, setResult] = useState<DriftReport | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const mutation = useMutation({
    mutationFn: () => runAdHocDrift(files, refIdx, curIdx),
    onSuccess: setResult,
  })

  return (
    <div className="space-y-4">
      <div
        onClick={() => inputRef.current?.click()}
        className="cursor-pointer rounded-lg border-2 border-dashed border-slate-300 p-6 text-center hover:border-navy"
      >
        <p className="text-sm text-slate-500">
          {files.length
            ? `${files.length} file(s) selected`
            : 'Drop CSV files here or click to upload'}
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv"
          className="hidden"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </div>

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

      {mutation.isError && (
        <p className="rounded bg-red-50 p-3 text-sm text-red-600">
          Drift analysis failed. Check that both files are valid CSVs with matching columns.
        </p>
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
