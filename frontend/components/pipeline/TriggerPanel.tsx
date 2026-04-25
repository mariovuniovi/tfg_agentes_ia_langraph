'use client'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { uploadFiles, startRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'
import { RunStatusBadge } from './RunStatusBadge'

function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function TriggerPanel({ onRunStarted }: { onRunStarted: (id: string) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const status = useRunStore((s) => s.status)

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  async function handleRun() {
    if (!files.length) return
    setLoading(true)
    try {
      const { paths } = await uploadFiles(files)
      const { run_id } = await startRun(paths)
      useRunStore.getState().setRunId(run_id)
      onRunStarted(run_id)
      toast.success('Pipeline started', {
        description: `${files.length} file${files.length > 1 ? 's' : ''} uploaded · ${run_id}`,
      })
    } catch (err) {
      toast.error('Upload failed', {
        description: err instanceof Error ? err.message : 'Check file format and try again',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-navy-900">Start Pipeline Run</h2>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mb-3 flex w-full items-center justify-center gap-2 rounded bg-navy px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
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
        <ul className="mb-3 space-y-1.5">
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

      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={loading || !files.length || status === 'running'}
          className="rounded bg-navy px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Starting...' : '▶ Run Pipeline'}
        </button>
        <RunStatusBadge />
      </div>
    </div>
  )
}
