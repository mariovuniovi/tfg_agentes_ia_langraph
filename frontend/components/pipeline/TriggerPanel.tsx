'use client'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { uploadFiles, validateSchema, startRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'
import { RunStatusBadge } from './RunStatusBadge'
import { formatBytes } from '@/lib/format'

export function TriggerPanel() {
  const files = useRunStore((s) => s.stagedFiles)
  const setStagedFiles = useRunStore((s) => s.setStagedFiles)
  const schemaJson = useRunStore((s) => s.schemaJson)
  const setSchemaJson = useRunStore((s) => s.setSchemaJson)
  const [loading, setLoading] = useState(false)
  const [schemaName, setSchemaName] = useState<string | null>(null)
  const [problemType, setProblemType] = useState<string | null>(null)
  const csvInputRef = useRef<HTMLInputElement>(null)
  const schemaInputRef = useRef<HTMLInputElement>(null)
  const status = useRunStore((s) => s.status)

  function removeFile(index: number) {
    setStagedFiles(files.filter((_, i) => i !== index))
  }

  async function handleSchemaChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (schemaInputRef.current) schemaInputRef.current.value = ''
    if (!file) return
    try {
      const result = await validateSchema(file)
      setSchemaJson(result.schema_json)
      setSchemaName(file.name)
      setProblemType(result.problem_type)
      toast.success(`Schema valid — ${result.problem_type}`, { description: file.name })
    } catch (err) {
      setSchemaJson('')
      setSchemaName(null)
      setProblemType(null)
      toast.error('Invalid schema', {
        description: err instanceof Error ? err.message : 'Check schema format',
      })
    }
  }

  async function handleRun() {
    if (!files.length || !schemaJson) return
    setLoading(true)
    try {
      const { paths } = await uploadFiles(files)
      const { run_id } = await startRun(paths, schemaJson)
      useRunStore.getState().setRunId(run_id)
      setStagedFiles([])
      toast.success('Pipeline started', { description: run_id })
    } catch (err) {
      toast.error('Failed to start pipeline', {
        description: err instanceof Error ? err.message : 'Check file format and try again',
      })
    } finally {
      setLoading(false)
    }
  }

  const canRun = files.length > 0 && !!schemaJson && !loading && status !== 'running'

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 space-y-4">
      <h2 className="text-sm font-semibold text-navy-900">Start Pipeline Run</h2>

      {/* Schema upload */}
      <div>
        <p className="mb-1.5 text-xs font-medium text-slate-500 uppercase tracking-wide">Schema</p>
        <button
          type="button"
          onClick={() => schemaInputRef.current?.click()}
          className="flex w-full items-center justify-center gap-2 rounded border border-dashed border-slate-300 px-4 py-2 text-sm text-slate-600 hover:border-navy hover:text-navy"
        >
          ↑ Upload schema JSON
        </button>
        <input
          ref={schemaInputRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={handleSchemaChange}
        />
        {schemaJson && schemaName && (
          <div className="mt-1.5 flex items-center gap-2 rounded bg-green-50 border border-green-200 px-3 py-1.5 text-sm text-green-800">
            <span className="truncate flex-1">{schemaName}</span>
            {problemType && <span className="shrink-0 text-xs font-medium">{problemType}</span>}
            <button
              type="button"
              onClick={() => { setSchemaJson(''); setSchemaName(null); setProblemType(null) }}
              className="shrink-0 text-green-400 hover:text-red-500"
              aria-label="Remove schema"
            >
              ×
            </button>
          </div>
        )}
        {!schemaJson && (
          <p className="mt-1 text-xs text-slate-400">Required before running the pipeline</p>
        )}
      </div>

      {/* CSV upload */}
      <div>
        <p className="mb-1.5 text-xs font-medium text-slate-500 uppercase tracking-wide">Dataset</p>
        <button
          type="button"
          onClick={() => csvInputRef.current?.click()}
          className="flex w-full items-center justify-center gap-2 rounded bg-navy px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
        >
          ↑ Upload CSV files
        </button>
        <input
          ref={csvInputRef}
          type="file"
          multiple
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const picked = Array.from(e.target.files ?? [])
            if (csvInputRef.current) csvInputRef.current.value = ''
            if (!picked.length) return
            setStagedFiles([...files, ...picked])
            toast.success(`${picked.length} file${picked.length > 1 ? 's' : ''} added`, {
              description: picked.map((f) => f.name).join(', '),
            })
          }}
        />

        {files.length > 0 && (
          <ul className="mt-2 space-y-1.5">
            {files.map((f, i) => (
              <li
                key={`${f.name}-${f.lastModified}`}
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
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={!canRun}
          title={!schemaJson ? 'Upload a schema JSON first' : !files.length ? 'Upload at least one CSV file' : undefined}
          className="rounded bg-navy px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Starting...' : '▶ Run Pipeline'}
        </button>
        <RunStatusBadge />
      </div>
    </div>
  )
}
