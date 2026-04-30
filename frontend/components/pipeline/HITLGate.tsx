'use client'
import { useRunStore } from '@/stores/run-store'
import { useApprove } from '@/hooks/use-approve'

export function HITLGate({ runId }: { runId: string | null }) {
  const hitlPending = useRunStore((s) => s.hitlPending)
  const interruptValue = useRunStore((s) => s.interruptValue)
  const { approve, isPending } = useApprove(runId)

  if (!hitlPending) return null
  if ((interruptValue as { type?: string })?.type === 'data_validation') return null

  return (
    <div className="rounded-lg border border-amber-600 bg-amber-50 p-4">
      <p className="mb-2 font-semibold text-amber-800">⚠ Deployment Gate</p>
      {interruptValue && (
        <pre className="mb-3 overflow-auto rounded bg-amber-100 p-2 font-mono text-xs text-amber-900">
          {JSON.stringify(interruptValue, null, 2)}
        </pre>
      )}
      <div className="flex gap-2">
        <button
          onClick={() => approve('approve')}
          disabled={isPending}
          className="rounded bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
        >
          Approve
        </button>
        <button
          onClick={() => approve('reject')}
          disabled={isPending}
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  )
}
