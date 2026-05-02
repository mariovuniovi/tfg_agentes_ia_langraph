'use client'
import { useRunStore } from '@/stores/run-store'
import type { RunStatus } from '@/types/api'

const STYLES: Record<RunStatus | 'idle', string> = {
  idle: 'bg-slate-200 text-slate-600',
  running: 'animate-pulse bg-blue-100 text-blue-700',
  awaiting_approval: 'bg-amber-100 text-amber-700',
  complete: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
}

export function RunStatusBadge() {
  const status = useRunStore((s) => s.status)
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${STYLES[status] ?? ''}`}>
      {status}
    </span>
  )
}
