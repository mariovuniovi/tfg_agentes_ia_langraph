'use client'
import { useRunStore } from '@/stores/run-store'
import type { RunStatus } from '@/types/api'

const STYLES: Record<RunStatus | 'idle', string> = {
  idle: 'bg-zinc-100 text-zinc-600',
  running: 'animate-pulse bg-indigo-50 text-indigo-700',
  awaiting_approval: 'bg-amber-50 text-amber-700',
  complete: 'bg-emerald-50 text-emerald-700',
  failed: 'bg-red-50 text-red-700',
}

export function RunStatusBadge() {
  const status = useRunStore((s) => s.status)
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${STYLES[status] ?? ''}`}>
      {status}
    </span>
  )
}
