'use client'
import { useEffect, useRef } from 'react'
import { useRunStore } from '@/stores/run-store'
import type { PipelineEventType } from '@/types/api'

const TYPE_COLORS: Record<PipelineEventType, string> = {
  routing: 'text-blue-600',
  tool_call: 'text-slate-500',
  tool_result: 'text-slate-400',
  agent_reasoning: 'text-indigo-600',
  hitl_request: 'font-semibold text-amber-600',
  run_complete: 'font-semibold text-green-600',
}

export function EventLog() {
  const events = useRunStore((s) => s.events)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [events.length])

  return (
    <div className="h-full overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-xs">
      {events.length === 0 && (
        <p className="text-slate-400">Waiting for pipeline events...</p>
      )}
      {events.map((e, i) => (
        <div key={`${e.timestamp_ms}-${i}`} className="mb-1 flex gap-2">
          <span className="shrink-0 text-slate-300">
            {new Date(e.timestamp_ms).toISOString().slice(11, 23)}
          </span>
          <span className={TYPE_COLORS[e.type]}>{e.type}</span>
          <span className="text-slate-500">{e.agent}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
