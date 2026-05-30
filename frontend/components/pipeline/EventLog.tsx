'use client'
import { useMemo, useEffect, useRef } from 'react'
import { useRunStore } from '@/stores/run-store'
import type { PipelineEvent, PipelineEventType } from '@/types/api'

const TYPE_COLORS: Record<PipelineEventType, string> = {
  run_info: 'text-zinc-400',
  routing: 'text-blue-600',
  tool_call: 'text-zinc-500',
  tool_result: 'text-zinc-400',
  agent_reasoning: 'text-indigo-500',
  planner_context: 'text-violet-600',
  hitl_request: 'font-semibold text-amber-700',
  run_complete: 'font-semibold text-green-600',
}

function formatContent(event: PipelineEvent, accumulatedContent?: string): string {
  switch (event.type) {
    case 'run_info': {
      const models = event.data.models as Record<string, string> | undefined
      if (!models) return ''
      return Object.entries(models).map(([agent, model]) => `${agent}=${model}`).join(' · ')
    }
    case 'routing':
      return `→ ${event.data.next ?? ''}`
    case 'tool_call':
      return String(event.data.tool_name ?? '')
    case 'tool_result': {
      const ms = typeof event.data.duration_ms === 'number'
        ? ` · ${Math.round(event.data.duration_ms as number)}ms`
        : ''
      return `${event.data.tool_name ?? ''}${ms}`
    }
    case 'agent_reasoning':
      return accumulatedContent ?? String(event.data.content ?? '')
    case 'planner_context': {
      const exps = (event.data.retrieved_experiences as unknown[]) ?? []
      const rules = (event.data.matched_rules as unknown[]) ?? []
      const cited = (event.data.evidence_used as unknown[]) ?? []
      const summary = event.data.plan_summary as Record<string, unknown> | undefined
      const candidates = (summary?.candidate_models as string[] | undefined) ?? []
      return (
        `${exps.length} similar experience${exps.length !== 1 ? 's' : ''} retrieved · ` +
        `${rules.length} rule${rules.length !== 1 ? 's' : ''} matched · ` +
        `${cited.length} cited · ` +
        `candidates: ${candidates.join(', ') || '—'}`
      )
    }
    case 'hitl_request':
      return 'Awaiting human approval'
    case 'run_complete':
      return event.data.error ? `Error: ${String(event.data.error).slice(0, 80)}` : 'Done'
    default:
      return ''
  }
}

interface Row {
  event: PipelineEvent
  accumulatedContent: string
  key: string
}

function groupEvents(events: PipelineEvent[]): Row[] {
  const rows: Row[] = []
  for (let i = 0; i < events.length; i++) {
    const event = events[i]
    const last = rows[rows.length - 1]
    if (
      last &&
      last.event.type === 'agent_reasoning' &&
      event.type === 'agent_reasoning' &&
      last.event.agent === event.agent
    ) {
      last.accumulatedContent += String(event.data.content ?? '')
    } else {
      rows.push({
        event,
        accumulatedContent: event.type === 'agent_reasoning'
          ? String(event.data.content ?? '')
          : '',
        key: `${event.timestamp_ms}-${i}`,
      })
    }
  }
  return rows
}

export function EventLog() {
  const events = useRunStore((s) => s.events)
  const bottomRef = useRef<HTMLDivElement>(null)

  const rows = useMemo(() => groupEvents(events), [events])

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [rows.length])

  return (
    <div className="h-full overflow-y-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 font-mono text-xs">
      {rows.length === 0 && (
        <p className="text-zinc-400">Waiting for pipeline events...</p>
      )}
      {rows.map(({ event, accumulatedContent, key }) => (
        <div key={key} className="mb-1 flex gap-2">
          <span className="shrink-0 whitespace-nowrap text-zinc-300">
            {new Date(event.timestamp_ms).toISOString().slice(11, 23)}
          </span>
          <span className={`shrink-0 whitespace-nowrap ${TYPE_COLORS[event.type]}`}>{event.type}</span>
          <span className="shrink-0 whitespace-nowrap text-zinc-500">{event.agent}</span>
          <span className="min-w-0 break-words text-zinc-600">
            {formatContent(event, accumulatedContent || undefined)}
          </span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
