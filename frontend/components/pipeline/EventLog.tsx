'use client'
import { useMemo, useState } from 'react'
import { useRunStore } from '@/stores/run-store'
import { Card } from '@/components/ui/Card'
import { displayAgentName } from '@/lib/agent-display'
import { buildToolDetailsViewModel, buildTimeline } from '@/lib/events-aggregate'
import type { PipelineEvent, PipelineEventType } from '@/types/api'

type Tab = 'timeline' | 'tools' | 'raw'

const TYPE_COLORS: Record<PipelineEventType, string> = {
  run_info: 'text-zinc-400',
  routing: 'text-indigo-600',
  tool_call: 'text-zinc-500',
  tool_result: 'text-zinc-400',
  agent_reasoning: 'text-violet-500',
  planner_context: 'text-violet-600',
  hitl_request: 'font-semibold text-amber-700',
  audit_report: 'text-violet-600',
  training_complete: 'text-emerald-600',
  deployment_complete: 'font-semibold text-emerald-600',
  token_usage: 'text-zinc-400',
  run_complete: 'font-semibold text-emerald-600',
}

const HITL_GATE_LABELS: Record<string, string> = {
  dataset_approval: 'Dataset Approval',
  deployment_approval: 'Deployment Approval',
}

const HITL_STATUS_STYLES: Record<string, string> = {
  approved: 'text-emerald-700',
  rejected: 'text-red-600',
  waiting: 'text-amber-600',
  none: 'text-zinc-400',
}

export function EventLog() {
  const events = useRunStore((s) => s.events)
  const runId = useRunStore((s) => s.runId)
  const [tab, setTab] = useState<Tab>('timeline')

  const timeline = useMemo(() => buildTimeline(events), [events])
  const toolDetails = useMemo(() => buildToolDetailsViewModel(events), [events])

  function downloadRaw() {
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `run-${runId ?? 'trace'}.json`
    document.body.appendChild(a); a.click(); a.remove()
    URL.revokeObjectURL(url)
  }

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'timeline', label: 'Timeline' },
    { key: 'tools',    label: 'Tool Details' },
    { key: 'raw',      label: 'Raw Logs' },
  ]

  return (
    <Card
      title="Events"
      actions={tab === 'raw'
        ? <button onClick={downloadRaw} className="rounded border border-[var(--color-border)] px-2 py-0.5 text-[11px] hover:bg-zinc-50">↓ Download raw trace JSON</button>
        : null}
    >
      <div className="mb-2 flex gap-1 border-b border-[var(--color-border)]">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 text-xs font-medium ${
              tab === t.key
                ? 'border-b-2 border-indigo-600 text-indigo-700'
                : 'text-zinc-500 hover:text-zinc-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="max-h-[calc(100vh-280px)] overflow-y-auto">
        {tab === 'timeline' && (
          <ul className="space-y-0.5 text-xs">
            {timeline.length === 0 && <li className="text-zinc-400">Waiting for events…</li>}
            {timeline.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="shrink-0 font-mono text-zinc-300">
                  {new Date(r.ts).toISOString().slice(11, 19)}
                </span>
                <span className="text-zinc-700">{r.text}</span>
              </li>
            ))}
          </ul>
        )}

        {tab === 'tools' && (
          <div className="space-y-4 text-xs">
            {/* Agents section */}
            {toolDetails.agents.length > 0 && (
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Agents</p>
                {toolDetails.agents.map((section) => (
                  <div key={section.name} className="mb-3">
                    <p className="mb-1 font-semibold text-zinc-700">{displayAgentName(section.name)}</p>
                    {section.tools.map((row) => (
                      <div key={row.toolName} className="grid grid-cols-[1fr_60px_80px] gap-2 border-b border-zinc-100 py-1 pl-3">
                        <span className="font-mono text-zinc-700">{row.toolName}</span>
                        <span className="text-right text-zinc-500">{row.calls} call{row.calls === 1 ? '' : 's'}</span>
                        <span className="text-right font-mono text-zinc-500">{row.totalMs > 0 ? `${row.totalMs} ms` : '—'}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}

            {/* LLM nodes section */}
            {toolDetails.llmNodes.length > 0 && (
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">LLM nodes</p>
                {toolDetails.llmNodes.map((r) => (
                  <div key={r.node} className="grid grid-cols-[1fr_80px_80px] gap-2 border-b border-zinc-100 py-1">
                    <span className="font-semibold text-zinc-700">{displayAgentName(r.node)}</span>
                    <span className="text-right text-zinc-500">{r.activations} activation{r.activations === 1 ? '' : 's'}</span>
                    <span className="text-right font-mono text-zinc-500">{(r.totalMs / 1000).toFixed(1)} s</span>
                  </div>
                ))}
              </div>
            )}

            {/* Human gates section */}
            {toolDetails.humanGates.length > 0 && (
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Human gates</p>
                {toolDetails.humanGates.map((g) => (
                  <div key={g.gate} className="flex items-center justify-between border-b border-zinc-100 py-1">
                    <span className="font-semibold text-zinc-700">{HITL_GATE_LABELS[g.gate] ?? g.gate}</span>
                    <span className={`font-mono text-[11px] ${HITL_STATUS_STYLES[g.status] ?? ''}`}>{g.status}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Deterministic steps section */}
            {toolDetails.deterministic.length > 0 && (
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Deterministic steps</p>
                {toolDetails.deterministic.map((d) => (
                  <div key={d.node} className="flex items-center justify-between border-b border-zinc-100 py-1">
                    <span className="font-semibold text-zinc-700">{displayAgentName(d.node)}</span>
                    <span className="text-[11px] text-emerald-600">completed</span>
                  </div>
                ))}
              </div>
            )}

            {toolDetails.agents.length === 0 && toolDetails.llmNodes.length === 0 &&
             toolDetails.humanGates.length === 0 && toolDetails.deterministic.length === 0 && (
              <p className="text-zinc-400">No activity yet.</p>
            )}
          </div>
        )}

        {tab === 'raw' && (
          <div className="font-mono text-[11px]">
            {events.length === 0 && <p className="text-zinc-400">Waiting for events…</p>}
            {events.map((e: PipelineEvent, i) => (
              <div key={i} className="mb-0.5 flex gap-2">
                <span className="shrink-0 text-zinc-300">{new Date(e.timestamp_ms).toISOString().slice(11, 23)}</span>
                <span className={`shrink-0 ${TYPE_COLORS[e.type] ?? 'text-zinc-500'}`}>{e.type}</span>
                <span className="shrink-0 text-zinc-500">{displayAgentName(e.agent)}</span>
                <span className="min-w-0 break-words text-zinc-600">
                  {(() => {
                    const t = e.type
                    if (t === 'routing') return `→ ${(e.data as { next?: string }).next ?? ''}`
                    if (t === 'tool_call' || t === 'tool_result') return String((e.data as { tool_name?: string }).tool_name ?? '')
                    if (t === 'run_complete') {
                      const err = (e.data as { error?: string }).error
                      return err ? `Error: ${err.slice(0, 80)}` : 'Done'
                    }
                    return ''
                  })()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  )
}
