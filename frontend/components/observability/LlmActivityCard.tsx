'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateLlmNodeActivity, aggregateTokenUsage, tokensByNode } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { formatK } from '@/lib/format'
import { Card } from '@/components/ui/Card'

export function LlmActivityCard() {
  const events = useRunStore((s) => s.events)
  const llmNodes = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])
  const rows = useMemo(() => aggregateLlmNodeActivity(events, llmNodes), [events, llmNodes])
  const tokenMap = useMemo(() => tokensByNode(aggregateTokenUsage(events)), [events])

  return (
    <Card title="LLM activity (current run)">
      {rows.length === 0
        ? <p className="text-xs text-zinc-400">No LLM activity yet.</p>
        : (
          <table className="w-full text-xs">
            <thead><tr className="text-left text-zinc-500">
              <th className="py-1">Node</th>
              <th>Activations</th>
              <th>Duration</th>
              <th>Tokens</th>
              <th>Status</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => {
                const tok = tokenMap.get(r.node)
                return (
                  <tr key={r.node} className="border-t border-zinc-100">
                    <td className="py-1 font-mono text-zinc-700">{displayAgentName(r.node)}</td>
                    <td>{r.activations}</td>
                    <td className="font-mono">{(r.total_ms / 1000).toFixed(1)} s</td>
                    <td className="font-mono text-zinc-600">
                      {tok ? `${formatK(tok.inputTokens)} in · ${formatK(tok.outputTokens)} out` : '—'}
                    </td>
                    <td className="text-emerald-700">ok</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )
      }
    </Card>
  )
}
