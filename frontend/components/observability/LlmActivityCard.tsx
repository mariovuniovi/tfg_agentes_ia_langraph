'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateLlmNodeActivity } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { Card } from '@/components/ui/Card'

export function LlmActivityCard() {
  const events = useRunStore((s) => s.events)
  const llmNodes = useMemo(() => {
    const info = events.find((e) => e.type === 'run_info')
    return Object.keys((info?.data as { models?: Record<string, string> } | undefined)?.models ?? {})
  }, [events])
  const rows = useMemo(() => aggregateLlmNodeActivity(events, llmNodes), [events, llmNodes])

  return (
    <Card title="LLM activity (current run)">
      {rows.length === 0
        ? <p className="text-xs text-zinc-400">No LLM activity yet.</p>
        : (
          <table className="w-full text-xs">
            <thead><tr className="text-left text-zinc-500">
              <th className="py-1">Node</th><th>Activations</th><th>Duration</th><th>Tokens</th><th>Status</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.node} className="border-t border-zinc-100">
                  <td className="py-1 font-mono text-zinc-700">{displayAgentName(r.node)}</td>
                  <td>{r.activations}</td>
                  <td className="font-mono">{(r.total_ms / 1000).toFixed(1)} s</td>
                  <td className="text-zinc-400">—</td>
                  <td className="text-emerald-700">ok</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      <p className="mt-2 text-[11px] text-zinc-400">Token counts shown when available — none recorded in current event stream.</p>
    </Card>
  )
}
