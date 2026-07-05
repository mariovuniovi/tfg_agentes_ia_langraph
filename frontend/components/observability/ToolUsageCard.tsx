'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateToolUsage } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { Card } from '@/components/ui/Card'

export function ToolUsageCard() {
  const events = useRunStore((s) => s.events)
  const rows = useMemo(() => aggregateToolUsage(events), [events])
  if (rows.length === 0) {
    return <Card title="Tool usage (current run)"><p className="text-xs text-zinc-400">No tool calls yet.</p></Card>
  }
  return (
    <Card title="Tool usage (current run)">
      <table className="w-full text-xs">
        <thead><tr className="text-left text-zinc-500"><th>Tool</th><th>Agent</th><th>Calls</th><th>Total ms</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.agent}-${r.tool_name}`} className="border-t border-zinc-100">
              <td className="py-1 font-mono text-zinc-700">{r.tool_name}</td>
              <td className="text-zinc-500">{displayAgentName(r.agent)}</td>
              <td>{r.calls}</td>
              <td className="font-mono">{r.total_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
