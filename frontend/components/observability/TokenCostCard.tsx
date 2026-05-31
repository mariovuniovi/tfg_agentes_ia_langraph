'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateTokenUsage } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { formatCost } from '@/lib/format'
import { Card } from '@/components/ui/Card'

export function TokenCostCard() {
  const events = useRunStore((s) => s.events)
  const summary = useMemo(() => aggregateTokenUsage(events), [events])
  const hasData = summary.rows.length > 0

  return (
    <Card
      title="Token Cost (current run)"
      actions={
        <span className="font-mono font-semibold text-zinc-800 text-xs">
          {hasData ? `Total: ${formatCost(summary.totalCostUsd)}` : '—'}
        </span>
      }
    >
      {!hasData ? (
        <p className="text-xs text-zinc-400">No LLM activity yet.</p>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-zinc-500">
              <th className="py-1">Node</th>
              <th>Model</th>
              <th className="text-right">Calls</th>
              <th className="text-right">Input</th>
              <th className="text-right">Output</th>
              <th className="text-right">Total</th>
              <th className="text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {summary.rows.map((r) => (
              <tr key={`${r.node}::${r.model}`} className="border-t border-zinc-100">
                <td className="py-1 font-mono text-zinc-700">{displayAgentName(r.node)}</td>
                <td className="font-mono text-zinc-500">{r.model}</td>
                <td className="text-right text-zinc-500">{r.calls}</td>
                <td className="text-right font-mono text-zinc-600">{r.inputTokens.toLocaleString()}</td>
                <td className="text-right font-mono text-zinc-600">{r.outputTokens.toLocaleString()}</td>
                <td className="text-right font-mono text-zinc-600">{r.totalTokens.toLocaleString()}</td>
                <td className="text-right font-mono font-semibold text-zinc-800">{formatCost(r.estimatedCostUsd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="mt-2 text-[11px] text-zinc-400">
        Estimated cost based on local pricing table.
        Prices: gpt-5.4-mini $0.75/$4.50 · gpt-5.4-nano $0.20/$1.25 per 1M input/output tokens.
        Update <code className="font-mono">model_pricing.yaml</code> if model prices change.
      </p>
    </Card>
  )
}
