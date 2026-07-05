'use client'
import { useMemo } from 'react'
import { useRunStore } from '@/stores/run-store'
import { aggregateTokenUsage } from '@/lib/events-aggregate'
import { displayAgentName } from '@/lib/agent-display'
import { formatCost } from '@/lib/format'
import { Card } from '@/components/ui/Card'

interface NodeGroup {
  node: string
  models: string[]
  calls: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  reasoningTokens: number
  reasoningEffort: string
  estimatedCostUsd: number | null
}

export function TokenCostCard() {
  const events = useRunStore((s) => s.events)
  const summary = useMemo(() => aggregateTokenUsage(events), [events])

  const nodeGroups = useMemo((): NodeGroup[] => {
    const map = new Map<string, NodeGroup>()
    for (const r of summary.rows) {
      const prev = map.get(r.node)
      if (prev) {
        if (!prev.models.includes(r.model)) prev.models.push(r.model)
        prev.calls += r.calls
        prev.inputTokens += r.inputTokens
        prev.outputTokens += r.outputTokens
        prev.totalTokens += r.totalTokens
        prev.reasoningTokens += r.reasoningTokens
        if (r.reasoningEffort && !prev.reasoningEffort) prev.reasoningEffort = r.reasoningEffort
        prev.estimatedCostUsd =
          prev.estimatedCostUsd !== null && r.estimatedCostUsd !== null
            ? prev.estimatedCostUsd + r.estimatedCostUsd
            : null
      } else {
        map.set(r.node, {
          node: r.node,
          models: r.model ? [r.model] : [],
          calls: r.calls,
          inputTokens: r.inputTokens,
          outputTokens: r.outputTokens,
          totalTokens: r.totalTokens,
          reasoningTokens: r.reasoningTokens,
          reasoningEffort: r.reasoningEffort,
          estimatedCostUsd: r.estimatedCostUsd,
        })
      }
    }
    return Array.from(map.values())
  }, [summary.rows])

  const hasData = nodeGroups.length > 0

  return (
    <Card
      title="LLM Usage & Estimated Cost"
      actions={
        <span className="font-mono font-semibold text-zinc-800 text-xs">
          {hasData ? `Total ${formatCost(summary.totalCostUsd)}` : '—'}
        </span>
      }
    >
      {!hasData ? (
        <p className="text-xs text-zinc-400">No LLM activity yet.</p>
      ) : (
        <div className="space-y-2">
          {nodeGroups.map((g) => (
            <div key={g.node} className="rounded border border-zinc-200 px-3 py-2 text-xs">
              <p className="mb-2 font-semibold text-zinc-800">{displayAgentName(g.node)}</p>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1">
                <span className="text-zinc-500">
                  Model: <span className="font-mono text-zinc-700">{g.models.join(', ') || '—'}</span>
                </span>
                <span className="text-zinc-500">
                  Calls: <span className="font-semibold text-zinc-700">{g.calls}</span>
                </span>
                <span className="text-zinc-500">
                  Input: <span className="font-mono text-zinc-700">{g.inputTokens.toLocaleString('en-US')}</span>
                </span>
                <span className="text-zinc-500">
                  Output: <span className="font-mono text-zinc-700">{g.outputTokens.toLocaleString('en-US')}</span>
                </span>
                <span className="text-zinc-500">
                  Total: <span className="font-mono text-zinc-700">{g.totalTokens.toLocaleString('en-US')}</span>
                </span>
                <span className="text-zinc-500">
                  Estimated cost:{' '}
                  <span className="font-mono font-semibold text-zinc-800">{formatCost(g.estimatedCostUsd)}</span>
                </span>
                {g.reasoningTokens > 0 && (
                  <span className="col-span-2 text-sky-600">
                    Reasoning:{g.reasoningEffort && <span className="ml-1 font-mono text-sky-500">{g.reasoningEffort}</span>}
                    {' '}<span className="font-mono">{g.reasoningTokens.toLocaleString('en-US')} tokens</span>
                    <span className="ml-2 text-[11px] text-sky-400">(included in output)</span>
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      <p className="mt-3 text-[11px] text-zinc-400">
        Estimated cost based on local pricing table.
        Prices: gpt-5.4-mini $0.75/$4.50 · gpt-5.4-nano $0.20/$1.25 per 1M input/output tokens.
        Update <code className="font-mono">model_pricing.yaml</code> if model prices change.
      </p>
    </Card>
  )
}
