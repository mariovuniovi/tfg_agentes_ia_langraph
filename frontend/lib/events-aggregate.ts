import type { PipelineEvent } from '@/types/api'

export interface ToolUsageRow {
  agent: string
  tool_name: string
  calls: number
  total_ms: number
}

export function aggregateToolUsage(events: PipelineEvent[]): ToolUsageRow[] {
  const map = new Map<string, ToolUsageRow>()
  for (const e of events) {
    if (e.type !== 'tool_result') continue
    const tool = (e.data as { tool_name?: string }).tool_name
    if (!tool) continue
    const key = `${e.agent}::${tool}`
    const dur = Number((e.data as { duration_ms?: number }).duration_ms ?? 0)
    const prev = map.get(key)
    if (prev) { prev.calls += 1; prev.total_ms += dur }
    else map.set(key, { agent: e.agent, tool_name: tool, calls: 1, total_ms: dur })
  }
  return Array.from(map.values())
}

export interface LlmNodeRow {
  node: string
  activations: number
  total_ms: number
}

export function aggregateLlmNodeActivity(events: PipelineEvent[], llmNodes: string[]): LlmNodeRow[] {
  const set = new Set(llmNodes)
  const map = new Map<string, LlmNodeRow>()
  let activeNode: string | null = null
  let activeStartMs = 0
  for (const e of events) {
    if (e.type !== 'routing') continue
    const next = (e.data as { next?: string }).next ?? ''
    if (activeNode && set.has(activeNode)) {
      const dur = e.timestamp_ms - activeStartMs
      const prev = map.get(activeNode)
      if (prev) { prev.activations += 1; prev.total_ms += dur }
      else map.set(activeNode, { node: activeNode, activations: 1, total_ms: dur })
    }
    activeNode = next
    activeStartMs = e.timestamp_ms
  }
  return Array.from(map.values())
}

export interface TimelineRow {
  ts: number
  text: string
  agent?: string
}

export function buildTimeline(events: PipelineEvent[]): TimelineRow[] {
  const rows: TimelineRow[] = []
  let lastRoutingNext = ''
  for (const e of events) {
    const t = e.timestamp_ms
    switch (e.type) {
      case 'run_info': {
        const models = Object.keys((e.data as { models?: Record<string, string> }).models ?? {})
        rows.push({ ts: t, text: `Pipeline started · LLM nodes: ${models.join(', ') || '—'}` })
        break
      }
      case 'routing': {
        const next = (e.data as { next?: string }).next ?? ''
        if (next && next !== lastRoutingNext) {
          rows.push({ ts: t, text: `Workflow moved to ${next}`, agent: 'controller' })
          lastRoutingNext = next
        }
        break
      }
      case 'tool_result': {
        const tn = (e.data as { tool_name?: string }).tool_name
        if (tn === 'load_dataset') {
          const r = (e.data as { result?: string }).result
          let summary = 'Dataset loaded'
          try { const j = JSON.parse(r ?? '{}'); summary = `Dataset loaded · ${j.row_count} rows × ${j.column_names?.length ?? '?'} cols` } catch {}
          rows.push({ ts: t, text: summary, agent: e.agent })
        }
        if (tn === 'validate_against_schema') {
          rows.push({ ts: t, text: 'Validation completed', agent: e.agent })
        }
        if (tn === 'train_model') {
          rows.push({ ts: t, text: 'Training completed', agent: e.agent })
        }
        break
      }
      case 'planner_context': {
        const cands = (e.data as { plan_summary?: { candidate_models?: string[] } }).plan_summary?.candidate_models ?? []
        rows.push({ ts: t, text: `Planner selected ${cands.length} candidates`, agent: 'planner' })
        break
      }
      case 'hitl_request': {
        const gate = (e.data as { type?: string }).type ?? ''
        rows.push({ ts: t, text: `${gate} approval requested`, agent: e.agent })
        break
      }
      case 'audit_report': {
        rows.push({ ts: t, text: 'Audit report generated', agent: 'report_writer' })
        break
      }
      case 'run_complete': {
        const err = (e.data as { error?: string }).error
        rows.push({ ts: t, text: err ? `Run failed: ${err.slice(0, 80)}` : 'Run complete', agent: e.agent })
        break
      }
    }
  }
  return rows
}
