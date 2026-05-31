import type { PipelineEvent } from '@/types/api'

// ---------------------------------------------------------------------------
// Tool-owner inference — frontend fallback when backend doesn't emit `node`
// ---------------------------------------------------------------------------

const DATA_VALIDATOR_TOOLS = new Set([
  'load_dataset', 'parse_datetime_column', 'merge_datasets', 'apply_column_mapping',
  'detect_temporal_gaps', 'check_missing_values', 'impute_missing_values',
  'validate_against_schema', 'check_data_quality',
])

const PLANNER_TOOLS = new Set([
  'list_available_models', 'retrieve_similar_experiences', 'retrieve_ml_knowledge',
  'inspect_model_details',
])

export function inferToolOwner(toolName: string): 'data_validator' | 'planner' | 'unknown' {
  if (DATA_VALIDATOR_TOOLS.has(toolName)) return 'data_validator'
  if (PLANNER_TOOLS.has(toolName)) return 'planner'
  return 'unknown'
}

/** Returns the concrete tool name, or null for unnamed internal events. */
export function getConcreteToolName(event: PipelineEvent): string | null {
  const d = event.data as Record<string, unknown>
  const name = d?.tool_name ?? d?.name ?? null
  return typeof name === 'string' && name.trim() !== '' ? name.trim() : null
}

/** Returns node_categories from run_info, with sensible fallbacks for legacy runs. */
export function getNodeCategories(events: PipelineEvent[]): {
  agents: string[]; llm_nodes: string[]; deterministic: string[]; hitl: string[]
} {
  const runInfo = events.find((e) => e.type === 'run_info')
  const cats = (runInfo?.data as { node_categories?: Record<string, string[]> } | undefined)?.node_categories
  if (cats) {
    return {
      agents:        cats.agents        ?? ['data_validator', 'planner'],
      llm_nodes:     cats.llm_nodes     ?? ['report_writer'],
      deterministic: cats.deterministic ?? ['controller', 'executor', 'evaluation', 'deployer'],
      hitl:          cats.hitl          ?? ['dataset_approval', 'deployment_approval'],
    }
  }
  return {
    agents:        ['data_validator', 'planner'],
    llm_nodes:     ['report_writer'],
    deterministic: ['controller', 'executor', 'evaluation', 'deployer'],
    hitl:          ['dataset_approval', 'deployment_approval'],
  }
}

// ---------------------------------------------------------------------------
// Semantic Tool Details ViewModel
// ---------------------------------------------------------------------------

export interface AgentToolRow {
  toolName: string
  calls: number
  totalMs: number
}

export interface AgentSection {
  name: string
  tools: AgentToolRow[]
}

/** Used in ToolDetailsViewModel (camelCase — new semantic layer). */
export interface LlmNodeRow {
  node: string
  activations: number
  totalMs: number
}

/** Used by aggregateLlmNodeActivity (snake_case — observability page compat). */
export interface LlmActivityRow {
  node: string
  activations: number
  total_ms: number
}

export interface TokenUsageEventData {
  node: string
  model: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cached_input_tokens?: number
  estimated_cost_usd: number
  source?: 'langchain_stream_usage_metadata'
}

export interface TokenUsageRow {
  node: string
  model: string
  calls: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  estimatedCostUsd: number
}

export interface TokenUsageSummary {
  rows: TokenUsageRow[]
  totalCostUsd: number
  totalInputTokens: number
  totalOutputTokens: number
}

export function aggregateTokenUsage(events: PipelineEvent[]): TokenUsageSummary {
  const map = new Map<string, TokenUsageRow>()
  for (const e of events) {
    if (e.type !== 'token_usage') continue
    const d = e.data as TokenUsageEventData
    const key = `${d.node}::${d.model}`
    const prev = map.get(key)
    if (prev) {
      prev.calls += 1
      prev.inputTokens += d.input_tokens
      prev.outputTokens += d.output_tokens
      prev.totalTokens += d.total_tokens
      prev.estimatedCostUsd += d.estimated_cost_usd
    } else {
      map.set(key, {
        node: d.node, model: d.model, calls: 1,
        inputTokens: d.input_tokens, outputTokens: d.output_tokens,
        totalTokens: d.total_tokens, estimatedCostUsd: d.estimated_cost_usd,
      })
    }
  }
  const rows = Array.from(map.values())
  return {
    rows,
    totalCostUsd: rows.reduce((s, r) => s + r.estimatedCostUsd, 0),
    totalInputTokens: rows.reduce((s, r) => s + r.inputTokens, 0),
    totalOutputTokens: rows.reduce((s, r) => s + r.outputTokens, 0),
  }
}

export function tokensByNode(
  summary: TokenUsageSummary,
): Map<string, { inputTokens: number; outputTokens: number }> {
  const m = new Map<string, { inputTokens: number; outputTokens: number }>()
  for (const r of summary.rows) {
    const prev = m.get(r.node)
    if (prev) {
      prev.inputTokens += r.inputTokens
      prev.outputTokens += r.outputTokens
    } else {
      m.set(r.node, { inputTokens: r.inputTokens, outputTokens: r.outputTokens })
    }
  }
  return m
}

export type HitlStatus = 'approved' | 'rejected' | 'waiting' | 'none'

export interface HitlGateRow {
  gate: string        // canonical name: dataset_approval / deployment_approval
  status: HitlStatus
}

export interface DeterministicRow {
  node: string
}

export interface ToolDetailsViewModel {
  agents: AgentSection[]
  llmNodes: LlmNodeRow[]
  humanGates: HitlGateRow[]
  deterministic: DeterministicRow[]
}

const HITL_TYPE_TO_GATE: Record<string, string> = {
  data_validation: 'dataset_approval',
  dataset_approval: 'dataset_approval',
  deployment: 'deployment_approval',
  deployment_approval: 'deployment_approval',
  deployer: 'deployment_approval',
}

export function buildToolDetailsViewModel(events: PipelineEvent[]): ToolDetailsViewModel {
  const cats = getNodeCategories(events)
  const agentSet = new Set(cats.agents)
  const llmSet = new Set(cats.llm_nodes)

  // --- agents: collect tool calls per owner ---
  const agentTools = new Map<string, Map<string, AgentToolRow>>()
  for (const agent of cats.agents) agentTools.set(agent, new Map())

  for (const e of events) {
    if (e.type !== 'tool_result') continue
    const toolName = getConcreteToolName(e)
    if (!toolName) continue

    // Prefer explicit node on the event, fall back to inference
    const rawNode = (e.data as { node?: string }).node
    const owner = rawNode && agentSet.has(rawNode)
      ? rawNode
      : inferToolOwner(toolName)
    if (!agentSet.has(owner)) continue

    const nodeMap = agentTools.get(owner)!
    const dur = Number((e.data as { duration_ms?: number }).duration_ms ?? 0)
    const prev = nodeMap.get(toolName)
    if (prev) { prev.calls += 1; prev.totalMs += dur }
    else nodeMap.set(toolName, { toolName, calls: 1, totalMs: dur })
  }

  const agentSections: AgentSection[] = cats.agents
    .map((name) => ({ name, tools: Array.from(agentTools.get(name)?.values() ?? []) }))
    .filter((s) => s.tools.length > 0)

  // --- LLM nodes: measure time between routing-in and next routing event ---
  const llmMap = new Map<string, LlmNodeRow>()
  let activeNode: string | null = null
  let activeStartMs = 0
  for (const e of events) {
    if (e.type !== 'routing') continue
    const next = (e.data as { next?: string }).next ?? ''
    if (activeNode && llmSet.has(activeNode)) {
      const dur = e.timestamp_ms - activeStartMs
      const prev = llmMap.get(activeNode)
      if (prev) { prev.activations += 1; prev.totalMs += dur }
      else llmMap.set(activeNode, { node: activeNode, activations: 1, totalMs: dur })
    }
    activeNode = next
    activeStartMs = e.timestamp_ms
  }
  const llmNodes = Array.from(llmMap.values())

  // --- Human gates ---
  const gateMap = new Map<string, HitlStatus>()
  for (const e of events) {
    if (e.type === 'hitl_request') {
      const raw = ((e.data as { type?: string }).type ?? e.agent ?? '').toLowerCase()
      const gate = HITL_TYPE_TO_GATE[raw] ?? raw
      if (!gateMap.has(gate)) gateMap.set(gate, 'waiting')
    }
    if (e.type === 'routing') {
      const next = ((e.data as { next?: string }).next ?? '').toLowerCase()
      if (HITL_TYPE_TO_GATE[next]) {
        const gate = HITL_TYPE_TO_GATE[next]
        if (!gateMap.has(gate)) gateMap.set(gate, 'waiting')
      }
    }
  }
  // Mark approved once a post-approval routing away from the gate is seen
  for (const e of events) {
    if (e.type === 'routing') {
      const next = (e.data as { next?: string }).next ?? ''
      if (next === 'planner' || next === 'executor') {
        if (gateMap.get('dataset_approval') === 'waiting') gateMap.set('dataset_approval', 'approved')
      }
      if (next === 'deployer') {
        if (gateMap.get('deployment_approval') === 'waiting') gateMap.set('deployment_approval', 'approved')
      }
    }
    if (e.type === 'run_complete') {
      // If gate was waiting and run completed, treat as approved (approval must have occurred)
      for (const [gate, status] of gateMap.entries()) {
        if (status === 'waiting') gateMap.set(gate, 'approved')
      }
    }
  }
  const humanGates: HitlGateRow[] = Array.from(gateMap.entries()).map(([gate, status]) => ({ gate, status }))

  // --- Deterministic: show nodes that appear in routing events ---
  const detSet = new Set(cats.deterministic)
  const seenDet = new Set<string>()
  for (const e of events) {
    if (e.type === 'routing') {
      const next = (e.data as { next?: string }).next ?? ''
      if (detSet.has(next)) seenDet.add(next)
    }
    if (e.type === 'training_complete') seenDet.add('executor')
    if (e.type === 'audit_report') seenDet.add('report_writer')  // only if actually in det
    if (e.type === 'deployment_complete') seenDet.add('deployer')
  }
  const deterministic: DeterministicRow[] = cats.deterministic
    .filter((n) => seenDet.has(n))
    .map((n) => ({ node: n }))

  return { agents: agentSections, llmNodes, humanGates, deterministic }
}

// ---------------------------------------------------------------------------
// Legacy exports (used by observability page)
// ---------------------------------------------------------------------------

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
    const tool = getConcreteToolName(e)
    if (!tool) continue
    const key = `${e.agent}::${tool}`
    const dur = Number((e.data as { duration_ms?: number }).duration_ms ?? 0)
    const prev = map.get(key)
    if (prev) { prev.calls += 1; prev.total_ms += dur }
    else map.set(key, { agent: e.agent, tool_name: tool, calls: 1, total_ms: dur })
  }
  return Array.from(map.values())
}

export function aggregateLlmNodeActivity(events: PipelineEvent[], llmNodes: string[]): LlmActivityRow[] {
  const set = new Set(llmNodes)
  const map = new Map<string, LlmActivityRow>()
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

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

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
        const cats = getNodeCategories([e])
        rows.push({ ts: t, text: `Pipeline started · agents: ${cats.agents.join(', ')}` })
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
        const tn = getConcreteToolName(e)
        if (!tn) break  // skip unnamed internal events
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
      case 'deployment_complete': {
        const uri = (e.data as { best_model_uri?: string }).best_model_uri ?? ''
        const m = uri.match(/models:\/([^/]+)\/(\d+)/)
        const human = m ? `${m[1]} v${m[2]}` : 'model'
        rows.push({ ts: t, text: `Deployment complete · ${human} (champion alias set)`, agent: 'deployer' })
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
