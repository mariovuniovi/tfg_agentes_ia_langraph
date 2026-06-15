import { describe, it, expect } from 'vitest'
import {
  aggregateToolUsage, aggregateLlmNodeActivity,
  inferToolOwner, getConcreteToolName, buildToolDetailsViewModel,
  aggregateTokenUsage, tokensByNode,
  type LlmActivityRow, type TokenUsageSummary,
} from '@/lib/events-aggregate'
import type { PipelineEvent } from '@/types/api'

function ev(type: string, agent: string, data: Record<string, unknown> = {}, ms = 0): PipelineEvent {
  return { type, agent, timestamp_ms: ms, data } as PipelineEvent
}

// ---------------------------------------------------------------------------
// Legacy aggregateToolUsage (observability page compat)
// ---------------------------------------------------------------------------

describe('aggregateToolUsage', () => {
  it('groups by agent + tool name with call counts and total ms', () => {
    const events = [
      ev('tool_call',   'data_validator', { tool_name: 'load_dataset' }, 0),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 200 }, 200),
      ev('tool_call',   'data_validator', { tool_name: 'load_dataset' }, 300),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 216 }, 516),
    ]
    const rows = aggregateToolUsage(events)
    expect(rows).toEqual([
      { agent: 'data_validator', tool_name: 'load_dataset', calls: 2, total_ms: 416 },
    ])
  })

  it('ignores unnamed tool_call and tool_result events', () => {
    const events = [
      ev('tool_call',   'model', {}, 0),          // unnamed — no tool_name
      ev('tool_result', 'tools', {}, 100),         // unnamed — no tool_name
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 50 }, 200),
    ]
    const rows = aggregateToolUsage(events)
    expect(rows).toHaveLength(1)
    expect(rows[0].tool_name).toBe('load_dataset')
  })
})

describe('aggregateLlmNodeActivity', () => {
  // `routing(next=X)` marks the END of node X; X's duration is the span from the
  // previous routing event (end of the prior phase) to routing(next=X). The first
  // node uses the run start (first event) as its origin.
  it('attributes each routing window to the node that just finished', () => {
    const events = [
      ev('run_info', 'system', {}, 0),
      ev('routing', 'controller', { next: 'data_validator' }, 10000),   // DV ran [0, 10000]
      ev('routing', 'controller', { next: 'dataset_approval' }, 15000), // HITL wait [10000, 15000]
      ev('routing', 'controller', { next: 'planner' }, 40000),          // planner ran [15000, 40000]
      ev('routing', 'controller', { next: 'executor' }, 50000),         // executor ran [40000, 50000]
      ev('routing', 'controller', { next: 'report_writer' }, 56000),    // report ran [50000, 56000]
    ]
    const rows: LlmActivityRow[] = aggregateLlmNodeActivity(
      events, ['data_validator', 'planner', 'report_writer'],
    )
    const byNode = Object.fromEntries(rows.map((r) => [r.node, r]))
    expect(byNode.data_validator).toMatchObject({ activations: 1, total_ms: 10000 })
    expect(byNode.planner).toMatchObject({ activations: 1, total_ms: 25000 })
    expect(byNode.report_writer).toMatchObject({ activations: 1, total_ms: 6000 })
  })

  it('does not charge the HITL-gate wait to data_validator (regression)', () => {
    const events = [
      ev('run_info', 'system', {}, 0),
      ev('routing', 'controller', { next: 'data_validator' }, 2000),     // DV ran [0, 2000]
      ev('routing', 'controller', { next: 'dataset_approval' }, 90000),  // 88s human wait — NOT DV
      ev('routing', 'controller', { next: 'planner' }, 95000),
    ]
    const dv = aggregateLlmNodeActivity(events, ['data_validator', 'planner'])
      .find((r) => r.node === 'data_validator')!
    expect(dv.total_ms).toBe(2000)
  })
})

// ---------------------------------------------------------------------------
// New semantic helpers
// ---------------------------------------------------------------------------

describe('getConcreteToolName', () => {
  it('returns tool name when present', () => {
    expect(getConcreteToolName(ev('tool_call', 'x', { tool_name: 'load_dataset' }))).toBe('load_dataset')
  })

  it('returns null when tool_name is absent', () => {
    expect(getConcreteToolName(ev('tool_call', 'model', {}))).toBeNull()
  })

  it('returns null when tool_name is empty string', () => {
    expect(getConcreteToolName(ev('tool_call', 'model', { tool_name: '' }))).toBeNull()
  })
})

describe('inferToolOwner', () => {
  it('maps data-validator tools to data_validator', () => {
    expect(inferToolOwner('load_dataset')).toBe('data_validator')
    expect(inferToolOwner('parse_datetime_column')).toBe('data_validator')
    expect(inferToolOwner('validate_against_schema')).toBe('data_validator')
    expect(inferToolOwner('check_data_quality')).toBe('data_validator')
  })

  it('maps planner tools to planner', () => {
    expect(inferToolOwner('list_available_models')).toBe('planner')
    expect(inferToolOwner('retrieve_similar_experiences')).toBe('planner')
    expect(inferToolOwner('retrieve_ml_knowledge')).toBe('planner')
    expect(inferToolOwner('inspect_model_details')).toBe('planner')
  })

  it('returns unknown for unrecognised tools', () => {
    expect(inferToolOwner('train_model')).toBe('unknown')
    expect(inferToolOwner('whatever')).toBe('unknown')
  })
})

// ---------------------------------------------------------------------------
// buildToolDetailsViewModel — the main semantic aggregation
// ---------------------------------------------------------------------------

function makeRunInfo(extra: Record<string, unknown> = {}): PipelineEvent {
  return ev('run_info', 'system', {
    node_categories: {
      agents: ['data_validator', 'planner'],
      llm_nodes: ['report_writer'],
      deterministic: ['controller', 'executor', 'evaluation', 'deployer'],
      hitl: ['dataset_approval', 'deployment_approval'],
    },
    ...extra,
  })
}

describe('buildToolDetailsViewModel', () => {
  it('groups data_validator tools under Agents, never under LLM nodes', () => {
    const events = [
      makeRunInfo(),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 200 }, 100),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 210 }, 400),
      ev('tool_result', 'data_validator', { tool_name: 'validate_against_schema', duration_ms: 50 }, 600),
    ]
    const vm = buildToolDetailsViewModel(events)
    const dvSection = vm.agents.find((s) => s.name === 'data_validator')
    expect(dvSection).toBeDefined()
    const loadRow = dvSection!.tools.find((t) => t.toolName === 'load_dataset')
    expect(loadRow!.calls).toBe(2)
    expect(loadRow!.totalMs).toBe(410)
    // data_validator must never appear in llmNodes
    expect(vm.llmNodes.map((n) => n.node)).not.toContain('data_validator')
  })

  it('groups planner tools under Agents → planner, never under LLM nodes', () => {
    const events = [
      makeRunInfo(),
      ev('tool_result', 'model', { tool_name: 'list_available_models', duration_ms: 30 }, 100),
      ev('tool_result', 'model', { tool_name: 'retrieve_similar_experiences', duration_ms: 40 }, 200),
      ev('tool_result', 'model', { tool_name: 'retrieve_ml_knowledge', duration_ms: 20 }, 300),
    ]
    const vm = buildToolDetailsViewModel(events)
    const planSection = vm.agents.find((s) => s.name === 'planner')
    expect(planSection).toBeDefined()
    expect(planSection!.tools.find((t) => t.toolName === 'retrieve_ml_knowledge')!.calls).toBe(1)
    expect(vm.llmNodes.map((n) => n.node)).not.toContain('planner')
  })

  it('shows report_writer under LLM nodes only', () => {
    const events = [
      makeRunInfo(),
      ev('routing', 'controller', { next: 'report_writer' }, 1000),
      ev('routing', 'controller', { next: 'deployer' }, 3000),
    ]
    const vm = buildToolDetailsViewModel(events)
    expect(vm.llmNodes.find((n) => n.node === 'report_writer')?.activations).toBe(1)
    expect(vm.agents.map((s) => s.name)).not.toContain('report_writer')
  })

  it('ignores unnamed tool_call events (no tool_name)', () => {
    const events = [
      makeRunInfo(),
      ev('tool_result', 'model', {}, 50),            // unnamed — must be ignored
      ev('tool_result', 'model', { tool_name: '' }, 100),  // empty name — must be ignored
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 100 }, 200),
    ]
    const vm = buildToolDetailsViewModel(events)
    const dvSection = vm.agents.find((s) => s.name === 'data_validator')
    expect(dvSection!.tools).toHaveLength(1)
    expect(dvSection!.tools[0].toolName).toBe('load_dataset')
  })

  it('maps hitl_request → dataset_approval gate', () => {
    const events = [
      makeRunInfo(),
      ev('hitl_request', 'data_validator', { type: 'data_validation' }, 100),
      ev('routing', 'controller', { next: 'planner' }, 200),  // triggers approved status
    ]
    const vm = buildToolDetailsViewModel(events)
    const gate = vm.humanGates.find((g) => g.gate === 'dataset_approval')
    expect(gate).toBeDefined()
    expect(gate!.status).toBe('approved')
  })

  it('shows executor/evaluation under deterministic when routing events exist', () => {
    const events = [
      makeRunInfo(),
      ev('routing', 'controller', { next: 'executor' }, 100),
      ev('routing', 'controller', { next: 'evaluation' }, 200),
      ev('training_complete', 'executor', {}, 300),
    ]
    const vm = buildToolDetailsViewModel(events)
    const nodeNames = vm.deterministic.map((d) => d.node)
    expect(nodeNames).toContain('executor')
    expect(nodeNames).toContain('evaluation')
  })

  it('aggregates multiple tool calls of same type correctly', () => {
    const events = [
      makeRunInfo(),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 100 }, 100),
      ev('tool_result', 'data_validator', { tool_name: 'parse_datetime_column', duration_ms: 50 }, 200),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 110 }, 300),
      ev('tool_result', 'data_validator', { tool_name: 'parse_datetime_column', duration_ms: 55 }, 400),
    ]
    const vm = buildToolDetailsViewModel(events)
    const tools = vm.agents.find((s) => s.name === 'data_validator')!.tools
    expect(tools.find((t) => t.toolName === 'load_dataset')!.calls).toBe(2)
    expect(tools.find((t) => t.toolName === 'parse_datetime_column')!.calls).toBe(2)
  })

  it('falls back to hardcoded taxonomy when run_info is absent', () => {
    const events = [
      ev('tool_result', 'model', { tool_name: 'list_available_models', duration_ms: 20 }, 100),
    ]
    const vm = buildToolDetailsViewModel(events)
    // Should still find planner section via inference fallback
    const planSection = vm.agents.find((s) => s.name === 'planner')
    expect(planSection!.tools[0].toolName).toBe('list_available_models')
  })
})

// ---------------------------------------------------------------------------
// aggregateTokenUsage
// ---------------------------------------------------------------------------

describe('aggregateTokenUsage', () => {
  it('returns empty summary when no token_usage events', () => {
    const summary = aggregateTokenUsage([])
    expect(summary.rows).toHaveLength(0)
    expect(summary.totalCostUsd).toBe(0)
    expect(summary.totalInputTokens).toBe(0)
    expect(summary.totalOutputTokens).toBe(0)
  })

  it('creates one row for a single token_usage event', () => {
    const events = [
      ev('token_usage', 'planner', {
        node: 'planner', model: 'gpt-5.4-mini',
        input_tokens: 1000, output_tokens: 200, total_tokens: 1200,
        estimated_cost_usd: 0.00165,
      }),
    ]
    const summary = aggregateTokenUsage(events)
    expect(summary.rows).toHaveLength(1)
    expect(summary.rows[0]).toMatchObject({
      node: 'planner', model: 'gpt-5.4-mini', calls: 1,
      inputTokens: 1000, outputTokens: 200, totalTokens: 1200,
      estimatedCostUsd: 0.00165,
    })
    expect(summary.totalCostUsd).toBeCloseTo(0.00165)
    expect(summary.totalInputTokens).toBe(1000)
    expect(summary.totalOutputTokens).toBe(200)
  })

  it('accumulates multiple calls for the same node+model', () => {
    const events = [
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4-mini', input_tokens: 1000, output_tokens: 200, total_tokens: 1200, estimated_cost_usd: 0.001 }),
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4-mini', input_tokens: 2000, output_tokens: 400, total_tokens: 2400, estimated_cost_usd: 0.002 }),
    ]
    const summary = aggregateTokenUsage(events)
    expect(summary.rows).toHaveLength(1)
    expect(summary.rows[0].calls).toBe(2)
    expect(summary.rows[0].inputTokens).toBe(3000)
    expect(summary.rows[0].outputTokens).toBe(600)
    expect(summary.rows[0].totalTokens).toBe(3600)
    expect(summary.totalCostUsd).toBeCloseTo(0.003)
  })

  it('creates separate rows for different nodes', () => {
    const events = [
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4-mini', input_tokens: 1000, output_tokens: 200, total_tokens: 1200, estimated_cost_usd: 0.001 }),
      ev('token_usage', 'report_writer', { node: 'report_writer', model: 'gpt-5.4-nano', input_tokens: 500, output_tokens: 100, total_tokens: 600, estimated_cost_usd: 0.0002 }),
    ]
    const summary = aggregateTokenUsage(events)
    expect(summary.rows).toHaveLength(2)
    expect(summary.totalCostUsd).toBeCloseTo(0.0012)
  })

  it('creates separate rows for same node but different models', () => {
    const events = [
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4-mini', input_tokens: 1000, output_tokens: 200, total_tokens: 1200, estimated_cost_usd: 0.001 }),
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4', input_tokens: 500, output_tokens: 100, total_tokens: 600, estimated_cost_usd: 0.009 }),
    ]
    const summary = aggregateTokenUsage(events)
    expect(summary.rows).toHaveLength(2)
  })

  it('ignores non-token_usage events', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }),
      ev('tool_result', 'data_validator', { tool_name: 'load_dataset', duration_ms: 100 }),
      ev('token_usage', 'planner', { node: 'planner', model: 'gpt-5.4-mini', input_tokens: 500, output_tokens: 100, total_tokens: 600, estimated_cost_usd: 0.001 }),
    ]
    const summary = aggregateTokenUsage(events)
    expect(summary.rows).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// tokensByNode
// ---------------------------------------------------------------------------

describe('tokensByNode', () => {
  it('returns empty map for empty summary', () => {
    const m = tokensByNode({ rows: [], totalCostUsd: 0, totalInputTokens: 0, totalOutputTokens: 0 })
    expect(m.size).toBe(0)
  })

  it('collapses multiple models for the same node into one entry', () => {
    const summary: TokenUsageSummary = {
      rows: [
        { node: 'planner', model: 'gpt-5.4-mini', calls: 1, inputTokens: 1000, outputTokens: 200, totalTokens: 1200, estimatedCostUsd: 0.001 },
        { node: 'planner', model: 'gpt-5.4', calls: 1, inputTokens: 500, outputTokens: 100, totalTokens: 600, estimatedCostUsd: 0.009 },
      ],
      totalCostUsd: 0.01, totalInputTokens: 1500, totalOutputTokens: 300,
    }
    const m = tokensByNode(summary)
    expect(m.get('planner')).toEqual({ inputTokens: 1500, outputTokens: 300 })
  })

  it('keeps separate nodes separate', () => {
    const summary: TokenUsageSummary = {
      rows: [
        { node: 'planner', model: 'gpt-5.4-mini', calls: 1, inputTokens: 1000, outputTokens: 200, totalTokens: 1200, estimatedCostUsd: 0.001 },
        { node: 'report_writer', model: 'gpt-5.4-nano', calls: 1, inputTokens: 500, outputTokens: 100, totalTokens: 600, estimatedCostUsd: 0.0002 },
      ],
      totalCostUsd: 0.0012, totalInputTokens: 1500, totalOutputTokens: 300,
    }
    const m = tokensByNode(summary)
    expect(m.size).toBe(2)
    expect(m.get('planner')).toEqual({ inputTokens: 1000, outputTokens: 200 })
    expect(m.get('report_writer')).toEqual({ inputTokens: 500, outputTokens: 100 })
  })
})
