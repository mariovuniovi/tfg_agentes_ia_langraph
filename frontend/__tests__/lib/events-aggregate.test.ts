import { describe, it, expect } from 'vitest'
import {
  aggregateToolUsage, aggregateLlmNodeActivity,
  inferToolOwner, getConcreteToolName, buildToolDetailsViewModel,
  type LlmActivityRow,
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
  it('measures duration between routing-in and next routing-out', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }, 1000),
      ev('routing', 'controller', { next: 'executor' }, 24400),
    ]
    const rows: LlmActivityRow[] = aggregateLlmNodeActivity(events, ['planner'])
    expect(rows[0]).toMatchObject({ node: 'planner', activations: 1, total_ms: 23400 })
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
