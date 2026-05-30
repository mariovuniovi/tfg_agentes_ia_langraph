import { describe, it, expect } from 'vitest'
import { aggregateToolUsage, aggregateLlmNodeActivity } from '@/lib/events-aggregate'
import type { PipelineEvent } from '@/types/api'

function ev(type: string, agent: string, data: Record<string, unknown> = {}, ms = 0): PipelineEvent {
  return { type, agent, timestamp_ms: ms, data } as PipelineEvent
}

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
})

describe('aggregateLlmNodeActivity', () => {
  it('measures duration between routing-in and next routing-out', () => {
    const events = [
      ev('routing', 'controller', { next: 'planner' }, 1000),
      ev('routing', 'controller', { next: 'executor' }, 24400),
    ]
    const rows = aggregateLlmNodeActivity(events, ['planner'])
    expect(rows[0]).toMatchObject({ node: 'planner', activations: 1, total_ms: 23400 })
  })
})
