# Token Cost Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-node token usage tracking and estimated cost display to the existing observability dashboard, using only the existing SSE event pipeline — no external services.

**Architecture:** The final `AIMessageChunk` in each LangChain streaming call already carries `usage_metadata`. We extend `parse_stream_event` in `pipeline_helpers.py` to detect this final chunk and emit a new `token_usage` SSE event (with model, token counts, and pre-computed cost). The frontend accumulates these events in the existing `events[]` array and renders them in an upgraded `LlmActivityCard` (token column) and a new `TokenCostCard` (per-node cost breakdown + total).

**Tech Stack:** Python / pyyaml (pricing YAML), LangChain `AIMessageChunk.usage_metadata`, TypeScript / Vitest, React (Next.js), Zustand run-store.

---

## File Map

| File | Role |
|---|---|
| `pyproject.toml` | Add `pyyaml>=6.0` (runtime) + `types-PyYAML` (dev/mypy) |
| `src/mlops_agents/observability/__init__.py` | Package marker (empty) |
| `src/mlops_agents/observability/model_pricing.yaml` | Manually-maintained price table (gpt-5.4-mini, gpt-5.4-nano) |
| `src/mlops_agents/observability/pricing.py` | `estimate_cost(model, in, out, cached=0) → float` + `_normalize()` |
| `api/services/pipeline_helpers.py` | Extend `parse_stream_event` with `elif usage_metadata` branch |
| `frontend/types/api.ts` | Add `'token_usage'` to `PipelineEventType` |
| `frontend/lib/format.ts` | `formatK(n)` + `formatCost(usd)` pure helpers |
| `frontend/lib/events-aggregate.ts` | Add `aggregateTokenUsage`, `tokensByNode`, three new interfaces |
| `frontend/components/observability/LlmActivityCard.tsx` | Fill token column (`8.2k in · 1.1k out`), remove placeholder note |
| `frontend/components/observability/TokenCostCard.tsx` | New card: per-node cost table + total |
| `frontend/app/observability/page.tsx` | Add `<TokenCostCard />` |
| `tests/test_observability/__init__.py` | Package marker (empty) |
| `tests/test_observability/test_pricing.py` | Unit tests for `pricing.py` |
| `frontend/__tests__/lib/events-aggregate.test.ts` | Add `aggregateTokenUsage` + `tokensByNode` tests |
| `frontend/__tests__/lib/format.test.ts` | Tests for `formatK` + `formatCost` |

---

## Task 1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml to runtime deps and types-PyYAML to dev**

Open `pyproject.toml`. In the `dependencies = [` block, after the `# HTTP & Logging` section, add:

```toml
    # Observability
    "pyyaml>=6.0",
```

In the `[dependency-groups]` dev block, add:

```toml
    "types-PyYAML>=6.0",
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync
```

Expected: Lock file updated, no errors. `python -c "import yaml; print(yaml.__version__)"` prints a version.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyyaml runtime dep and types-PyYAML for mypy"
```

---

## Task 2: Pricing module (TDD)

**Files:**
- Create: `src/mlops_agents/observability/__init__.py`
- Create: `src/mlops_agents/observability/model_pricing.yaml`
- Create: `src/mlops_agents/observability/pricing.py`
- Create: `tests/test_observability/__init__.py`
- Create: `tests/test_observability/test_pricing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_observability/__init__.py` (empty file).

Create `tests/test_observability/test_pricing.py`:

```python
import pytest
from mlops_agents.observability.pricing import estimate_cost, _normalize


def test_normalize_strips_date_suffix():
    assert _normalize("gpt-5.4-mini-2025-11-01") == "gpt-5.4-mini"


def test_normalize_strips_provider_prefix():
    assert _normalize("openai/gpt-5.4-mini") == "gpt-5.4-mini"


def test_normalize_strips_both():
    assert _normalize("openai/gpt-5.4-mini-2025-11-01") == "gpt-5.4-mini"


def test_normalize_plain_name_unchanged():
    assert _normalize("gpt-5.4-mini") == "gpt-5.4-mini"


def test_unknown_model_returns_zero():
    assert estimate_cost("gpt-unknown-xyz", 1000, 1000) == 0.0


def test_mini_cost_per_million():
    # gpt-5.4-mini: $0.75 input + $4.50 output per 1M
    cost = estimate_cost("gpt-5.4-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(5.25)


def test_nano_cost_per_million():
    # gpt-5.4-nano: $0.20 input + $1.25 output per 1M
    cost = estimate_cost("gpt-5.4-nano", 1_000_000, 1_000_000)
    assert cost == pytest.approx(1.45)


def test_mini_small_call():
    # 1000 input + 200 output tokens
    cost = estimate_cost("gpt-5.4-mini", 1000, 200)
    assert cost == pytest.approx(1000 * 0.75 / 1_000_000 + 200 * 4.50 / 1_000_000)
    # = 0.00075 + 0.00090 = 0.00165


def test_date_suffix_still_matches():
    cost_plain = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_dated = estimate_cost("gpt-5.4-mini-2025-11-01", 1000, 1000)
    assert cost_plain == pytest.approx(cost_dated)


def test_provider_prefix_still_matches():
    cost_plain = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_prefixed = estimate_cost("openai/gpt-5.4-mini", 1000, 1000)
    assert cost_plain == pytest.approx(cost_prefixed)


def test_cached_tokens_zero_cost_when_not_published():
    # cached_input_per_1m = 0.0 in YAML, so cached tokens add nothing
    cost_without = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_with = estimate_cost("gpt-5.4-mini", 1000, 1000, cached_input_tokens=500)
    assert cost_without == pytest.approx(cost_with)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_observability/test_pricing.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (pricing.py doesn't exist yet).

- [ ] **Step 3: Create the observability package**

Create `src/mlops_agents/observability/__init__.py` (empty file).

Create `src/mlops_agents/observability/model_pricing.yaml`:

```yaml
# Estimated costs in USD per 1M tokens
# Source: https://platform.openai.com/docs/pricing (checked 2026-05-31)
# Update this file when model pricing changes or new models are added.
# Restart the API server after editing (values are cached at startup).

gpt-5.4-mini:
  input_per_1m: 0.75
  output_per_1m: 4.50
  cached_input_per_1m: 0.0   # not published; update when available

gpt-5.4-nano:
  input_per_1m: 0.20
  output_per_1m: 1.25
  cached_input_per_1m: 0.0   # not published; update when available
```

Create `src/mlops_agents/observability/pricing.py`:

```python
import re
from functools import lru_cache
from pathlib import Path

import yaml

_PRICING_FILE = Path(__file__).parent / "model_pricing.yaml"
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, float]]:
    # Cached for the lifetime of the process — restart required after editing model_pricing.yaml
    with open(_PRICING_FILE) as f:
        return yaml.safe_load(f)


def _normalize(model: str) -> str:
    """Strip date suffix and provider prefix: 'openai/gpt-5.4-mini-2025-11-01' → 'gpt-5.4-mini'."""
    key = model.split("/")[-1]
    return _DATE_SUFFIX_RE.sub("", key)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Return estimated USD cost. Returns 0.0 for unknown models (no crash)."""
    p = _load().get(_normalize(model))
    if not p:
        return 0.0
    return (
        input_tokens * p.get("input_per_1m", 0) / 1_000_000
        + output_tokens * p.get("output_per_1m", 0) / 1_000_000
        + cached_input_tokens * p.get("cached_input_per_1m", 0) / 1_000_000
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_observability/test_pricing.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/observability/ tests/test_observability/
git commit -m "feat(observability): add pricing module with YAML-backed cost estimation"
```

---

## Task 3: Emit token_usage events from pipeline_helpers

**Files:**
- Modify: `api/services/pipeline_helpers.py`

- [ ] **Step 1: Replace the AIMessageChunk block**

Open `api/services/pipeline_helpers.py`. The current file has this structure at the top:

```python
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
```

Add the import:

```python
from mlops_agents.observability.pricing import estimate_cost
```

Then replace the entire `if isinstance(message_chunk, AIMessageChunk):` block (currently lines 66–84) with:

```python
    if isinstance(message_chunk, AIMessageChunk):
        tool_calls = message_chunk.tool_calls
        if tool_calls:
            tool_name: str = tool_calls[0]["name"]
            _tool_start_times[tool_name] = now_ms
            return PipelineEvent(
                type="tool_call",
                agent=agent,
                timestamp_ms=now_ms,
                data={"tool_name": tool_name, "arguments": tool_calls[0].get("args", {})},
            )
        elif message_chunk.content:
            return PipelineEvent(
                type="agent_reasoning",
                agent=agent,
                timestamp_ms=now_ms,
                data={"content": message_chunk.content},
            )
        elif message_chunk.usage_metadata:
            usage = message_chunk.usage_metadata
            model_name: str = (
                (message_chunk.response_metadata or {}).get("model_name", "")
                or agent
            )
            input_t: int = usage.get("input_tokens", 0)
            output_t: int = usage.get("output_tokens", 0)
            cached_t: int = (usage.get("input_token_details") or {}).get("cache_read", 0)
            return PipelineEvent(
                type="token_usage",
                agent=agent,
                timestamp_ms=now_ms,
                data={
                    "node": agent,
                    "model": model_name,
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": usage.get("total_tokens", input_t + output_t),
                    "cached_input_tokens": cached_t if cached_t else None,
                    "estimated_cost_usd": estimate_cost(model_name, input_t, output_t, cached_t),
                    "source": "langchain_stream_usage_metadata",
                },
            )
        return None
```

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
uv run pytest tests/ -m "not integration" -v --tb=short
```

Expected: Same pass count as before (all non-integration tests pass).

- [ ] **Step 3: Commit**

```bash
git add api/services/pipeline_helpers.py
git commit -m "feat(observability): emit token_usage SSE event from final AIMessageChunk"
```

---

## Task 4: Frontend types + aggregation (TDD)

**Files:**
- Modify: `frontend/types/api.ts`
- Modify: `frontend/lib/events-aggregate.ts`
- Modify: `frontend/__tests__/lib/events-aggregate.test.ts`

- [ ] **Step 1: Add 'token_usage' to PipelineEventType**

Open `frontend/types/api.ts`. In `PipelineEventType`, add `'token_usage'`:

```typescript
export type PipelineEventType =
  | 'run_info'
  | 'routing'
  | 'tool_call'
  | 'tool_result'
  | 'agent_reasoning'
  | 'planner_context'
  | 'hitl_request'
  | 'audit_report'
  | 'training_complete'
  | 'deployment_complete'
  | 'token_usage'
  | 'run_complete'
```

- [ ] **Step 2: Write failing tests for aggregateTokenUsage and tokensByNode**

Open `frontend/__tests__/lib/events-aggregate.test.ts`. Add to the imports at the top:

```typescript
import {
  aggregateToolUsage, aggregateLlmNodeActivity,
  inferToolOwner, getConcreteToolName, buildToolDetailsViewModel,
  aggregateTokenUsage, tokensByNode,
  type LlmActivityRow, type TokenUsageSummary,
} from '@/lib/events-aggregate'
```

Append these describe blocks at the end of the file:

```typescript
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd frontend && npx vitest run __tests__/lib/events-aggregate.test.ts 2>&1 | tail -20
```

Expected: Failures on `aggregateTokenUsage` and `tokensByNode` (not exported yet).

- [ ] **Step 4: Add types and functions to events-aggregate.ts**

Open `frontend/lib/events-aggregate.ts`. Add these exports after the existing `LlmActivityRow` interface (around line 80):

```typescript
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd frontend && npx vitest run __tests__/lib/events-aggregate.test.ts 2>&1 | tail -20
```

Expected: All tests pass (existing + new).

- [ ] **Step 6: Commit**

```bash
git add frontend/types/api.ts frontend/lib/events-aggregate.ts frontend/__tests__/lib/events-aggregate.test.ts
git commit -m "feat(observability): add aggregateTokenUsage and tokensByNode to events-aggregate"
```

---

## Task 5: format.ts utilities (TDD)

**Files:**
- Create: `frontend/lib/format.ts`
- Create: `frontend/__tests__/lib/format.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/__tests__/lib/format.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { formatK, formatCost } from '@/lib/format'

describe('formatK', () => {
  it('returns plain string for values below 1000', () => {
    expect(formatK(0)).toBe('0')
    expect(formatK(999)).toBe('999')
  })

  it('formats exactly 1000 as 1.0k', () => {
    expect(formatK(1000)).toBe('1.0k')
  })

  it('formats thousands with one decimal place', () => {
    expect(formatK(1500)).toBe('1.5k')
    expect(formatK(8200)).toBe('8.2k')
    expect(formatK(12345)).toBe('12.3k')
  })
})

describe('formatCost', () => {
  it('returns $0.00 for zero', () => {
    expect(formatCost(0)).toBe('$0.00')
  })

  it('uses 5 decimal places for values below $0.01', () => {
    expect(formatCost(0.00341)).toBe('$0.00341')
    expect(formatCost(0.00003)).toBe('$0.00003')
    expect(formatCost(0.00999)).toBe('$0.00999')
  })

  it('uses 4 decimal places for values at $0.01 or above', () => {
    expect(formatCost(0.01512)).toBe('$0.0151')
    expect(formatCost(0.01115)).toBe('$0.0112')
    expect(formatCost(1.23456)).toBe('$1.2346')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd frontend && npx vitest run __tests__/lib/format.test.ts 2>&1 | tail -10
```

Expected: `Cannot find module '@/lib/format'`.

- [ ] **Step 3: Create format.ts**

Create `frontend/lib/format.ts`:

```typescript
export function formatK(n: number): string {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n)
}

export function formatCost(usd: number): string {
  if (usd === 0) return '$0.00'
  if (usd < 0.01) return '$' + usd.toFixed(5)
  return '$' + usd.toFixed(4)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npx vitest run __tests__/lib/format.test.ts 2>&1 | tail -10
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/format.ts frontend/__tests__/lib/format.test.ts
git commit -m "feat(observability): add formatK and formatCost display helpers"
```

---

## Task 6: Update LlmActivityCard

**Files:**
- Modify: `frontend/components/observability/LlmActivityCard.tsx`

- [ ] **Step 1: Rewrite the component**

Replace the full contents of `frontend/components/observability/LlmActivityCard.tsx` with:

```tsx
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
```

- [ ] **Step 2: Run frontend tests to verify no regressions**

```bash
cd frontend && npx vitest run 2>&1 | tail -15
```

Expected: Same pass count as before (existing tests unaffected).

- [ ] **Step 3: Commit**

```bash
git add frontend/components/observability/LlmActivityCard.tsx
git commit -m "feat(observability): fill token column in LlmActivityCard"
```

---

## Task 7: TokenCostCard + wire into observability page

**Files:**
- Create: `frontend/components/observability/TokenCostCard.tsx`
- Modify: `frontend/app/observability/page.tsx`

- [ ] **Step 1: Create TokenCostCard**

Create `frontend/components/observability/TokenCostCard.tsx`:

```tsx
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
```

- [ ] **Step 2: Add TokenCostCard to the observability page**

Replace the full contents of `frontend/app/observability/page.tsx` with:

```tsx
'use client'
import { PipelineHealthCard } from '@/components/observability/PipelineHealthCard'
import { LlmActivityCard } from '@/components/observability/LlmActivityCard'
import { ToolUsageCard } from '@/components/observability/ToolUsageCard'
import { TokenCostCard } from '@/components/observability/TokenCostCard'

export default function ObservabilityPage() {
  return (
    <div className="space-y-3 p-3">
      <PipelineHealthCard />
      <LlmActivityCard />
      <TokenCostCard />
      <ToolUsageCard />
    </div>
  )
}
```

- [ ] **Step 3: Run frontend tests to verify no regressions**

```bash
cd frontend && npx vitest run 2>&1 | tail -15
```

Expected: All tests pass.

- [ ] **Step 4: Run full backend test suite**

```bash
uv run pytest -m "not integration" --tb=short 2>&1 | tail -10
```

Expected: All non-integration tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/observability/TokenCostCard.tsx frontend/app/observability/page.tsx
git commit -m "feat(observability): add TokenCostCard with per-node cost breakdown"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| `pyyaml>=6.0` + `types-PyYAML` in deps | Task 1 |
| `model_pricing.yaml` with gpt-5.4-mini + gpt-5.4-nano prices | Task 2 |
| `pricing.py` with `estimate_cost`, `_normalize`, `lru_cache` | Task 2 |
| `cache_read` key (not `cached_tokens`) for cached token extraction | Task 3 |
| `elif` branch ordering in `parse_stream_event` | Task 3 |
| `token_usage` added to `PipelineEventType` | Task 4 |
| `TokenUsageEventData`, `TokenUsageRow`, `TokenUsageSummary` interfaces | Task 4 |
| `aggregateTokenUsage` groups by node+model, accumulates calls | Task 4 |
| `tokensByNode` collapses models per node | Task 4 |
| `formatK` and `formatCost` in `frontend/lib/format.ts` | Task 5 |
| `LlmActivityCard` token column with `8.2k in · 1.1k out` format | Task 6 |
| `LlmActivityCard` "none recorded" note removed | Task 6 |
| `TokenCostCard` three-part layout (header+total, table, footer) | Task 7 |
| Total shows `—` until first event, then `$X.XXXX` | Task 7 |
| Footer disclaimer with prices + "Update model_pricing.yaml" note | Task 7 |
| Unit tests for `pricing.py` | Task 2 |
| Unit tests for `aggregateTokenUsage` + `tokensByNode` | Task 4 |
| Unit tests for `formatK` + `formatCost` | Task 5 |

All 18 spec requirements covered. No gaps found.
