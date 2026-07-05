# Token Cost Observability — Internal Dashboard Design

## Goal

Add token usage and estimated cost tracking to the existing observability dashboard.
No external services required. All data flows through the existing SSE event pipeline.
Langfuse integration is deferred as a future optional extension.

## Context

The three LLM-using nodes are:

| Node | Category | Model (env var) |
|---|---|---|
| `data_validator` | agent | `OPENAI_MODEL_DATA_VALIDATOR` = `gpt-5.4-mini` |
| `planner` | agent | `OPENAI_MODEL_PLANNER` = `gpt-5.4-mini` |
| `report_writer` | llm_node | `OPENAI_MODEL_REPORT_WRITER` = `gpt-5.4-nano` |

The final `AIMessageChunk` in every LangChain streaming call carries `usage_metadata`
(`input_tokens`, `output_tokens`, `total_tokens`) and `response_metadata` (`model_name`).
This data is already in the stream but currently discarded by `parse_stream_event`.

---

## Architecture

```
AIMessageChunk (final chunk, usage_metadata set)
    ↓
parse_stream_event()          [api/services/pipeline_helpers.py]
    ↓ detect usage_metadata
estimate_cost()               [src/mlops_agents/observability/pricing.py]
    ↓
token_usage SSE event         [emitted into existing event queue]
    ↓
Frontend run-store            [events[] array — no schema change needed]
    ↓
aggregateTokenUsage()         [frontend/lib/events-aggregate.ts]
    ↓
TokenCostCard + LlmActivityCard (token column filled)
```

---

## Backend

### New module: `src/mlops_agents/observability/`

Three files:

**`__init__.py`** — empty.

**`model_pricing.yaml`**:

```yaml
# Estimated costs in USD per 1M tokens
# Source: https://platform.openai.com/docs/pricing (checked 2026-05-31)
# Update this file when model pricing changes or new models are added.

gpt-5.4-mini:
  input_per_1m: 0.75
  output_per_1m: 4.50
  cached_input_per_1m: 0.0   # not published; update when available

gpt-5.4-nano:
  input_per_1m: 0.20
  output_per_1m: 1.25
  cached_input_per_1m: 0.0   # not published; update when available
```

**`pricing.py`**:

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
    """Strip date suffix and provider prefix so 'openai/gpt-5.4-mini-2025-11-01' → 'gpt-5.4-mini'."""
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

### Modified: `api/services/pipeline_helpers.py`

Replace the `isinstance(message_chunk, AIMessageChunk)` block with the full three-branch
`if / elif / elif` structure. The branches are mutually exclusive: OpenAI's final streaming
chunk carries usage_metadata with empty content; earlier chunks carry content but no usage_metadata.

```python
from mlops_agents.observability.pricing import estimate_cost

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
        model: str = (
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
                "model": model,
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": usage.get("total_tokens", input_t + output_t),
                "cached_input_tokens": cached_t if cached_t else None,
                "estimated_cost_usd": estimate_cost(model, input_t, output_t, cached_t),
                "source": "langchain_stream_usage_metadata",
            },
        )
```

No changes needed in `pipeline.py` — `token_usage` events fall through the existing
`else` branch (flush reasoning, then enqueue the event).

---

## Frontend

### `frontend/types/api.ts`

Add `'token_usage'` to `PipelineEventType`.

### `frontend/lib/events-aggregate.ts`

Add types and function:

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
  rows: TokenUsageRow[]            // grouped by node + model
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
```

Helper for `LlmActivityCard` (sums tokens per node, collapsing multiple models):

```typescript
export function tokensByNode(summary: TokenUsageSummary): Map<string, { inputTokens: number; outputTokens: number }> {
  const m = new Map<string, { inputTokens: number; outputTokens: number }>()
  for (const r of summary.rows) {
    const prev = m.get(r.node)
    if (prev) { prev.inputTokens += r.inputTokens; prev.outputTokens += r.outputTokens }
    else m.set(r.node, { inputTokens: r.inputTokens, outputTokens: r.outputTokens })
  }
  return m
}
```

### `frontend/components/observability/LlmActivityCard.tsx`

- Import `aggregateTokenUsage`, `tokensByNode`
- Build token map from events
- Replace `<td className="text-zinc-400">—</td>` with:
  - If token data exists for node: `8.2k in · 1.1k out` (formatted with `formatK`)
  - If absent: `—`
- Remove the "Token counts shown when available — none recorded" note entirely

Token format helper `formatK` and cost helper `formatCost` both live in `frontend/lib/format.ts` (see below).

### NEW `frontend/components/observability/TokenCostCard.tsx`

Three-part layout:

**Header row:**
```
Token Cost (current run)          Total: $0.01512
```
Total is right-aligned, bold. Shows `—` until the first `token_usage` event arrives; switches to `$X.XXXXX` once data is present.

**Table** (shown only when rows exist):
```
Node             Model           Calls    In       Out     Total      Cost
data_validator   gpt-5.4-mini      2    3,512     648     4,160   $0.00341
planner          gpt-5.4-mini      1    8,200   1,100     9,300   $0.01115
report_writer    gpt-5.4-nano      1      980     290     1,270   $0.00056
```
Numbers formatted with `toLocaleString()`. Cost formatted with `formatCost(usd)`:
- `usd === 0` → `$0.00`
- `usd < 0.01` → `$X.XXXXX` (5 decimal places — preserves precision for micro-costs)
- `usd >= 0.01` → `$X.XXXX` (4 decimal places)

Both helpers live in `frontend/lib/format.ts`:

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

When no rows: `<p>No LLM activity yet.</p>`

**Footer:**
```
Estimated cost based on local pricing table.
Prices: gpt-5.4-mini $0.75/$4.50 · gpt-5.4-nano $0.20/$1.25 per 1M input/output tokens.
Update model_pricing.yaml if model prices change.
```
Font size 11px, zinc-400.

### `frontend/app/observability/page.tsx`

Add `<TokenCostCard />` after `<LlmActivityCard />`.

---

## Files changed

| File | Change |
|---|---|
| `pyproject.toml` | MODIFY — add `pyyaml>=6.0` to direct dependencies, `types-PyYAML` to dev group |
| `src/mlops_agents/observability/__init__.py` | NEW (empty) |
| `src/mlops_agents/observability/pricing.py` | NEW |
| `src/mlops_agents/observability/model_pricing.yaml` | NEW |
| `api/services/pipeline_helpers.py` | MODIFY — detect usage_metadata, emit token_usage |
| `frontend/types/api.ts` | MODIFY — add `'token_usage'` |
| `frontend/lib/events-aggregate.ts` | MODIFY — add aggregateTokenUsage, tokensByNode, TokenUsageRow, TokenUsageSummary, TokenUsageEventData |
| `frontend/components/observability/LlmActivityCard.tsx` | MODIFY — fill token column, remove placeholder note |
| `frontend/components/observability/TokenCostCard.tsx` | NEW |
| `frontend/app/observability/page.tsx` | MODIFY — add TokenCostCard |
| `tests/test_observability/test_pricing.py` | NEW — unit tests for pricing.py |
| `frontend/__tests__/lib/events-aggregate.test.ts` | MODIFY — add aggregateTokenUsage tests |
| `frontend/lib/format.ts` | NEW — `formatK`, `formatCost` helpers |

Total: 13 files (6 new, 7 modified).

---

## Future extension: Langfuse (not in scope)

When adding Langfuse later:

1. Add `langfuse` + `langfuse-langchain` to `pyproject.toml`
2. Add `OBSERVABILITY_BACKEND=none | langfuse` to settings
3. Register `CallbackHandler` on each `ChatOpenAI` in `utils/llm.py` when backend is `langfuse`
4. Add Langfuse + Postgres + Redis to `docker-compose.yml`

The internal `token_usage` events remain the source of truth for the app UI regardless of
whether Langfuse is enabled.
