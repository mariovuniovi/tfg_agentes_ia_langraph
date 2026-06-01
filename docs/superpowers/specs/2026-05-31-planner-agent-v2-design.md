# Planner Agent v2 — Design Spec

**Date:** 2026-05-31
**Branch:** `feature/container` (continues current branch)
**Status:** Approved for planning

## Goal

Turn the Model Planner from a structured-output LLM into a real tool-using agent, and enrich its output schema so the UI can honestly explain *why* each candidate was selected. The deterministic helpers stay deterministic; the agent gains agency over *what to fetch and when*. After this refactor the system has **two true agents** (`data_validator`, `planner`) instead of one.

## Non-goals

- Embeddings / vector similarity (deterministic bucket-based retrieval stays as-is)
- The planner gaining the ability to train, evaluate, register, or deploy models
- Removing the deterministic executor / evaluator / deployer guardrails
- Free-text planner output (the `PlannerOutput` Pydantic contract is preserved and extended)
- A separate "explainer LLM" step (the agent itself produces the rationales)

## Architecture decisions (locked in brainstorming + grilling)

| # | Decision | Choice |
|---|---|---|
| A1 | Validation context source | **Hybrid** — deterministic for invariants (registry, exhaustiveness), agent-observed for citation honesty (experience + rule refs) |
| A2 | Agent shape | Single `create_agent(..., response_format=PlannerOutput)` ReAct agent (mirrors `data_validator`) |
| A3 | `planner_context` SSE payload | Carries deterministic universe + agent tool trace + soft conflicts |
| A4 | Scope | One spec: tools + output schema enrichment + Planner tab redesign |
| A5 | Per-candidate schema | `CandidateSpec` gains `priority`+`reason`+`evidence_refs`+`risks`. `RejectedModelSpec` gains `evidence_refs` + optional `reconsider_if` |
| A6 | Similarity tier source | Deterministic, computed at retrieval. Thresholds: high ≥ 0.7, medium 0.4–0.7, low < 0.4 |
| A7 | Conflict detection | Deterministic flag + LLM `evidence_conflicts[]` carries resolution. Hard vs soft tiers; only hard blocks validation |
| A8 | Tool I/O | Closure-bound via `build_planner_tools(profile, task_meta, problem_type, trace)`. Tools take only varying args |
| A9 | Decision basis | LLM produces `DecisionBasis(primary_evidence, secondary_evidence, final_strategy)` with `EvidenceReference` lists |

### Grilling-derived decisions (Q1–Q7)

| # | Decision |
|---|---|
| Q1 | `supports_exogenous: bool` + `supports_missing: bool` become first-class `ModelSpec` fields. Backfill every entry in `src/mlops_agents/models/registry.yaml`. Exposed via `summary_dict()` + `details_dict()` |
| Q2 | DatasetProfile gains numeric `target_mean`/`target_std`/`target_min`/`target_max` for non-classification problems. `compare_target_scales(a, b)` returns `None` if either side lacks numeric stats (graceful for legacy `ExperienceRecord`s) |
| Q3 | `settings.planner_max_retrieved: int = 20`. `validation_ctx.similar_experiences` pre-fetches at this depth so conflict detection always sees what the agent cites. Tool default stays `top_k=5` |
| Q4 | Sequencing: finish prior refactor (Slice 5.6 smoke + final code review + tag) BEFORE starting Planner v2 implementation. Planner v2 modifies same files (`PipelineStepper`, `RunHeader`, `ResultsDashboard`, `app/pipeline/page.tsx`) — clean base avoids blame-attribution confusion |
| Q5 | Retry on validation failure = cold restart. Fresh trace, fresh tools, error feedback prepended as `HumanMessage`. Max 2 attempts (existing pattern preserved) |
| Q6 | `settings.planner_max_inspect_calls: int = 3`. Tool gate enforces per-tool cap on `inspect_model_details` (returns recoverable error message after 3rd call). Total `planner_max_tool_calls=6` stays as global ceiling |
| Q7 | New module `src/mlops_agents/agents/taxonomy.py` exports `NODE_CATEGORIES: dict[str, list[str]]` as single source of truth. Both `api/services/pipeline.py` (for `run_info.node_categories`) and any other consumer import from here |

## High-level architecture

```
planner_node
│
├── validation_ctx = build_planner_validation_context(profile, task_meta, problem_type)
│   (deterministic ground truth, built ONCE before retry loop)
│
├── for attempt in range(2):       ← retry-once pattern
│     trace = ToolTrace()
│     tools = build_planner_tools(profile, task_meta, problem_type, trace)
│     agent = create_agent(model=..., tools=tools, response_format=PlannerOutput, ...)
│     result = agent.invoke(messages, config={"recursion_limit": settings.planner_max_iterations})
│     output = result["structured_response"]
│
│     try:
│         _check_plan_integrity(output, trace, validation_ctx)
│         _check_plan_exhaustiveness(output.plan, validation_ctx.available_model_keys)
│         _check_evidence_references_hybrid(output, validation_ctx, trace)
│         _check_conflict_resolution_present_if_flagged(output, validation_ctx, trace)
│         break
│     except PlannerValidationError as exc:
│         last_error = str(exc)
│         if attempt == 1: raise
│
├── sort candidates by priority
└── return Command(goto="workflow_controller", update={...trace, validation_context, training_plan...})
```

## File map

```
src/mlops_agents/planning/                ← NEW module
  __init__.py
  tools.py            # build_planner_tools(dataset_profile, task_metadata, problem_type, trace)
                      # 4 closure-bound @tool functions
  agent.py            # build_planner_agent(tools) — wraps create_agent
  node.py             # planner_node() — entry, validation orchestration
  validation.py       # _check_plan_integrity, _check_plan_exhaustiveness,
                      # _check_evidence_references_hybrid,
                      # _check_conflict_resolution_present_if_flagged,
                      # _detect_conflicts
  context.py          # build_planner_validation_context(profile, task_meta, problem_type)
                      # returns PlannerValidationContext
  trace.py            # ToolTrace pydantic model + tool wrapper utilities
  prompts.py          # message builders + planner input formatter

src/mlops_agents/contracts/planner.py     ← MODIFIED
  - CandidateSpec: + priority, reason, evidence_refs, risks
  - RejectedModelSpec: + evidence_refs, reconsider_if
  - PlannerOutput: + decision_basis, + evidence_conflicts (list)
  - DecisionBasis (NEW): primary_evidence, secondary_evidence, final_strategy
  - EvidenceConflict (NEW): summary, affected_models, conflicting_evidence_refs, resolution
  - ExperienceSummary: + relevance_tier, matched_buckets, mismatched_buckets, target_scale_note
  - PlannerValidationContext (NEW): available_model_keys, available_model_specs,
                                    similar_experiences, matched_rules, rules_by_id,
                                    task_metadata, problem_type

src/mlops_agents/experience/retrieval.py  ← MODIFIED
  - extend RetrievalView with matched_buckets, mismatched_buckets, target_scale_note
  - helper: derive_relevance_tier(similarity_score) → 'high'|'medium'|'low'
  - helper: compare_target_scales(profile_a, profile_b) → str|None

src/mlops_agents/agents/planner.py        ← THIN SHIM
  - re-export `from mlops_agents.planning.node import planner_node`
  - flagged for deletion in a follow-up cleanup

src/mlops_agents/agents/registry.py       ← MODIFIED
  - planner factory → mlops_agents.planning.agent.build_planner_agent

src/mlops_agents/graphs/mlops_graph.py    ← MODIFIED
  - import planner_node from mlops_agents.planning.node

src/mlops_agents/config/settings.py       ← MODIFIED
  - planner_max_iterations: int = 10        # LangGraph recursion_limit (enforced)
  - planner_max_tool_calls: int = 6         # total tool budget (enforced via gate)
  - planner_max_inspect_calls: int = 3      # per-tool cap on inspect_model_details (enforced via gate)
  - planner_max_retrieved: int = 20         # depth of validation_ctx pre-fetch + clamp for retrieve_similar_experiences
  - planner_timeout_seconds: int = 60       # RESERVED for future wall-clock enforcement; NOT enforced in v2

src/mlops_agents/prompts/planner.yaml     ← REWRITTEN
  - new system prompt explaining tools, mandatory retrieval, output contract,
    decision_basis requirement, conflict resolution conditional requirement

src/mlops_agents/state/agent_state.py     ← MODIFIED
  - + planner_tool_trace: dict
  - + planner_validation_context: dict (audit subset)

src/mlops_agents/models/loader.py         ← MODIFIED (small)
  - ModelSpec: + supports_exogenous: bool = False
              + supports_missing: bool = False
  - ModelSpec.summary_dict()  (headline fields for list_available_models)
  - ModelSpec.details_dict()  (full info for inspect_model_details)

src/mlops_agents/models/registry.yaml     ← MODIFIED
  - backfill supports_exogenous (and supports_missing where missing) for every model entry

src/mlops_agents/training/profiler.py     ← MODIFIED
  - DatasetProfile: + target_mean, target_std, target_min, target_max (numeric only, None for classification)
  - profiler computes them when target is numeric; leaves None otherwise

src/mlops_agents/experience/schema.py     ← MODIFIED
  - ExperienceRecord: + target_mean, target_std, target_min, target_max (all Optional, default None for legacy compat)

src/mlops_agents/agents/taxonomy.py       ← NEW
  - NODE_CATEGORIES: dict[str, list[str]]
  - is_agent(name), is_llm_node(name), is_deterministic(name) helpers
```

```
api/services/pipeline.py                  ← MODIFIED
  - _planner_output_record: include decision_basis, evidence_conflicts,
                            soft_conflicts, cited_experience_ids, cited_rule_ids,
                            per-candidate full objects
  - run_info event payload: add node_categories
                            {agents: [...], llm_nodes: [...], deterministic: [...]}
```

```
frontend/                                  ← UI redesign
  types/api.ts                            ← MODIFIED
    - extend CandidateFull, RejectedModelFull (rename or extend existing CandidateSpec / RejectedSpec)
    - + DecisionBasis, EvidenceConflict, SoftConflict
    - extend ExperienceSummary
    - extend PlannerContextData

  components/pipeline/PlannerPanel.tsx     ← NEW (extracted from ResultsDashboard.tsx)
    composes:
      PlannerSummaryHeader
      DecisionBasisCard
      ConflictPanel        (renders both hard + soft, hard with amber border, soft neutral)
      CandidateRationaleList → CandidateCard
      RejectedModelsList → RejectedModelCard
      EvidenceQualityCard
      SimilarPastRunsList → ExperienceCard
      MatchedRulesList         (mostly today's)
      PlannerWarningsList      (today's)
      <details>"View full planning analysis"  (audit drawer)

  components/pipeline/planner/             ← NEW directory for sub-components
    PlannerSummaryHeader.tsx
    DecisionBasisCard.tsx
    ConflictPanel.tsx
    CandidateCard.tsx
    RejectedModelCard.tsx
    EvidenceQualityCard.tsx
    ExperienceCard.tsx

  components/pipeline/PipelineStepper.tsx  ← MODIFIED
    - STAGES: model_planning.type 'llm' → 'agent'

  components/pipeline/RunHeader.tsx        ← MODIFIED
    - props: agents[], llmNodes[], deterministic[]
    - render single dense flex-wrap row with three category labels

  components/pipeline/ResultsDashboard.tsx ← MODIFIED
    - inline PlannerPanel removed; imports new component

  app/pipeline/page.tsx                    ← MODIFIED
    - derive node_categories from run_info event; pass to RunHeader
    - fallback to legacy models list for old runs

  __tests__/components/pipeline/PlannerPanel.test.tsx   ← NEW
  __tests__/components/pipeline/planner/*.test.tsx       ← NEW per sub-component
```

```
tests/                                     ← BACKEND TESTS
  test_planning/
    __init__.py
    test_tools.py            # each tool: bound profile/task_meta/problem_type;
                             # trace recorded; tool_call_count enforcement
    test_agent.py            # agent with mocked tool returns produces valid PlannerOutput
    test_validation.py       # integrity, exhaustiveness, hybrid refs, conflict resolution
                             # mandatory tools; priority uniqueness; registry self-citation;
                             # known_future override rule
    test_context.py          # validation context construction is deterministic + idempotent
    test_trace.py            # tool wrappers record observations; dedup; count
    test_conflict_detection.py  # hard vs soft branches; rule prefer/avoid conflicts
    test_node.py             # full node flow with mocked LLM
  test_contracts/
    test_planner_schemas.py  # priority uniqueness, evidence_ref validation per source type
```

---

## Section 1 — Tool implementations

```python
# src/mlops_agents/planning/tools.py
from typing import Any
from langchain_core.tools import tool
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_models_for, get_model
from mlops_agents.config.settings import settings
from mlops_agents.planning.trace import ToolTrace

_MAX_CALLS_ERR = {"error": "max_tool_calls exceeded — terminate and produce final PlannerOutput"}


def build_planner_tools(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    trace: ToolTrace,
) -> list:
    """Build closure-bound planner tools that record observations to the shared trace."""

    def _gate(tool_name: str | None = None) -> bool:
        """Returns False when the call should be rejected. Enforces both the global
        max_tool_calls ceiling and the per-tool inspect cap."""
        if trace.tool_call_count >= settings.planner_max_tool_calls:
            return False
        if tool_name == "inspect_model_details":
            inspected_count = len(trace.inspected_model_keys)
            if inspected_count >= settings.planner_max_inspect_calls:
                return False
        trace.tool_call_count += 1
        return True

    _MAX_INSPECT_ERR = {
        "error": "max inspect_model_details calls reached — produce final PlannerOutput "
                 "using available info or call other tools"
    }

    def _dedup(field: list[str], new_items: set[str]) -> list[str]:
        return sorted(set(field) | new_items)

    @tool
    def list_available_models() -> list[dict] | dict:
        """List all models in the registry for the current problem type. Returns one entry per
        model with headline fields (model_key, family, complexity_rank, supports_exogenous,
        supports_missing, use_when, avoid_when). Call this once at the start of planning.
        Models not in this list cannot be recommended."""
        if not _gate(): return _MAX_CALLS_ERR
        specs = get_models_for(problem_type)
        out = [s.summary_dict() for s in specs]
        trace.called_tools = _dedup(trace.called_tools, {"list_available_models"})
        trace.listed_model_keys = _dedup(trace.listed_model_keys, {s["model_key"] for s in out})
        trace.raw_observations.append({"tool": "list_available_models", "result_count": len(out)})
        return out

    @tool
    def retrieve_similar_experiences(top_k: int = 5) -> list[dict] | dict:
        """Retrieve the top-k most similar past training experiences for the current dataset.
        Similarity is deterministic (bucket-based, no embeddings). Each result includes
        experience_id, similarity_score, relevance_tier, matched_buckets, mismatched_buckets,
        target_scale_note, best_model, validation_score, metric_name, dataset_summary.
        Use these to inform candidate selection. Call this once unless you need a wider net.
        top_k is clamped to [1, settings.planner_max_retrieved] so it never exceeds what the
        deterministic validation context pre-fetched (otherwise the agent could cite
        experiences outside the conflict-detection window)."""
        if not _gate(): return _MAX_CALLS_ERR
        top_k = max(1, min(top_k, settings.planner_max_retrieved))
        pool = ExperiencePool(settings.experience_db_path)
        views = pool.find_similar(dataset_profile, problem_type, top_k)
        out = [_view_to_tool_dict(v) for v in views]
        trace.called_tools = _dedup(trace.called_tools, {"retrieve_similar_experiences"})
        trace.retrieved_experience_ids = _dedup(
            trace.retrieved_experience_ids, {o["experience_id"] for o in out}
        )
        trace.raw_observations.append({
            "tool": "retrieve_similar_experiences", "top_k": top_k, "returned": len(out),
        })
        return out

    @tool
    def retrieve_ml_knowledge() -> list[dict] | dict:
        """Retrieve static ML rules that match the current dataset profile and task metadata.
        Each rule returns rule_id, prefer (preferred model keys), avoid_or_deprioritize,
        recommend (free-text recommendation), summary. Use these as static expert knowledge.
        Call this once."""
        if not _gate(): return _MAX_CALLS_ERR
        rule_input = {**dataset_profile, **task_metadata, "problem_type": problem_type}
        # NOTE: if task_metadata keys collide with profile keys, task_metadata wins.
        # Future hardening: namespace into nested dict + update match_rules signature.
        matched = match_rules(rule_input)
        out = [{
            "rule_id": r.rule_id, "prefer": r.prefer,
            "avoid_or_deprioritize": r.avoid_or_deprioritize,
            "recommend": r.recommend, "summary": r.reason,
        } for r in matched]
        trace.called_tools = _dedup(trace.called_tools, {"retrieve_ml_knowledge"})
        trace.retrieved_rule_ids = _dedup(trace.retrieved_rule_ids, {r["rule_id"] for r in out})
        trace.raw_observations.append({"tool": "retrieve_ml_knowledge", "returned": len(out)})
        return out

    @tool
    def inspect_model_details(model_key: str) -> dict:
        """Get full registry metadata for one model: family, complexity_rank,
        supports_exogenous, supports_missing, search_space hint, default_params,
        use_when, avoid_when, notes. Use this sparingly — only when list_available_models
        doesn't give you enough info to decide. Hard cap of 3 inspects per planner run.
        Returns {"error": ...} if model_key unknown."""
        # Per-tool cap check FIRST (before incrementing) so cap-hit doesn't burn budget
        inspected_count = len(trace.inspected_model_keys)
        if inspected_count >= settings.planner_max_inspect_calls:
            return _MAX_INSPECT_ERR
        if not _gate("inspect_model_details"): return _MAX_CALLS_ERR
        try:
            spec = get_model(model_key)
        except KeyError:
            trace.raw_observations.append({"tool": "inspect_model_details", "model_key": model_key, "error": "unknown"})
            return {"error": f"unknown model_key: {model_key!r}"}
        out = spec.details_dict()
        trace.called_tools = _dedup(trace.called_tools, {"inspect_model_details"})
        trace.inspected_model_keys = _dedup(trace.inspected_model_keys, {model_key})
        trace.raw_observations.append({"tool": "inspect_model_details", "model_key": model_key})
        return out

    return [list_available_models, retrieve_similar_experiences, retrieve_ml_knowledge, inspect_model_details]
```

```python
# src/mlops_agents/planning/trace.py
from pydantic import BaseModel, Field

class ToolTrace(BaseModel):
    called_tools: list[str] = Field(default_factory=list)
    listed_model_keys: list[str] = Field(default_factory=list)
    retrieved_experience_ids: list[str] = Field(default_factory=list)
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    inspected_model_keys: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    raw_observations: list[dict] = Field(default_factory=list)
```

---

## Section 2 — Contract changes

```python
# src/mlops_agents/contracts/planner.py
class EvidenceReference(BaseModel):
    source: Literal["dataset_profile", "task_metadata", "registry", "experience", "rule"]
    source_id: str | None = None       # required non-empty for registry/experience/rule
    relevance_note: str | None = None  # short why-this-matters explanation

class CandidateSpec(BaseModel):
    model_key: str
    priority: int = Field(ge=1)                       # ≥1, unique within plan
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)  # must include registry self-citation
    risks: list[str] = Field(default_factory=list)

class RejectedModelSpec(BaseModel):
    model_key: str
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)  # must include registry self-citation
    reconsider_if: str | None = None

class DecisionBasis(BaseModel):
    primary_evidence: list[EvidenceReference] = Field(min_length=1)
    secondary_evidence: list[EvidenceReference] = Field(default_factory=list)
    final_strategy: str = Field(min_length=1)

class EvidenceConflict(BaseModel):
    summary: str = Field(min_length=1)
    affected_models: list[str] = Field(min_length=1)
    conflicting_evidence_refs: list[EvidenceReference] = Field(min_length=1)
    resolution: str = Field(min_length=1)

class ExperienceSummary(BaseModel):
    experience_id: str
    similarity_score: float
    relevance_tier: Literal["high", "medium", "low"]   # derived from similarity_score
    matched_buckets: list[str] = Field(default_factory=list)
    mismatched_buckets: list[str] = Field(default_factory=list)
    target_scale_note: str | None = None
    dataset_summary: str
    models_trained: list[str]
    best_model: str
    validation_score: float | None
    metric_name: str
    candidate_results: list[CandidateResultCompact]   # existing

class PlannerOutput(BaseModel):
    planning_analysis: str
    decision_basis: DecisionBasis                       # NEW required
    evidence_used: list[EvidenceReference] = Field(default_factory=list)  # legacy/general refs
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)  # required non-empty IF hard conflict flagged
    risks_or_warnings: list[str] = Field(default_factory=list)
    plan: TrainingPlan

class PlannerValidationContext(BaseModel):
    """Deterministic ground-truth context — independent of agent behavior."""
    problem_type: str
    task_metadata: dict
    available_model_keys: list[str]
    available_model_specs: list[ModelSpec]            # full ModelSpec objects for richer checks
    similar_experiences: list[ExperienceSummary]     # deterministic retrieval at settings.planner_max_retrieved (=20 by default) so conflict detection always sees what the agent could cite
    matched_rules: list[dict]
    rules_by_id: dict[str, dict]                      # for fast lookup in conflict detection
```

```yaml
# src/mlops_agents/prompts/planner.yaml — rewritten
template: |
  You are the Model Planner Agent.

  Your job: produce a validated TrainingPlan.

  ## You have four tools
  - list_available_models() — start here. Lists the registry universe for this problem_type.
  - retrieve_similar_experiences(top_k=5) — past runs by deterministic similarity to this dataset.
  - retrieve_ml_knowledge() — static ML rules matching this dataset profile.
  - inspect_model_details(model_key) — deep info on one model. Use sparingly; only when needed.

  ## You must
  - Call list_available_models() at least once.
  - Call retrieve_similar_experiences() at least once.
  - Call retrieve_ml_knowledge() at least once.
  - Recommend only models returned by list_available_models() for this problem_type.
  - Classify EVERY available model as either a candidate or a rejected model.
  - For every candidate: provide reason, evidence_refs, risks. evidence_refs MUST include
    at least one registry reference where source_id == the candidate's model_key.
  - For every rejected model: provide reason and evidence_refs (same registry rule). Optionally
    add reconsider_if when the rejection is conditional on circumstances that could change.
  - If a model is selected mainly because of a rule or registry prior (no supporting experience),
    state that explicitly in reason or risks. Do not imply empirical support that doesn't exist.
  - Cite only experiences and rules you actually retrieved this run.
  - Provide decision_basis with primary_evidence and secondary_evidence as lists of
    EvidenceReference objects (not free-text labels), plus a final_strategy narrative.
  - If you observe a conflict between retrieved experiences, retrieved rules, registry guidance,
    or your selected/rejected models, fill evidence_conflicts with at least one entry.
    A deterministic post-check will also flag conflicts; if it does and you did not provide
    a resolution, the planner run will be rejected and retried.

  ## You must NOT
  - Recommend a model_key not in the registry.
  - Cite an experience or rule you didn't retrieve via tools.
  - Call training, evaluation, MLflow, or deployment tools — you don't have any.
  - Exceed ~5 tool calls. Default sequence: list_available_models, retrieve_similar_experiences,
    retrieve_ml_knowledge, optional inspect_model_details (1–2), then produce output.

  ## Forecasting-specific guidance
  - If history_length is very_short or short: prefer single_split validation; include
    statistical baselines (naive, seasonal_naive, ets, auto_arima where registered);
    avoid high-complexity supervised models unless similar experiences strongly support them.
  - If expected_drift is high and history is sufficient: prefer rolling_window.
  - For unknown_future exogenous columns: choose extension strategies from
    {naive_carry, ets, auto_arima, drop}.
  - known_future variables must not appear in per-column unknown-future overrides at all.

  ## Output
  Produce a PlannerOutput. The schema is enforced — fill every required field.
  Conservative beats optimistic when evidence is weak or conflicting.
```

---

## Section 3 — Validation chain

Full validation surface — each function in `planning/validation.py`. See Section 4 of the design conversation for full code; key invariants summarized:

**`_check_plan_integrity(output, trace, ctx)`**:
- Required tools called: `{list_available_models, retrieve_similar_experiences, retrieve_ml_knowledge} ⊆ trace.called_tools`
- `trace.tool_call_count <= settings.planner_max_tool_calls`
- Candidate priorities ≥ 1, unique
- No overlap between candidates and rejected
- Forecasting plans: `validation_strategy ∈ {single_split, rolling_window, expanding_window}`
- Forecasting plans: exog strategies ∈ `{naive_carry, ets, auto_arima, drop}`
- Forecasting plans: `known_future` columns cannot appear in per-column unknown-future overrides at all (no "drop" loophole)
- Per-candidate registry self-citation present
- Per-rejected registry self-citation present

**`_check_plan_exhaustiveness(plan, ctx.available_model_keys)`**:
- `set(available_model_keys) ⊆ {c.model_key for c in candidates} ∪ {r.model_key for r in rejected}`

**`_check_evidence_references_hybrid(output, ctx, trace)`** — collects all refs (from `evidence_used`, every candidate's `evidence_refs`, every rejected's `evidence_refs`, `decision_basis.primary_evidence`, `decision_basis.secondary_evidence`, every `evidence_conflicts[*].conflicting_evidence_refs`):
- `source ∈ {dataset_profile, task_metadata}` → `source_id` must be None
- `source == registry` → `source_id` non-empty AND in `ctx.available_model_keys`
- `source == experience` → `source_id` non-empty AND in `trace.retrieved_experience_ids`
- `source == rule` → `source_id` non-empty AND in `trace.retrieved_rule_ids`

**`_check_conflict_resolution_present_if_flagged(output, ctx, trace)`**:
- `flagged = _detect_conflicts(ctx, trace, output.plan, output)` returns list of hard conflicts
- If `flagged` non-empty: `output.evidence_conflicts` must be non-empty
- Each `evidence_conflicts[*].resolution` must be non-empty

**`_detect_conflicts(ctx, trace, plan, output)`** returns hard list; soft list is computed separately and attached to the SSE event but doesn't gate validation:

```python
def _detect_conflicts(ctx, trace, plan, output):
    hard = []
    candidate_keys = {c.model_key for c in plan.candidates}
    rejected_keys = {r.model_key for r in plan.models_not_recommended}

    # Hard: cited-experience winner not in candidates
    cited_experience_ids = {ref.source_id for ref in _collect_all_refs(output) if ref.source == "experience"}
    cited_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in cited_experience_ids and e.best_model
    }
    omitted_cited = cited_winners - candidate_keys
    if omitted_cited:
        hard.append({"type": "cited_experience_winner_not_selected",
                     "models": sorted(omitted_cited), "severity": "hard"})

    # Hard: cited rule's prefer/avoid contradicted
    cited_rule_ids = {ref.source_id for ref in _collect_all_refs(output) if ref.source == "rule"}
    for rid in cited_rule_ids:
        rule = ctx.rules_by_id.get(rid)
        if not rule: continue
        avoid_in_cands = set(rule.get("avoid_or_deprioritize", [])) & candidate_keys
        prefer_in_rej = set(rule.get("prefer", [])) & rejected_keys
        if avoid_in_cands:
            hard.append({"type": "cited_rule_avoid_violated", "rule_id": rid,
                         "models": sorted(avoid_in_cands), "severity": "hard"})
        if prefer_in_rej:
            hard.append({"type": "cited_rule_prefer_rejected", "rule_id": rid,
                         "models": sorted(prefer_in_rej), "severity": "hard"})
    return hard


def detect_soft_conflicts(ctx, trace, plan, output):
    """Non-blocking conflicts surfaced as info in the UI. Takes `output` so cited refs
    can be subtracted from the soft set (cited cases are already handled by hard detection)."""
    soft = []
    candidate_keys = {c.model_key for c in plan.candidates}

    retrieved_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in trace.retrieved_experience_ids and e.best_model
    }
    cited_experience_ids = {ref.source_id for ref in _collect_all_refs(output) if ref.source == "experience"}
    cited_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in cited_experience_ids and e.best_model
    }
    soft_omitted = (retrieved_winners - cited_winners) - candidate_keys
    if soft_omitted:
        soft.append({
            "type": "retrieved_experience_winner_not_selected",
            "models": sorted(soft_omitted),
            "summary": (
                f"{len(soft_omitted)} model(s) won in retrieved experiences but were not cited "
                f"or selected: {sorted(soft_omitted)}."
            ),
        })
    return soft
```

---

## Section 4 — UI redesign (Planner tab)

Component tree (under `frontend/components/pipeline/planner/`):
- `PlannerSummaryHeader` — chip grid + planner_status badge
- `DecisionBasisCard` — primary/secondary evidence chips + strategy narrative
- `ConflictPanel(hard, soft)` — amber border for hard, neutral for soft; only renders subsections that have items
- `CandidateRationaleList` — `.map(CandidateCard)` sorted by priority
- `CandidateCard` — model_key + priority + reason + evidence_refs chip list + risks bullet list, expandable
- `RejectedModelsList` — collapsed by default, expandable per item
- `RejectedModelCard` — model_key + reason + evidence_refs + reconsider_if (when present)
- `EvidenceQualityCard` — counts + relevance tier breakdown + scale-comparison warning
- `SimilarPastRunsList` — sorted by similarity_score desc
- `ExperienceCard` — tier badge + similarity_score + matched_buckets / mismatched_buckets / target_scale_note + best_model + metric (with scale warning if mismatched) + cited badge
- `MatchedRulesList` (mostly today's component, reuses existing layout)
- `PlannerWarningsList` (today's)
- audit drawer at bottom: `<details>View full planning analysis</details>` — preserves the long `planning_analysis` text for debugging

Mockup ordering (vertical):
```
PlannerSummaryHeader
DecisionBasisCard
ConflictPanel             (conditional)
CandidateRationaleList
RejectedModelsList
EvidenceQualityCard
SimilarPastRunsList
MatchedRulesList
PlannerWarningsList       (conditional)
<details>Full planning analysis</details>
```

### Type extensions (`frontend/types/api.ts`)

```ts
export interface EvidenceReference {
  source: 'dataset_profile' | 'task_metadata' | 'registry' | 'experience' | 'rule'
  source_id?: string | null
  relevance_note?: string
}

export interface CandidateFull {
  model_key: string
  priority: number
  reason: string
  evidence_refs: EvidenceReference[]
  risks: string[]
}

export interface RejectedModelFull {
  model_key: string
  reason: string
  evidence_refs: EvidenceReference[]
  reconsider_if?: string | null
}

export interface DecisionBasis {
  primary_evidence: EvidenceReference[]
  secondary_evidence: EvidenceReference[]
  final_strategy: string
}

export interface EvidenceConflict {
  summary: string
  affected_models: string[]
  conflicting_evidence_refs: EvidenceReference[]
  resolution: string
}

export interface SoftConflict {
  type: string                      // e.g. 'retrieved_experience_winner_not_selected'
  models: string[]
  summary: string
}

export interface ExperienceSummary {
  experience_id: string
  similarity_score: number
  relevance_tier: 'high' | 'medium' | 'low'
  matched_buckets: string[]
  mismatched_buckets: string[]
  target_scale_note: string | null
  dataset_name: string
  problem_type: string
  best_model: string
  validation_score: number
  metric_name?: string
}

export interface PlannerContextData {
  retrieved_experiences: ExperienceSummary[]
  matched_rules: MatchedRule[]
  evidence_used: EvidenceReference[]
  planning_analysis: string
  plan_summary: {
    candidates_full: CandidateFull[]
    rejected_full: RejectedModelFull[]
    candidate_models: string[]              // legacy back-compat
    models_not_recommended: string[]        // legacy back-compat
  }
  warnings: string[]
  decision_basis: DecisionBasis
  evidence_conflicts: EvidenceConflict[]
  soft_conflicts: SoftConflict[]
  cited_experience_ids: string[]
  cited_rule_ids: string[]
  planner_status: 'ok' | 'retry_ok' | 'failed'
}
```

### Stepper badge flip

`PipelineStepper.tsx`: change the `model_planning` entry's `type` from `'llm'` to `'agent'`.

### RunHeader taxonomy

`RunHeader.tsx` accepts `agents: string[]`, `llmNodes: string[]`, `deterministic: string[]`. Renders as a single dense `flex-wrap` line:

```tsx
<div className="mt-1 flex flex-wrap gap-x-4 text-[11px] text-[var(--color-fg-subtle)]">
  <span>Agents: {agents.join(' · ')}</span>
  <span>LLM: {llmNodes.join(' · ')}</span>
  <span>Deterministic: {deterministic.join(' · ')}</span>
</div>
```

Backend `run_info` event extends `data.node_categories: {agents, llm_nodes, deterministic}`. Frontend reads this; falls back to legacy `data.models` for old runs.

---

## Testing strategy

| Layer | Test target | Approach |
|---|---|---|
| Tools | each of 4 tools | unit: bound profile/task_meta produces expected output; trace recorded; dedup works; max_tool_calls short-circuits |
| Trace | ToolTrace | unit: dedup, count increment, model_dump shape |
| Validation | `_check_plan_integrity` | parametrized failure cases: missing tool, priority dup, priority < 1, candidate overlap, missing FC settings, invalid val_strategy, invalid exog strategy, known_future in unknown-future overrides, missing registry self-cite |
| Validation | `_check_plan_exhaustiveness` | unit: missing model raises with sorted list |
| Validation | `_check_evidence_references_hybrid` | unit: each source type, both deterministic and observed paths |
| Validation | `_check_conflict_resolution_present_if_flagged` | unit: hard conflict flagged + empty conflicts raises; resolved conflicts pass |
| Conflict detection | `_detect_conflicts` | unit: cited-experience-winner-omitted; cited-rule-avoid-violated; cited-rule-prefer-rejected; soft branch separate |
| Context | `build_planner_validation_context` | unit: deterministic across runs; identical input ⇒ identical output |
| Agent | `build_planner_agent` | unit (mock LLM with canned tool calls): returns structured PlannerOutput |
| Node | `planner_node` | integration with mocked LLM: full flow including retry on validation failure |
| Frontend | `PlannerPanel` | RTL: renders without crashing; sub-components render with their data shapes |
| Frontend | `ConflictPanel` | RTL: amber for hard, neutral for soft, hidden when both empty |
| Frontend | `CandidateRationaleList` | RTL: sorted by priority; expandable evidence/risks |
| Frontend | `SimilarPastRunsList` | RTL: sorted by similarity_score; tier badges; "cited" pill on cited only |
| Frontend | `RunHeader` | RTL: three category labels, fallback to legacy when node_categories absent |

The pre-existing `use-approve.test.tsx` failure remains acceptable as documented in the prior refactor — out of scope.

## Open boundaries / what could change

- `create_agent` in `langchain==1.2.14` supports `response_format` — confirmed at spec time. No fallback needed.
- Soft-conflict UI styling may need tuning after seeing real output — placeholder is neutral zinc, may want a subtle sky/info color if too quiet.
- `target_scale_note` heuristic threshold (when to warn) — currently "an order of magnitude" (~10×). Refine if warning fires too often or too rarely after slice smokes.
- `settings.planner_timeout_seconds`: **NOT enforced in v2**. `recursion_limit` (default 10) and `planner_max_tool_calls` (default 6, with `planner_max_inspect_calls=3` per-tool) are the only bounds that actually constrain agent execution. The `planner_timeout_seconds` setting is reserved for a future iteration that adds wall-clock enforcement (e.g., a `concurrent.futures` wrapper or middleware). Spec must not claim wall-clock timeout protection.
- Backward compat for legacy `ExperienceRecord`s lacking `target_*` numeric stats: `compare_target_scales` returns `None` → no scale warning shown for those experiences. Acceptable; seed script could be re-run to populate if needed.

## Acceptance criteria

1. The planner actively calls `list_available_models`, `retrieve_similar_experiences`, `retrieve_ml_knowledge` on every successful run. Validation fails the run if not.
2. The planner may optionally call `inspect_model_details`.
3. Final output is a valid `PlannerOutput` with all required fields filled (`decision_basis`, candidates with priority+reason+evidence_refs+risks, rejected with evidence_refs).
4. Every candidate's `evidence_refs` contains at least one registry reference where `source_id == model_key`. Same for rejected.
5. The planner cannot cite an experience or rule it did not retrieve via tools.
6. When deterministic hard-conflict detection flags ≥1 conflict, `output.evidence_conflicts` is non-empty and each `resolution` is non-empty. Otherwise the run is rejected and retried once.
7. Soft conflicts (retrieved-but-not-cited winners) appear in the UI as info; they do not gate validation.
8. The executor remains deterministic and trains only `plan.candidates`, sorted by priority.
9. All existing TrainingPlan validation checks still pass; `_check_plan_integrity` and `_check_plan_exhaustiveness` enforce stricter invariants than before.
10. `PipelineStepper` shows `model_planning` with the `Agent` badge.
11. `RunHeader` taxonomy reads `Agents: data_validator · planner | LLM: report_writer | Deterministic: controller · executor · evaluation · deployer`.
12. The Planner tab in `ResultsDashboard` renders: summary header, decision basis, conditional conflict panel, per-candidate rationale cards (sorted by priority), rejected models list (collapsed), evidence quality card, enriched similar-past-runs cards with relevance tier + buckets + target-scale notes, matched rules, warnings, and a collapsible "View full planning analysis" drawer.
13. No embeddings or vector search introduced — deterministic similarity retrieval preserved exactly.
14. No new API surface beyond what's needed for the SSE payload extensions.
