# Results Chapter Design (Capítulo 6)

**Date:** 2026-06-09
**Status:** Approved
**Scope:** Chapter 6 of the TFG thesis — Resultados

---

## Context

The system is a multi-agent MLOps platform for time-series forecasting (primary), with secondary support for classification and regression. The main architectural claims are:

1. A deterministic workflow controller orchestrates specialist agents — the controller is robust, not the agents.
2. A planner agent selects models adaptively using an experience pool (similarity retrieval) and static rules.
3. The experience pool accumulates empirical priors that make future decisions better.
4. HITL gates are lightweight because the agents make good-enough decisions that don't require heavy human correction.

The results chapter must provide empirical evidence for all four claims.

**Primary focus: forecasting.** Classification/regression appear only as generality demonstration (one brief case study).

---

## Evidence available at writing time

- **53 experience pool JSON records** — 6 classification, 5 regression, 14 forecasting benchmark entries (plus live pipeline runs)
- **14 curated forecasting benchmark datasets** — run through `run_benchmark.py`, covering 3 regime families: statistical (4), supervised (5), random_walk (5)
- **Live pipeline runs** — grid_demand_forecast (already completed, 2026-06-08/09), plus runs to be executed during writing
- **EventLog timestamps** from the frontend — per-phase wall time for each live run
- **TokenCostCard data** — LLM token usage per run

---

## Chapter 6 Structure

---

### 6.1 — Evaluación del pool de experiencias (benchmarks)

**Purpose:** Validate that the 14 forecasting benchmark experiences in the pool are correct and represent trustworthy priors for the planner.

**Content:**

1. **Table: 14 forecasting benchmark results**

   | Dataset | Frequency | Horizon | Champion model | Family | Family check | RMSE | SMAPE |
   |---------|-----------|---------|----------------|--------|--------------|------|-------|
   | air_passengers | MS | 12 | ETS | statistical | ✓ | … | … |
   | co2_mauna_loa | MS | 6 | ETS/AutoARIMA | statistical | ✓ | … | … |
   | … | … | … | … | … | … | … | … |

2. **Summary statistics:**
   - Family match rate (e.g. "12 of 14 datasets → correct regime")
   - Model selection frequency: how often ETS/AutoARIMA won (statistical), LightGBM/RF (supervised), naive (random_walk)
   - Per-regime commentary: explain why the pattern holds (statistical short series → no exog → ETS wins; exogenous daily → tree ensembles generalize better; financial EMH → naive wins)

**Data source:** Extract from the 14 `*_forecasting_*.json` files in `experience_pool/`. No new runs needed.

---

### 6.2 — Comportamiento agéntico: casos de estudio

Five focused case studies, each targeting a different agentic claim.

---

#### 6.2.1 — Recuperación exacta del pool

**Dataset:** `air_passengers` (already in pool from benchmark)
**Run:** Execute through the live UI pipeline (not the benchmark runner — needs to go through the full agentic graph).

**What to show:**
- DatasetProfile computed from the uploaded CSV matches the stored profile → similarity ≈ 1.0
- Planner tool trace: `retrieve_experiences` returns air_passengers as top hit
- Planner cites that experience as primary evidence in its `reason` field
- ETS is recommended — same as the benchmark champion
- Final test metrics (RMSE, SMAPE)
- Screenshot: EventLog showing the retrieval tool call + planner plan output

**Thesis claim validated:** The similarity function correctly identifies identical datasets; the pool is actually consulted by the planner.

---

#### 6.2.2 — Recuperación por similitud (dataset distinto)

**Dataset:** `grid_demand_forecast` (custom dataset, not in pool — already run on 2026-06-08/09)

**What to show:**
- Dataset profile (short, weekly, seasonal + trend, no exog of future kind)
- Top retrieved experiences from pool: air_passengers or similar (weekly seasonal statistical series) with similarity ~0.7–0.8
- Planner reasoning trace: cites retrieved experiences + static rules → recommends ETS
- Final test metrics + forecast chart (train actuals vs test actuals vs test predicted)
- Phase timing table (see 6.3)

**Thesis claim validated:** The planner generalizes from similar-but-not-identical experiences.

---

#### 6.2.3 — Supervisado con variables exógenas

**Dataset:** `metro_traffic_volume` or `bike_sharing_daily` (through live UI — these are in the pool from benchmark, so use the *other* one not in pool, or use a variation)
**Alternative:** Upload a new custom supervised dataset with known exogenous features.

**What to show:**
- Planner detects `exogenous_features_available = true` in profile
- Pool retrieves a supervised experience as top hit (bike_sharing_daily or metro)
- Planner recommends LightGBM/RF (supervised family) over ETS, explicitly citing exogenous feature availability in the reason
- HITL approval card screenshot: training plan shown to human, approved
- Final test metrics

**Thesis claim validated:** The planner correctly switches model family when exogenous features are available; the experience pool correctly surfaces supervised priors.

---

#### 6.2.4 — Agente data_validator: tamaño y join discovery

**Purpose:** Stress-test the data_validator agent specifically — the only agent the chapter doesn't otherwise examine in depth.

**Sub-experiment A — Dataset size stress test:**
- Run 3 datasets of different sizes (e.g. ~100 rows / ~1,000 rows / ~10,000 rows) through the validator only (or through the full pipeline, noting validator phase time)
- Show: does profiling time scale linearly? Does the agent's behavior change (e.g. more missing-value warnings on dirty larger datasets)?
- Table: dataset size → validator phase time → key profile fields detected

**Sub-experiment B — Join discovery:**
- Upload a base dataset that has a joinable column (e.g. date/week) alongside a secondary dataset with matching key
- Show: `discover_joins` tool call in EventLog, join candidate surfaced
- Then upload a dataset where no sensible join exists → agent correctly skips or returns empty candidates
- Screenshot: join discovery tool call result in EventLog

**Thesis claim validated:** The data_validator is robust across dataset sizes; the join discovery tool is exercised meaningfully, not trivially.

---

#### 6.2.5 — Robustez del controlador determinista

**Purpose:** Empirically validate that the deterministic workflow controller handles errors gracefully — failures route correctly, no crashes.

**Forced error scenarios (2–3):**

1. **Malformed dataset** — upload a CSV with all-NaN target column or completely missing target column. Expected: validator raises an error state; controller routes to an informative error response without crashing the graph.

2. **Schema mismatch** — upload a dataset where the declared target column doesn't exist in the file. Expected: schema validation step catches it before training is attempted.

3. **Training failure edge case** — upload a dataset too small for the requested horizon (e.g. 15 rows, horizon=12). Expected: executor or validator detects this and the controller routes to a graceful stop.

**What to show:**
- Screenshot or EventLog excerpt for each error scenario
- State the expected vs. observed behavior
- Confirm: no unhandled exceptions, no silent data corruption, controller always reaches a terminal state

**Thesis claim validated:** The deterministic spine is robust; agentic failures don't cascade.

---

#### 6.2.6 — Generalidad del sistema (clasificación)

**Dataset:** Iris or Wine (1 page maximum)
**Run:** Through live UI pipeline.

**What to show:**
- Planner selects a tree-based classifier (not ETS, not naive) — demonstrates the system correctly switches problem domain
- One screenshot of the planner output / training result
- 1–2 sentences: the architecture is problem-agnostic; forecasting improvements don't break classification

---

### 6.3 — Coste agéntico y supervisión humana

**Purpose:** Quantify the cost of agentic intelligence and argue the tradeoff is favorable.

---

#### 6.3.1 — Desglose temporal por fases

**Data source:** EventLog timestamps from case studies 6.2.2 and 6.2.3.

**Table: phase-by-phase wall time (two runs):**

| Phase | Type | Run A (grid_demand) | Run B (supervised) |
|-------|------|--------------------|--------------------|
| data_validator | Agentic (LLM) | Xs | Xs |
| planner | Agentic (LLM) | Xs | Xs |
| executor | Deterministic | Xs | Xs |
| evaluation gate | Deterministic | Xs | Xs |
| report_writer | Agentic (LLM) | Xs | Xs |
| deployer | Deterministic | Xs | Xs |
| **Total** | | Xs | Xs |
| **Deterministic only** | | Xs | Xs |
| **LLM overhead** | | Xs | Xs |

**Analysis:** LLM overhead = total − (executor + evaluation + deployer). A traditional deterministic pipeline would only pay the deterministic portion. The overhead buys: adaptive model selection, experience accumulation, natural-language audit reports, HITL gates.

**Honest note:** LLM latency is bounded by the GitHub Models free tier rate limits (150 RPD) and API latency, not by the system's own computation. On a paid tier the agentic phases would be significantly faster.

---

#### 6.3.2 — Supervisión humana en las puertas HITL

**Purpose:** Show that HITL approval is lightweight when agents make good decisions, and reason about when it would be heavy.

**Content:**

1. **From the case studies:** Record the time spent at each HITL gate (dataset approval + deployment approval). In the happy path (agent plan is correct), approval is a quick review — seconds to a minute. Screenshot of the approval card showing what the human sees.

2. **The argument:** The approval cost scales inversely with agent decision quality. Because the planner uses the experience pool + static rules + schema validation, the human's job at the gate is to confirm, not to design. If the pool were empty or the planner hallucinated a plan, the human would need to read, understand, and re-write the training plan — taking significantly longer than doing it manually.

3. **The honest limitation:** For datasets genuinely unlike anything in the pool (novel domain, unusual frequency, new exog structure), the planner's plan is less reliable → the human must spend more time at the gate. This is the main incentive for continuing to grow the experience pool — more priors → lighter gates → closer to true automation.

4. **Bridge to Trabajo Futuro:** mention that a larger pool (achieved through more benchmark runs or production use) progressively reduces the HITL burden, approaching fully automated operation for common dataset profiles.

---

## Experiments to run (checklist)

Before writing the chapter, the following live pipeline runs must be completed:

- [ ] **air_passengers** through live UI (exact-match retrieval demo — 6.2.1)
- [ ] **Custom supervised forecasting dataset with exog** through live UI (6.2.3) — or metro/bike_sharing via UI if not already done as live run
- [ ] **3 datasets of different sizes** through validator (6.2.4A)
- [ ] **Join discovery pair** — two datasets with a joinable column (6.2.4B)
- [ ] **3 forced error scenarios** through live UI (6.2.5)
- [ ] **Iris or Wine** through live UI (6.2.6)
- [ ] **Extract EventLog timestamps** from grid_demand run(s) already in pool (6.3.1)
- [ ] **Record HITL gate time** during each live run (6.3.2)

---

## Evidence format per case study

Each case study should contain:
1. **Dataset description** — 2–3 sentences on what makes this dataset interesting for this case study
2. **Agent trace excerpt** — EventLog screenshot or JSON snippet showing the key tool call / decision
3. **Key result** — metric table or timing table
4. **1-sentence conclusion** — what this confirms about the system

---

## Self-review

- **Completeness:** All four thesis claims have at least one case study providing direct evidence.
- **Scope:** Forecasting-primary structure respected; classification is 1 page.
- **No tautology:** The exact-match demo (6.2.1) is framed as validation of the similarity function, not as a benchmark repeat.
- **Honest limitations acknowledged:** LLM latency rate limit caveat (6.3.1), novel-dataset HITL cost (6.3.2).
- **No new development required** for 6.1, 6.2.2, 6.3.1 — data already exists. The remaining sections need live runs only, no code changes.
