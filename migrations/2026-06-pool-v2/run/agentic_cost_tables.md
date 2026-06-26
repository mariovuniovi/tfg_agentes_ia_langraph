# Agentic pipeline — cost / token / timing tables (2026-06-26)

Re-run of the per-node cost/time tables on the `data/samples` demo datasets with the
**new 15-dataset forecasting pool** and the **frequency-aware seasonality search**.

## Method
- Full agentic pipeline (data_validator → planner → executor → evaluation → report_writer → deployer),
  run headless with both HITL gates auto-approved. Telemetry captured with the *same*
  `parse_stream_event` + `estimate_cost` the app uses, so numbers match an app run in expectation.
- **N = 5 runs per dataset** (Large = 8 clean runs), averaged as **mean ± std** to absorb LLM stochasticity.
- Models (fixed config, identical every run): `data_validator` & `planner` = `gpt-5.4-mini`,
  `report_writer` = `gpt-5.4-nano`.
- 2 Large runs were discarded as **CPU-contention artifacts** (see Caveats).

## Summary (mean ± std)

| Dataset | Type | n | cost $ | total compute s | executor s | LLM-time % |
|---|---|---|---|---|---|---|
| Revenue (120, monthly) | forecasting | 5 | **0.093 ± 0.008** | 138 ± 29 | 9.0 ± 0.6 | 93% |
| Bakery (1 000, daily) | forecasting | 5 | **0.120 ± 0.020** | 249 ± 49 | 50.3 ± 8.4 | 79% |
| Large (10 000, hourly) | forecasting | 8 | **0.115 ± 0.018** | 255 ± 34 | **88.2 ± 7.7** | 65% |
| Grid (4-file join) | forecasting | 5 | **0.184 ± 0.016** | 180 ± 35 | 10.1 ± 0.1 | 94% |
| AirPassengers (3-file join) | forecasting | 5 | **0.154 ± 0.018** | 143 ± 26 | 8.6 ± 0.4 | 94% |
| Retail (4-file join) | regression | 5 | **0.165 ± 0.035** | 146 ± 15 | 6.5 ± 1.4 | 95% |

## Per-node (mean ± std — time · cost)

| Node | Revenue | Bakery | Large | Grid | AirPass | Retail |
|---|---|---|---|---|---|---|
| data_validator | 38s · $.033 | 33s · $.032 | 25s · $.031 | **78s · $.125** | **42s · $.082** | **60s · $.115** |
| planner | 84s · $.056 | 158s · $.083 | 133s · $.079 | 84s · $.055 | 85s · $.068 | 73s · $.047 |
| executor | 9s | 50s | 88s | 10s | 9s | 6s |
| evaluation | 0.1s | 0.0s | 0.1s | 0.0s | 0.0s | 0.0s |
| report_writer | 6s · $.005 | 7s · $.005 | 9s · $.005 | 8s · $.005 | 7s · $.005 | 7s · $.003 |
| deployer | 0.3s | 0.3s | 0.3s | 0.3s | 0.3s | 0.0s |

## Champion-model stability (over the runs)

| Dataset | Champions | Stable? |
|---|---|---|
| Revenue | `auto_arima` ×5 | 100% |
| Grid | `auto_arima` ×N | 100% |
| AirPassengers | `auto_arima` ×N | 100% |
| Bakery | `ets` (mostly), occ. `auto_arima` | flips between 2 statistical models |
| Retail | `random_forest` (mostly), occ. `ridge` | flips between 2 regression models |
| Large | `extra_trees` (mostly), occ. `random_forest` | flips between 2 tree ensembles |

## Key findings
1. **Cost is reproducible** (±5–19%). A single app run is representative *for cost*.
2. **Joins relocate cost to `data_validator`** — join discovery ~3–4×'s validator cost ($0.08–0.13 vs
   $0.033 single-file); planner cost stays flat. The multi-table tax is a data-validation cost.
3. **The Large 10 000-row case is tractable and stable** (~88s training), where it previously timed out
   (>600s). The seasonality work (frequency-aware `season_length`, length-gated AutoARIMA approximation,
   bounded rolling window) is what makes it feasible.
4. **Only the model choice is LLM-driven.** The planner LLM picks *which* candidate models to try; the
   **trial budget (60), search spaces, and validation strategy are deterministic defaults** (the prompt
   never asks the LLM to set them). The executor then deterministically picks the lowest-RMSE model.
   → champion is 100% stable for strong-signal series; flips only between statistically-tied models.

## Caveats (measurement honesty)
- **CPU contention:** running analysis scripts concurrently with a measurement run starves the local ML
  training (training is CPU-bound, single machine). This inflated 2 Large runs to 744s / 1607s executor
  (the *same* model that normally trains in ~6s took 1514s under contention). Those 2 runs were excluded;
  the clean 8 give 88.2 ± 7.7s. All other datasets train in <65s and showed no contention.
- **Intermittent connectivity:** the AI steps call the OpenAI API over the network. Connection drops caused
  one 91-min `RemoteProtocolError` hang and add noise to the *wall-time* of the LLM nodes (validator,
  planner, report_writer). Mitigated with a request timeout + retries + an in-process watchdog. **Token/cost
  is unaffected** (usage is counted from successful replies). → report cost as the stable metric; treat
  LLM-node *times* as network-dependent (hence the std on planner time).
- Deploy rates vary (Large/Retail low) due to inter-run MLflow-registry comparison (later candidates are
  compared against a champion an earlier run promoted) — a batch artifact, not a pipeline property.
