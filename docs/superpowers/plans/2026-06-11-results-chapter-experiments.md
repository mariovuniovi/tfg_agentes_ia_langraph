# Results Chapter — Experiment Execution Plan

**Date:** 2026-06-11  
**Target:** Chapter 6 (Resultados) — sections 6.2 and 6.3  
**Total runs:** 13 (+ 0 for 6.2.2 which reuses existing logs)

---

## Summary table

| Section | Dataset(s) | Runs | Purpose |
|---------|-----------|------|---------|
| 6.2.1 | `air_passengers.csv` | 1 | Exact-match retrieval from pool |
| 6.2.2 | *(grid_demand — already run 2026-06-08)* | 0 | Reuse existing pool JSON logs |
| 6.2.3 | `medium_daily_bakery.csv` | 1 | Supervised forecasting with exog |
| 6.2.4A | small / medium / large | 3 | Validator size scaling |
| 6.2.4B | ap_split + forecasting2 + retail + catalog | 4 | Join discovery stress test |
| 6.2.5 | 3 broken CSVs | 3 | Controller robustness |
| 6.2.6 | `iris.csv` | 1 | Generality — classification domain |
| **Total** | | **13** | |

---

## Recommended execution order

```
1. 6.2.6  — iris (2 min)       → confirms system boots correctly
2. 6.2.5  — 3 broken (5 min)   → robustness before long runs
3. 6.2.4A — size test (15 min) → validator only, no training needed
4. 6.2.1  — air_passengers (10 min) → exact-match demo
5. 6.2.3  — bakery with exog (15 min) → HITL + timing for 6.3
6. 6.2.4B — join discovery (20 min)
7. 6.2.2  — extract from existing JSON (no new run)
```

Run error scenarios (6.2.5) before production runs to avoid accidentally storing broken records in the pool.

---

## 6.2.1 — Exact-match pool retrieval

**File:** `data/samples/join_discovery/air_passengers_base.csv`  
**Schema:** `air_passengers_joined_schema.json` (problem_type=forecasting, target=passengers, datetime=month, frequency=MS, horizon=12)

| What to annotate | Expected value |
|-----------------|----------------|
| Similarity score of top `retrieve_experiences` hit | ≈ 1.0 |
| Model recommended by planner | AutoARIMA or ETS |
| Planner `reason` field — copy verbatim | must cite air_passengers experience |
| Final RMSE | ~23.6 |
| Screenshot | EventLog with `retrieve_experiences` tool call visible |

---

## 6.2.2 — Similarity retrieval (different dataset)  *(no new run)*

**Source:** `experience_pool/grid_demand_forecast_forecasting_2026-06-08_001.json`

| What to extract | Field path |
|----------------|-----------|
| Top retrieved experience + similarity score | `planner_output.retrieved_experiences[0]` |
| Model recommended | `planner_output.selected_model` |
| Planner reason (verbatim excerpt) | `planner_output.reason` |
| Forecast chart | Already saved in MLflow run |

If `planner_output` in the JSON lacks enough detail → run grid_demand once through the live UI.

---

## 6.2.3 — Supervised forecasting with exogenous variables

**File:** `data/samples/size_test/medium_daily_bakery.csv`  
**Schema:** `medium_daily_bakery_schema.json` (target=units_sold, datetime=date, frequency=D, horizon=14, exog: is_weekend=known_future, day_of_week=known_future)

| What to annotate | Expected value |
|-----------------|----------------|
| `exogenous_features_available` in dataset profile | true |
| Top retrieved experience + similarity score | metro_traffic_volume, ~0.5–0.7 |
| Model recommended | RandomForest or LightGBM (NOT ETS) |
| Planner `reason` mentioning exog | copy verbatim |
| Screenshot | HITL approval card showing training plan |
| HITL gate time (dataset approval) | record start→approve timestamp |
| HITL gate time (deployment approval) | record start→approve timestamp |
| Final RMSE | — |
| Phase timestamps (all 6 phases) | for 6.3.1 timing table |

---

## 6.2.4A — Validator size stress test

Run each dataset only until the `data_validator` phase completes. Training is not required.

| Run | File | Rows | Frequency | Target |
|-----|------|------|-----------|--------|
| A1 | `size_test/small_monthly_revenue.csv` | 120 | MS | `revenue_eur` |
| A2 | `size_test/medium_daily_bakery.csv` | 1,000 | D | `units_sold` |
| A3 | `size_test/large_hourly_factory.csv` | 10,000 | H | `consumption_kwh` |

**Annotate per run:**
- `data_validator` phase duration: timestamp_end − timestamp_start from EventLog
- Profile fields detected: n_rows tier, frequency, seasonality_detected, trend_detected, missing_rate
- Any agent warnings about anomalies or data quality

**Thesis table to produce:**

| Dataset | Rows | Validator time (s) | n_rows tier | seasonality | trend | missing |
|---------|------|--------------------|-------------|-------------|-------|---------|
| small   | 120  | ?                  | ?           | ?           | ?     | ?       |
| medium  | 1,000| ?                  | ?           | ?           | ?     | ?       |
| large   | 10,000| ?                 | ?           | ?           | ?     | ?       |

---

## 6.2.4B — Join discovery

Upload all files simultaneously per sub-experiment. Record the `discover_joins` tool call result.

| Run | Files | Expected joins | Key types |
|-----|-------|---------------|-----------|
| B1 | `air_passengers_base.csv` + `aviation_context.csv` | 1 (month) | date string |
| B2 | `forecasting2/` — all 4 CSVs | 3 (timestamp_week / obs_date / week_start_date / week_ref) | date strings, different names |
| B3 | `retail_star_schema/` — all 5 CSVs | 5 (customer_id, product_id, store_id, brand_id, city_code) | int FKs + string attribute |
| B4 | `product_catalog.csv` alone | 0 | negative control |

**Annotate per run:**
- Full JSON returned by `discover_joins` tool call (copy from EventLog)
- For B3: did the agent detect `city_code` (shared attribute join)? → key result
- For B4: did the agent return an empty list? → must be yes
- Screenshot: EventLog with `discover_joins` call and result expanded
- `data_validator` phase duration

---

## 6.2.5 — Controller robustness

| Run | File | Error injected | Expected behavior |
|-----|------|---------------|-------------------|
| E1 | `broken/all_nan_target.csv` | `revenue` = 100% NaN | Validator detects → informative stop |
| E2 | `broken/missing_target_col.csv` | `sales` column absent (schema declares it) | Schema mismatch → stop before training |
| E3 | `broken/too_small_for_horizon.csv` | 15 rows, horizon=12 | Executor or validator → graceful stop |

**Annotate per run:**
- Node where the error was detected (data_validator / executor)
- Exact error message shown to the user (copy verbatim)
- Did the graph reach a clean terminal state? (yes / no)
- Screenshot of the EventLog or error message in the UI

---

## 6.2.6 — Generality: classification domain

**File:** `data/samples/iris.csv`  
**Schema:** problem_type=classification, target=species (or target), 150 rows

| What to annotate | Expected value |
|-----------------|----------------|
| Model recommended by planner | Logistic Regression (very_small → rule 1) |
| Does planner avoid ETS/naive? | yes — confirms domain switching |
| Final macro-F1 | ~0.97 |
| Screenshot | Planner output showing model selection |

---

## 6.3.1 — Phase-by-phase timing table

**Data source:** EventLog from 6.2.2 (grid_demand) and 6.2.3 (bakery).

Fill this table during the runs:

| Phase | Type | Run A: grid_demand | Run B: bakery |
|-------|------|--------------------|---------------|
| data_validator | Agentic (LLM) | ? s | ? s |
| planner | Agentic (LLM) | ? s | ? s |
| executor | Deterministic | ? s | ? s |
| evaluation | Deterministic | ? s | ? s |
| report_writer | Agentic (LLM) | ? s | ? s |
| deployer | Deterministic | ? s | ? s |
| **Total** | | **? s** | **? s** |
| **LLM overhead** | | **? s** | **? s** |

LLM overhead = total − (executor + evaluation + deployer).

---

## 6.3.2 — HITL gate timing

Record during 6.2.1 and 6.2.3:

| Gate | Run | Presented at | Approved at | Duration |
|------|-----|-------------|-------------|----------|
| Dataset approval | 6.2.1 air_passengers | ? | ? | ? s |
| Deployment approval | 6.2.1 air_passengers | ? | ? | ? s |
| Dataset approval | 6.2.3 bakery | ? | ? | ? s |
| Deployment approval | 6.2.3 bakery | ? | ? | ? s |

---

## Dataset + schema quick reference

| Section | Primary file(s) | Schema file |
|---------|----------------|-------------|
| 6.2.1 | `join_discovery/air_passengers_base.csv` | `air_passengers_joined_schema.json` |
| 6.2.3 | `size_test/medium_daily_bakery.csv` | `medium_daily_bakery_schema.json` |
| 6.2.4A small | `size_test/small_monthly_revenue.csv` | `small_monthly_revenue_schema.json` |
| 6.2.4A medium | `size_test/medium_daily_bakery.csv` | `medium_daily_bakery_schema.json` |
| 6.2.4A large | `size_test/large_hourly_factory.csv` | `large_hourly_factory_schema.json` |
| 6.2.4B B1 | `join_discovery/air_passengers_base.csv` + `aviation_context.csv` | `air_passengers_joined_schema.json` |
| 6.2.4B B2 | `forecasting2/` (4 CSVs) | `forecasting2/grid_demand_schema.json` |
| 6.2.4B B3 | `retail_star_schema/` (5 CSVs) | `retail_sales_schema.json` |
| 6.2.4B B4 | `join_discovery/product_catalog.csv` | `product_catalog_schema.json` |
| 6.2.5 E1 | `broken/all_nan_target.csv` | `all_nan_target_schema.json` |
| 6.2.5 E2 | `broken/missing_target_col.csv` | `missing_target_col_schema.json` |
| 6.2.5 E3 | `broken/too_small_for_horizon.csv` | `too_small_for_horizon_schema.json` |
| 6.2.6 | `iris.csv` | *(declare inline in UI)* |
