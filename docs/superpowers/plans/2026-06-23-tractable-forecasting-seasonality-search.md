# Tractable, Frequency-Aware Forecasting Seasonality Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the statistical forecasters search a richer, frequency-appropriate set of seasonal periods (capturing sub-annual cycles) while keeping each candidate's validation tractable on long/high-frequency series.

**Architecture:** Three coupled levers. (1) A shared seasonality policy returns a *per-model-family* grid of candidate `season_length`s, pruned to periods the data can actually estimate (≥2 cycles, measured against the smallest validation fold). (2) `AutoARIMA` uses **length-gated** `approximation` — the fast CSS order search on long series (`n_obs > 500`), where exact search is infeasible and CSS≈MLE selection anyway, and exact search on short/medium series where it is cheap and more accurate. (3) For long series, validation switches to a **bounded rolling window** sized to the largest candidate season, so `AutoETS` (which has no approximation knob) and ARIMA both fit on a few seasonal cycles instead of the whole history. Together the richer grids become affordable; the champion is still selected on the current dataset's validation.

**Tech Stack:** Python 3.12, statsforecast (AutoARIMA/AutoETS/SeasonalNaive), Optuna, pandas, pydantic, pytest, uv.

## Global Constraints

- Use `uv run` for every command — never activate the venv manually.
- Keep `src/mlops_agents/forecasting/seasonality.py` free of executor/model imports (no dependency cycles) — stdlib + `re` only.
- Type hints everywhere; match existing module style.
- Never mutate a registry `SearchSpaceSpec` in place — always `model_copy`.
- Pre-existing lint warnings in `executor.py` (`N806`, `SIM108`) are out of scope; do not "fix" unrelated code.
- Tests must not make real LLM calls. Statistical-model tests use real statsforecast fits on tiny synthetic series (fast).
- Frequency aliases are already normalized by `normalize_frequency` (`MS/ME→M`, `W-MON→W`, `h→H`); reuse it, do not re-implement.

## Measured facts this plan relies on (from profiling, do not re-derive)

- AutoARIMA `m=24`, `approximation=False`: n=240→4.9s, n=500→**45.2s**. `approximation=True`: n=240/500→**0.8s**, n=2000→6.9s, n=10000→33.3s.
- AutoETS `m=24` (no approximation knob): n=240→0.1s, n=500→0.3s, n=1000→0.5s, n=10000→**10.9s**. Window cap is its only lever.
- `seasonal_naive` fit is ~free (≈0.01s) regardless of period → can afford the richest grid.
- `AutoARIMA(...).approximation` attribute exists and reflects the constructor arg.

## File Structure

- `src/mlops_agents/forecasting/seasonality.py` — **modify**. Add `season_length_grid(model_key, freq, n_obs)` and `max_season_length(freq)`. Keep existing `normalize_frequency`, `canonical_season_length`, `default_season_length` (still used by factories/exog_extender).
- `src/mlops_agents/training/executor.py` — **modify**. Change `_narrow_seasonality_to_freq` to use `season_length_grid`; add `_min_fold_train_len` and prune against it; inject per-fit `series_length` into `task_metadata` at each AutoARIMA fit (validation fold + retrain) for the approximation policy; pass `max_season_length(freq)` into the rolling-window resolution.
- `src/mlops_agents/models/factories.py` — **modify**. Add `arima_use_approximation(n_obs)`; `build_auto_arima` applies it (length-gated `approximation`).
- `src/mlops_agents/training/validation_policy.py` — **modify**. `resolve_rolling_window_size` becomes season-aware; `resolve_validation_strategy` uses a bounded rolling window for long series.
- Tests: `tests/test_training/test_seasonality.py`, `tests/test_training/test_executor_forecasting.py`, `tests/test_models/test_factories.py`, `tests/test_training/test_validation_policy.py` — **modify**.

---

### Task 1: Per-model seasonal grid policy

**Files:**
- Modify: `src/mlops_agents/forecasting/seasonality.py`
- Test: `tests/test_training/test_seasonality.py`

**Interfaces:**
- Consumes: existing `normalize_frequency(freq) -> str | None`.
- Produces:
  - `season_length_grid(model_key: str, freq, n_obs: int) -> list[int] | None` — candidate periods for the model family at this frequency, pruned to `m == 1 or n_obs >= 2*m`; always contains `1`; `None` for unknown frequency.
  - `max_season_length(freq) -> int | None` — largest period any model could request at this frequency (max of the "rich" grid), for window sizing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training/test_seasonality.py`:

```python
from mlops_agents.forecasting.seasonality import max_season_length, season_length_grid


def test_season_length_grid_tiers_by_model_family():
    # daily, ample history: seasonal_naive rich, ets modest, auto_arima tight
    assert season_length_grid("seasonal_naive", "D", 10000) == [1, 7, 30]
    assert season_length_grid("ets", "D", 10000) == [1, 7]
    assert season_length_grid("auto_arima", "D", 10000) == [1, 7]


def test_season_length_grid_weekly_differs_by_tier():
    assert season_length_grid("seasonal_naive", "W", 10000) == [1, 4, 13, 52]
    assert season_length_grid("ets", "W", 10000) == [1, 13, 52]
    assert season_length_grid("auto_arima", "W", 10000) == [1, 52]


def test_season_length_grid_prunes_periods_without_two_cycles():
    # 60 weekly obs: 52 needs >=104 -> dropped; 13 needs >=26 -> kept; 4 kept
    assert season_length_grid("seasonal_naive", "W", 60) == [1, 4, 13]


def test_season_length_grid_always_keeps_nonseasonal_floor():
    # tiny series: every seasonal period pruned, 1 survives
    assert season_length_grid("auto_arima", "W", 10) == [1]


def test_season_length_grid_unknown_freq_returns_none():
    assert season_length_grid("ets", None, 1000) is None
    assert season_length_grid("ets", "weird", 1000) is None


def test_season_length_grid_handles_pandas_aliases():
    assert season_length_grid("auto_arima", "W-MON", 10000) == [1, 52]
    assert season_length_grid("ets", "h", 10000) == [1, 24]


def test_season_length_grid_unknown_model_defaults_to_tight():
    assert season_length_grid("some_future_model", "D", 10000) == [1, 7]


def test_max_season_length_uses_largest_candidate_period():
    assert max_season_length("h") == 168   # hourly rich grid max
    assert max_season_length("D") == 30
    assert max_season_length("W") == 52
    assert max_season_length(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_seasonality.py -q`
Expected: FAIL with `ImportError: cannot import name 'season_length_grid'`.

- [ ] **Step 3: Implement the grid policy**

Append to `src/mlops_agents/forecasting/seasonality.py` (after `default_season_length`):

```python
# Candidate seasonal periods per model-family "tier", keyed by base frequency unit.
# seasonal_naive fits are ~free -> richest grid; ETS is moderate -> skip huge m;
# AutoARIMA is the most expensive per fit -> tightest grid. 1 = non-seasonal floor.
_SEASON_GRID_BY_TIER: dict[str, dict[str, list[int]]] = {
    "rich":   {"H": [1, 24, 168], "D": [1, 7, 30], "W": [1, 4, 13, 52], "M": [1, 3, 12], "Q": [1, 4], "Y": [1]},
    "modest": {"H": [1, 24],      "D": [1, 7],     "W": [1, 13, 52],    "M": [1, 3, 12], "Q": [1, 4], "Y": [1]},
    "tight":  {"H": [1, 24],      "D": [1, 7],     "W": [1, 52],        "M": [1, 12],    "Q": [1, 4], "Y": [1]},
}

_MODEL_GRID_TIER: dict[str, str] = {
    "seasonal_naive": "rich",
    "ets": "modest",
    "auto_arima": "tight",
}

_MIN_CYCLES = 2  # need >= 2 full seasonal cycles to estimate a period


def season_length_grid(model_key: str, freq: Any | None, n_obs: int) -> list[int] | None:
    """Candidate seasonal periods for ``model_key`` at frequency ``freq``.

    The grid is chosen by model family (cost), then pruned to periods estimable
    from ``n_obs`` observations (``m == 1`` or ``n_obs >= 2*m``). The non-seasonal
    period 1 is always retained as a floor. Returns ``None`` for unknown
    frequencies so the caller keeps the model's original search space.
    """
    unit = normalize_frequency(freq)
    if unit is None:
        return None
    tier = _MODEL_GRID_TIER.get(model_key, "tight")
    grid = _SEASON_GRID_BY_TIER[tier].get(unit)
    if grid is None:
        return None
    pruned = [m for m in grid if m == 1 or n_obs >= _MIN_CYCLES * m]
    return pruned or [1]


def max_season_length(freq: Any | None) -> int | None:
    """Largest seasonal period any model might request at this frequency.

    Used to size the validation window so it can support every candidate's
    seasonality. Returns ``None`` for unknown frequencies.
    """
    unit = normalize_frequency(freq)
    grid = _SEASON_GRID_BY_TIER["rich"].get(unit) if unit else None
    return max(grid) if grid else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_seasonality.py -q`
Expected: PASS (all, including the pre-existing normalize/canonical tests).

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/forecasting/seasonality.py tests/test_training/test_seasonality.py
git commit -m "feat(forecasting): per-model frequency-aware season_length grids with 2-cycle guard"
```

---

### Task 2: Apply the grid in the executor's seasonality narrowing

**Files:**
- Modify: `src/mlops_agents/training/executor.py` (`_narrow_seasonality_to_freq` at line ~57; its call site at line ~690; the import at line ~28)
- Test: `tests/test_training/test_executor_forecasting.py`

**Interfaces:**
- Consumes: `season_length_grid(model_key, freq, n_obs)` from Task 1.
- Produces:
  - `_narrow_seasonality_to_freq(spec: SearchSpaceSpec, freq, model_key: str, n_obs: int) -> SearchSpaceSpec` — replaces the categorical `season_length` choices with the model's pruned grid; returns `spec` unchanged for unknown freq / non-categorical / no `season_length`.
  - `_min_fold_train_len(vs: ValidationStrategy, train_pool_len: int) -> int` — the smallest training set any validation fold will see (rolling → `window_size`; expanding → first-fold size; single_split → full pool). Used as the `n_obs` passed into season pruning, so a capped rolling window doesn't keep a period the actual folds can't estimate.

- [ ] **Step 1: Replace the `_narrow_seasonality_to_freq` tests**

In `tests/test_training/test_executor_forecasting.py`, delete the existing block of `test_narrow_seasonality_*` tests (the ones asserting single canonical values like `[7]`/`[52]` and the alias variants) and replace with:

```python
def test_narrow_seasonality_applies_model_grid_and_does_not_mutate():
    sp = get_model("auto_arima").search_space
    out = _narrow_seasonality_to_freq(sp, "D", "auto_arima", 10000)
    assert _season_choices(out) == [1, 7]
    assert _season_choices(sp) == [4, 7, 12, 24, 52]  # original registry spec untouched


def test_narrow_seasonality_seasonal_naive_gets_rich_grid():
    sp = get_model("seasonal_naive").search_space
    out = _narrow_seasonality_to_freq(sp, "W", "seasonal_naive", 10000)
    assert _season_choices(out) == [1, 4, 13, 52]


def test_narrow_seasonality_unknown_freq_keeps_full_grid():
    sp = get_model("ets").search_space
    out = _narrow_seasonality_to_freq(sp, None, "ets", 1000)
    assert _season_choices(out) == [4, 7, 12, 24, 52]


def test_narrow_seasonality_frequency_policy_wins_over_override():
    overridden = narrow_search_space("auto_arima", {"season_length": SearchParamOverride(choices=[52])})
    assert _season_choices(overridden) == [52]
    # daily data: frequency grid replaces the (wrong) override
    assert _season_choices(_narrow_seasonality_to_freq(overridden, "D", "auto_arima", 10000)) == [1, 7]


def test_narrow_seasonality_noop_when_no_season_length_param():
    sp = get_model("random_forest_forecaster").search_space
    assert _narrow_seasonality_to_freq(sp, "D", "random_forest_forecaster", 10000) is sp


def test_min_fold_train_len_rolling_uses_window():
    vs = ValidationStrategy(type="rolling_window", n_folds=5, horizon=8, step_size=8, window_size=70)
    assert _min_fold_train_len(vs, 1000) == 70


def test_min_fold_train_len_expanding_uses_first_fold():
    # first fold trains on train_pool_len - (k-1)*horizon
    vs = ValidationStrategy(type="expanding_window", n_folds=5, horizon=8, step_size=8)
    assert _min_fold_train_len(vs, 200) == 200 - 4 * 8  # 168


def test_min_fold_train_len_single_split_uses_full_pool():
    vs = ValidationStrategy(type="single_split", n_folds=1, horizon=8)
    assert _min_fold_train_len(vs, 200) == 200


def test_narrow_seasonality_prunes_season_unestimable_in_rolling_fold():
    # full pool (104 weekly rows) would keep 52, but each rolling fold sees only 70
    # rows -> 52 (needs >=104) must be dropped; 13 (>=26) and 4 kept.
    sp = get_model("seasonal_naive").search_space
    vs = ValidationStrategy(type="rolling_window", n_folds=5, horizon=8, step_size=8, window_size=70)
    prune_n = _min_fold_train_len(vs, 104)
    out = _narrow_seasonality_to_freq(sp, "W", "seasonal_naive", prune_n)
    assert _season_choices(out) == [1, 4, 13]
```

Add these imports near the top of `tests/test_training/test_executor_forecasting.py` (the file already imports `get_model` and `_narrow_seasonality_to_freq` lower down — add `ValidationStrategy` and `_min_fold_train_len`):

```python
from mlops_agents.contracts.training import ValidationStrategy  # add to existing contracts import
from mlops_agents.training.executor import _min_fold_train_len    # add beside _narrow_seasonality_to_freq import
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py -q -k "narrow_seasonality or min_fold_train_len"`
Expected: FAIL — current `_narrow_seasonality_to_freq` takes only `(spec, freq)` (4-arg calls raise `TypeError`); `_min_fold_train_len` does not exist (`ImportError`).

- [ ] **Step 3: Update the import**

In `src/mlops_agents/training/executor.py`, change the seasonality import (line ~28):

```python
from mlops_agents.forecasting.seasonality import max_season_length, season_length_grid
```

(Replaces the existing `from mlops_agents.forecasting.seasonality import canonical_season_length` — `canonical_season_length` is no longer used in this module.)

Also add `ValidationStrategy` to the existing `from mlops_agents.contracts.training import (...)` block (line ~21) — it types the `_min_fold_train_len` helper:

```python
from mlops_agents.contracts.training import (
    ForecastingSettings,
    TrainingPlan,
    TrainingPlanCandidate,
    TrainingResult,
    TrialBudget,
    ValidationStrategy,
)
```

- [ ] **Step 4: Rewrite `_narrow_seasonality_to_freq`**

Replace the function body (line ~57):

```python
def _narrow_seasonality_to_freq(
    spec: SearchSpaceSpec, freq: str | None, model_key: str, n_obs: int
) -> SearchSpaceSpec:
    """Replace a categorical season_length grid with the model's frequency-aware,
    length-pruned candidate periods (see seasonality.season_length_grid).

    Known frequencies are authoritative even after a planner/user search-space
    override. Unknown frequencies, non-categorical spaces, and models without a
    season_length parameter keep their original search space.
    """
    param = spec.params.get("season_length")
    grid = season_length_grid(model_key, freq, n_obs)
    if grid is None or param is None or param.type != "categorical":
        return spec
    if list(param.choices) == grid:
        return spec
    new_params = dict(spec.params)
    new_params["season_length"] = param.model_copy(update={"choices": grid})
    return spec.model_copy(update={"params": new_params})
```

Add the `_min_fold_train_len` helper directly below it:

```python
def _min_fold_train_len(vs: ValidationStrategy, train_pool_len: int) -> int:
    """Smallest training set any validation fold will use.

    Season pruning must use this, not the full pool: a capped rolling window (or the
    first expanding fold) can be far shorter than the pool, so a period the pool
    could nominally estimate may be unestimable in the folds that actually run.
    """
    if vs.type == "rolling_window" and vs.window_size:
        return vs.window_size
    if vs.type == "expanding_window":
        return train_pool_len - (vs.n_folds - 1) * vs.horizon
    return train_pool_len  # single_split
```

- [ ] **Step 5: Update the call site**

In `_run_candidate_forecasting` (line ~690) — `vs` is already defined (and its `window_size` resolved) earlier in the function, so prune against the effective fold length:

```python
    prune_n = _min_fold_train_len(vs, len(pool))
    narrowed = _narrow_seasonality_to_freq(narrowed, freq, candidate.model_key, prune_n)
```

- [ ] **Step 6: Run the narrowing + helper tests to verify they pass**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py -q -k "narrow_seasonality or min_fold_train_len"`
Expected: PASS.

- [ ] **Step 7: Run the full forecasting executor suite (no regressions)**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py tests/test_training/test_executor_forecasting_leakage.py -q -m "not integration"`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_forecasting.py
git commit -m "feat(forecasting): narrow season_length to per-model frequency grid in executor"
```

---

### Task 3: Length-gated AutoARIMA `approximation`

**Files:**
- Modify: `src/mlops_agents/models/factories.py` (add `arima_use_approximation` + constant; `build_auto_arima` line ~116)
- Modify: `src/mlops_agents/training/executor.py` (inject `series_length` into `task_metadata` in `_run_candidate_forecasting`)
- Test: `tests/test_models/test_factories.py`

**Interfaces:**
- Produces:
  - `arima_use_approximation(n_obs: int) -> bool` — `True` when `n_obs > 500` (full exact ARIMA order search is too slow *and* CSS selection ≈ exact MLE selection at that length).
  - `build_auto_arima(spec)` sets `AutoARIMA(approximation=...)` from `arima_use_approximation(spec["task_metadata"]["series_length"])`; absent `series_length` ⇒ exact (`False`).
- Consumes (executor): injects `series_length` = the **actual training rows of each fit** into the `task_metadata` passed to the factory — `len(cand_train)` per validation fold and `len(train_pool)` at champion retrain — so the policy reflects the real per-fit cost (bounded windows stay exact; long fits use CSS).

**Rationale:** `approximation=True` only approximates the *order search*; the chosen order is still fit with full MLE. On long series this is ~50× faster (measured: m=24, n=500 → 45.2s exact vs 0.8s approx) and the CSS-vs-MLE order difference is smallest. On short/medium series exact search is cheap (n=240 → 4.9s) and worth keeping for accuracy. The measured cost cliff is between n≈240 and n≈500, so the 500 threshold is a tunable lower bound — lower it toward ~300 if a ~400–500-row dataset ever appears.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models/test_factories.py`:

```python
def test_arima_use_approximation_threshold():
    from mlops_agents.models.factories import arima_use_approximation
    assert arima_use_approximation(10000) is True
    assert arima_use_approximation(501) is True
    assert arima_use_approximation(500) is False
    assert arima_use_approximation(120) is False


def test_build_auto_arima_approximation_is_length_conditional():
    from mlops_agents.models.factories import build_auto_arima
    long = build_auto_arima(
        {"task_metadata": {"frequency": "h", "series_length": 10000}, "params": {"season_length": 24}}
    )
    assert long.models[0].approximation is True
    short = build_auto_arima(
        {"task_metadata": {"frequency": "MS", "series_length": 120}, "params": {"season_length": 12}}
    )
    assert short.models[0].approximation is False


def test_build_auto_arima_defaults_to_exact_when_length_unknown():
    from mlops_agents.models.factories import build_auto_arima
    sf = build_auto_arima({"task_metadata": {"frequency": "D"}, "params": {"season_length": 7}})
    assert sf.models[0].approximation is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models/test_factories.py -q -k "approximation or auto_arima"`
Expected: FAIL — `arima_use_approximation` does not exist; `build_auto_arima` ignores `series_length`.

- [ ] **Step 3: Implement the policy + factory**

In `src/mlops_agents/models/factories.py`, add near the top (after the imports, before the tabular helpers):

```python
_ARIMA_APPROX_MIN_OBS = 500  # above this, exact ARIMA order search is too slow -> use CSS approximation


def arima_use_approximation(n_obs: int) -> bool:
    """Use AutoARIMA's fast CSS order search only when the series is long enough that
    exact search is too slow AND CSS order selection ~= exact-MLE selection. Short and
    medium series keep the exact search, where it is cheap and more accurate.
    """
    return n_obs > _ARIMA_APPROX_MIN_OBS
```

Replace `build_auto_arima` (line ~116):

```python
def build_auto_arima(spec: dict[str, Any]):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    task_metadata = spec["task_metadata"]
    freq = task_metadata["frequency"]
    season_length = spec["params"].get("season_length", default_season_length(freq))
    # Length-gated CSS approximation for the order search; final order is full MLE.
    # series_length is injected by the executor; absent (ad-hoc callers) -> exact.
    approximation = arima_use_approximation(task_metadata.get("series_length", 0))
    return StatsForecast(
        models=[AutoARIMA(season_length=season_length, approximation=approximation)],
        freq=freq, n_jobs=1,
    )
```

- [ ] **Step 4: Run factory tests to verify they pass**

Run: `uv run pytest tests/test_models/test_factories.py -q -k "approximation or auto_arima"`
Expected: PASS.

- [ ] **Step 5: Inject `series_length` at every AutoARIMA fit (per fit, not once)**

The approximation gate must see the **actual rows this fit trains on**, not the full pool — a bounded rolling window may fit only ~336 rows while the pool has 10k, and gating on 10k would needlessly approximate that small, accuracy-sensitive fit. There are two fit sites.

(a) Validation folds — in `_run_candidate_forecasting`'s `fit_score`, the statsforecast branch (`src/mlops_agents/training/executor.py` line ~612-614). Replace:

```python
            if is_stat:
                # statsforecast path: ignores exog (existing behavior)
                sf = factory({"task_metadata": task_metadata, "params": params})
                sf.fit(_to_sf_format(cand_train, target, dt_col, sid_cols))
```

with (gate on the fold's own training size):

```python
            if is_stat:
                # statsforecast path: ignores exog (existing behavior). series_length is
                # the fold's actual training size, so the AutoARIMA approximation gate
                # reflects this fit (small bounded windows -> exact; long fits -> CSS).
                fit_metadata = {**task_metadata, "series_length": len(cand_train)}
                sf = factory({"task_metadata": fit_metadata, "params": params})
                sf.fit(_to_sf_format(cand_train, target, dt_col, sid_cols))
```

(b) Champion retrain — in `_retrain_forecasting` (line ~766-780). The champion is retrained on the **full** train pool, so its fit size is `len(train_pool)`. Add `fit_metadata` once after the pool copy (line ~771) and use it in both factory calls:

```python
    train_pool = train_pool.copy()
    train_pool[dt_col] = pd.to_datetime(train_pool[dt_col])
    fit_metadata = {**task_metadata, "series_length": len(train_pool)}
    path = models_dir / f"champion_{champion['model_key']}.pkl"
    if _is_statsforecast_model(champion["model_key"]):
        sf = factory({"task_metadata": fit_metadata, "params": champion["best_params"]})
        sf.fit(_to_sf_format(train_pool, target, dt_col, sid_cols))
        with path.open("wb") as f:
            pickle.dump(sf, f)
        return path

    forecaster = factory({"task_metadata": fit_metadata, "params": champion["best_params"]})
```

(An auto_arima champion on a long series retrains on the full pool, so this correctly enables approximation there — without it, retrain would default to exact and blow up on 10k rows.)

- [ ] **Step 6: Run the forecasting executor suite (no regressions)**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py tests/test_training/test_executor_forecasting_leakage.py tests/test_models/test_factories.py -q -m "not integration"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/models/factories.py src/mlops_agents/training/executor.py tests/test_models/test_factories.py
git commit -m "perf(forecasting): length-gated AutoARIMA approximation (exact on short/medium, CSS on long)"
```

> **Related, out of scope:** `training/exog_extender.py::_fit_auto_arima` builds its own `AutoARIMA` for the `auto_arima` exog-extension strategy and has the same per-fit cost on long series. It's rarely selected (`resolve_exog_strategies` favours `naive_carry`/`ets`), so it's left for a follow-up; the same `arima_use_approximation` policy would apply there.

---

### Task 4: Bounded rolling-window validation for long series

**Files:**
- Modify: `src/mlops_agents/training/validation_policy.py` (`resolve_rolling_window_size` line ~52; `resolve_validation_strategy` line ~26; module constants line ~18)
- Modify: `src/mlops_agents/training/executor.py` (rolling-window resolution at line ~555; add `max_season_length` import — already added in Task 2)
- Test: `tests/test_training/test_validation_policy.py`

**Interfaces:**
- Consumes: `max_season_length(freq)` from Task 1.
- Produces:
  - `resolve_rolling_window_size(total_history, horizon, n_folds, season_length)` — window covers `>= 2*season_length` when a season is given.
  - `resolve_validation_strategy(task_metadata, n_obs)` — returns `rolling_window` with a bounded `window_size` when `train_pool_len > _LARGE_SERIES_THRESHOLD`; otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training/test_validation_policy.py`:

```python
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    resolve_validation_strategy,
)


def test_resolve_rolling_window_size_is_season_aware():
    # window must cover >= 2 seasonal cycles (2*168 = 336 dominates 3*horizon and floor)
    assert resolve_rolling_window_size(10000, 24, 5, 168) == 336


def test_long_series_uses_bounded_rolling_window():
    vs = resolve_validation_strategy({"forecast_horizon": 24, "frequency": "h"}, n_obs=10000)
    assert vs.type == "rolling_window"
    assert vs.window_size == 336          # 2 * hourly max season (168)
    assert vs.window_size < 1000          # bounded, not ~10k


def test_medium_series_keeps_expanding_window():
    vs = resolve_validation_strategy({"forecast_horizon": 14, "frequency": "D"}, n_obs=1000)
    assert vs.type == "expanding_window"
    assert vs.window_size is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_validation_policy.py -q -k "season_aware or bounded_rolling or expanding_window"`
Expected: FAIL — `resolve_rolling_window_size(...)` ignores `season_length` (returns 72 not 336); long series currently returns `expanding_window`.

- [ ] **Step 3: Make `resolve_rolling_window_size` season-aware + add the threshold constant**

In `src/mlops_agents/training/validation_policy.py`, add the import and constant near the top (after line ~16):

```python
from mlops_agents.forecasting.seasonality import max_season_length
```

Add to the constants block (after line ~23):

```python
_LARGE_SERIES_THRESHOLD = 2000  # train-pool rows above which we cap the window
```

Replace `resolve_rolling_window_size` (line ~52):

```python
def resolve_rolling_window_size(
    total_history: int,
    horizon: int,
    n_folds: int,
    season_length: int | None,
) -> int:
    """Bounded training window for rolling-window validation.

    Sized to the larger of 3*horizon, 2 seasonal cycles, and a floor — so seasonal
    models keep enough history to estimate their period while each fit stays cheap
    on long series. Capped so the folds still fit in the available history.
    """
    base = max(_HORIZON_MULTIPLIER * horizon, _MIN_CYCLES_WINDOW * (season_length or 0), _WINDOW_SIZE_FLOOR)
    upper = total_history - n_folds * horizon
    return min(base, max(upper, horizon))
```

Add the `_MIN_CYCLES_WINDOW` constant next to the others (after line ~23):

```python
_MIN_CYCLES_WINDOW = 2  # rolling window must hold >= 2 seasonal cycles
```

- [ ] **Step 4: Switch long series to a bounded rolling window in `resolve_validation_strategy`**

Replace the tail of `resolve_validation_strategy` (lines ~43-49) with:

```python
    k = min(k_max, _MAX_FOLDS)
    if k == 1:
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)
    freq = task_metadata.get("frequency")
    if drift == "high":
        # high drift: rolling window, size resolved downstream by the executor
        return ValidationStrategy(
            type="rolling_window", n_folds=k, horizon=horizon, step_size=horizon
        )
    if train_pool_len > _LARGE_SERIES_THRESHOLD:
        # use train_pool_len (not n_obs): folds are carved from the pool AFTER the
        # final test horizon is removed, matching validate_forecasting_plan's `upper`.
        window = resolve_rolling_window_size(train_pool_len, horizon, k, max_season_length(freq))
        return ValidationStrategy(
            type="rolling_window", n_folds=k, horizon=horizon,
            step_size=horizon, window_size=window,
        )
    return ValidationStrategy(
        type="expanding_window", n_folds=k, horizon=horizon, step_size=horizon
    )
```

- [ ] **Step 5: Pass the season into the executor's window resolution**

In `src/mlops_agents/training/executor.py`, the rolling-window resolution block (line ~555) currently calls `resolve_rolling_window_size(len(pool), horizon, vs.n_folds, season_length=None)`. Change the last arg:

```python
                "window_size": resolve_rolling_window_size(
                    len(pool), horizon, vs.n_folds, season_length=max_season_length(freq),
                )
```

(`max_season_length` is already imported in this module from Task 2; `freq` is defined at line ~551.)

- [ ] **Step 6: Run the validation-policy tests to verify they pass**

Run: `uv run pytest tests/test_training/test_validation_policy.py -q -m "not integration"`
Expected: PASS. If a pre-existing test asserted a non-season-aware rolling window value, update it to the new season-aware value (the only intended behavior change).

- [ ] **Step 7: Run the full training suite (no regressions)**

Run: `uv run pytest tests/test_training tests/test_models tests/test_contracts -q -m "not integration"`
Expected: PASS.

- [ ] **Step 8: End-to-end smoke check on the stress dataset**

Run (PowerShell — this session's primary shell; a `bash` heredoc would fail here):

```powershell
@'
import json, time, warnings, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from mlops_agents.contracts.training import ForecastingSettings, TrainingPlanCandidate
from mlops_agents.training.executor import _run_candidate_forecasting
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.validation_policy import resolve_validation_strategy
from mlops_agents.training.exog_policy import resolve_exog_strategies
s = json.load(open("data/samples/size_test/large_hourly_factory_schema.json", encoding="utf-8"))
p = Path("data/samples/size_test/large_hourly_factory.csv"); df = pd.read_csv(p)
md = {"name": s["name"], "problem_type": "forecasting", "target_column": s["target_column"],
      "datetime_column": s["datetime_column"], "forecast_horizon": s["forecast_horizon"],
      "frequency": s["frequency"], "series_id_columns": [], "exogenous_columns": s.get("exogenous_columns", [])}
prof = build_dataset_profile(p, md)
fs = ForecastingSettings(validation_strategy=resolve_validation_strategy(md, len(df)),
                         exog_strategies=resolve_exog_strategies(df, md, md["frequency"]))
print("validation:", fs.validation_strategy.model_dump())
for mk in ["seasonal_naive", "ets", "auto_arima"]:
    c = TrainingPlanCandidate(priority=1, model_key=mk, requested_trials=5, reason="x")
    t0 = time.perf_counter()
    r = _run_candidate_forecasting(c, df.copy(), md, 5, "rmse", "minimize", fs, prof)
    print(f"  {mk:16s} {time.perf_counter()-t0:6.1f}s  trials={r['n_trials_used']}  {r['best_params']}")
'@ | uv run python -
```

(If executing through the `Bash` tool instead of the PowerShell terminal, write the snippet to a temp `.py` file and `uv run python` it — avoid heredoc/here-string fragility.)

Expected: `validation` shows `rolling_window` with `window_size: 336`; each candidate finishes in single-digit-to-low-tens of seconds (no multi-minute auto_arima/ets). auto_arima on the 336-row bounded window uses **exact** search (`len(cand_train)=336 < 500`); if that proves too slow here, lower `_ARIMA_APPROX_MIN_OBS` toward ~300. This is a manual check, not committed.

- [ ] **Step 9: Commit**

```bash
git add src/mlops_agents/training/validation_policy.py src/mlops_agents/training/executor.py tests/test_training/test_validation_policy.py
git commit -m "perf(forecasting): bounded season-aware rolling window for long-series validation"
```

---

## Out of scope (separate follow-up plan)

**Two-stage hyperparameter budget (successive-halving across candidates).** This is an orthogonal *trial-budget* optimization that mainly benefits the ML lag forecasters (RF/XGB/LGBM/SVR), whose cost scales with trial count — not the statistical models addressed here (now ~1 config, ~1s/fit). It lives in `trial_budget.allocate_trials` + the executor candidate loop and warrants its own plan with its own screening/finalist-margin design. After this plan lands, statistical fits are tractable; revisit two-stage only if ML-forecaster trial counts become the bottleneck.

## Benchmark note (after implementation)

These changes alter forecasting outputs (richer seasonal search, length-gated `approximation`-based ARIMA order selection on long series, bounded windows on long series). Short/medium datasets keep exact ARIMA search, so their AutoARIMA results change only via the seasonal-grid widening, not approximation. The committed seeded experience pool predates these changes. To keep the reproducibility fixture consistent with the code, **re-run the forecasting subset of the benchmark** (classification/regression records are unaffected). Add a one-line note to the commit/thesis that statistical forecasters use a frequency-aware seasonal grid, length-gated `approximation` for ARIMA order selection, and bounded-window validation on long series.

## Self-Review

**Spec coverage:**
- Richer, frequency-aware, family-differentiated grids → Task 1 (`season_length_grid`) + Task 2 (wired into executor).
- Sub-annual seasonality captured (e.g. weekly `[1,4,13,52]`, monthly `[1,3,12]`) → Task 1 grids.
- ≥2-cycle estimability guard → Task 1 (`_MIN_CYCLES`) + tested.
- Per-fit cost of AutoARIMA → Task 3 (length-gated `approximation`: CSS on long series, exact on short/medium).
- Per-fit cost of AutoETS (no approximation knob) → Task 4 (bounded window).
- Long-series tractability without dropping candidates → Task 4 (rolling window sized to `2*max_season`).
- "Frequency wins over override" preserved → Task 2 test.
- Champion still selected on current-dataset validation → unchanged; no task touches `_pick_champion`.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code; every test step shows assertions.

**Type consistency:** `season_length_grid(model_key, freq, n_obs) -> list[int] | None` and `max_season_length(freq) -> int | None` are used with those exact signatures in Tasks 2 and 4. `_narrow_seasonality_to_freq(spec, freq, model_key, n_obs)` 4-arg form is defined in Task 2 and called with `prune_n` (an int) at the call site. `_min_fold_train_len(vs, train_pool_len) -> int` is defined in Task 2 and produces that `prune_n`; it reads `vs.type`, `vs.window_size`, `vs.n_folds`, `vs.horizon` (all present on `ValidationStrategy`, which is added to the executor's contracts import). `resolve_rolling_window_size(total_history, horizon, n_folds, season_length)` keeps its parameter names; only the body and the *caller's first argument* (`train_pool_len`) change. Window value `336 = max(3*24, 2*168, 50)` is consistent across Task 4's tests and the smoke check, and is unaffected by the `n_obs → train_pool_len` change here (`upper` does not bind: `9976 − 120 = 9856 > 336`).

**Review feedback folded in:**
- *Season pruning* uses `_min_fold_train_len`, so a rolling window capped below `2·max_season` on a short high-drift series correctly drops the unestimable period (covered by `test_narrow_seasonality_prunes_season_unestimable_in_rolling_fold`); expanding windows prune against the first (smallest) fold.
- *Window sizing* in `resolve_validation_strategy` uses `train_pool_len` (not `n_obs`), matching `validate_forecasting_plan`'s `upper` so `resolve` and `validate` can never disagree by a horizon.
- *Approximation gate* keys off the **actual per-fit training size**, injected as `series_length` at both fit sites (`len(cand_train)` per validation fold in `fit_score`; `len(train_pool)` at champion retrain). A bounded window therefore stays on exact search (336 < 500) while a long unwindowed/retrain fit uses CSS — the gate reflects the fit that actually runs, not the pool. `build_auto_arima` reads `task_metadata["series_length"]` (default 0 → exact) — consistent with the injected key name at both sites.
