# Pipeline UI Rich Panels — Design Spec

**Date:** 2026-04-17
**Scope:** `dashboard/pages/01_pipeline.py` + `dashboard/pipeline_helpers.py`
**Goal:** Upgrade the pipeline page to a two-column split layout with live data panels that fill in progressively as each agent completes.

---

## Context

The current pipeline page shows only a streaming log. The thesis evaluator's primary focus is the live pipeline — watching it execute in real time. The upgrade adds a right-column panel area with three tabs (Data, Training, Evaluation) that light up progressively as each agent finishes. The HITL approval moment stays as the full-width centrepiece.

Existing spec: `docs/superpowers/specs/2026-04-16-pipeline-ui-design.md` (three-phase state machine, already implemented). This spec extends that work — it does not replace it.

---

## Layout

Once a run starts, the page splits into two columns for the `idle → awaiting_approval` and `complete` phases:

- **Left column (40%):** streaming log, unchanged from current implementation
- **Right column (60%):** `st.tabs(["📊 Data", "🏋️ Training", "📈 Evaluation"])` — rendered once, updated in-place as agents complete

When `phase == "awaiting_approval"`, columns collapse to full width to give the HITL panel maximum visual presence.

In `phase == "complete"`, columns return — the outcome banner appears full-width at the bottom, tabs show their final data.

---

## Session State — New Fields

Three new fields added to `_DEFAULTS`:

| Key | Type | Set after |
|-----|------|-----------|
| `validation_report` | `dict` | `data_validator` completes |
| `training_metrics` | `dict` | `trainer` completes |
| `evaluation_report` | `dict` | `evaluator` completes |
| `dataset_preview` | `list[dict]` | `data_validator` completes (10-row sample) |

All default to empty (`{}` / `[]`). Cleared by "Run Again" via `st.session_state.clear()`.

---

## Tab Content

### 📊 Data tab
Shown after `data_validator` completes (`validation_report` non-empty).

- Metric row: row count, column count, missing values %, pass/fail badge
- `st.dataframe` of first 10 rows of the dataset (loaded once, cached in `dataset_preview`)
- Evidently summary: key fields from `validation_report` rendered as `st.dataframe` or `st.json` expander

Empty state: `st.info("Waiting for data validation to complete...")`

### 🏋️ Training tab
Shown after `trainer` completes (`training_metrics` non-empty).

- Metric row: model type, best metric value (accuracy or F1), MLflow run ID
- `st.dataframe` of `training_metrics` dict (accuracy, precision, recall, F1)
- MLflow run ID displayed as plain text (no live link required)

Empty state: `st.info("Waiting for model training to complete...")`

### 📈 Evaluation tab
Shown after `evaluator` completes (`evaluation_report` non-empty).

- Metric row: candidate vs. baseline comparison, recommendation (promote / reject / retrain)
- `st.dataframe` comparing candidate and champion metrics side by side
- Agent natural-language summary: last message from evaluator rendered as `st.markdown`

Empty state: `st.info("Waiting for model evaluation to complete...")`

---

## Data Flow

Inside the streaming loop (and after resume), after each event the code reads current graph state and updates panel session fields:

```python
state = graph.get_state(config).values
panel = extract_panel_data(state)
for key, val in panel.items():
    if val:
        st.session_state[key] = val
```

`extract_panel_data(state: dict) -> dict` is a new pure function in `pipeline_helpers.py`. It returns a dict with keys `validation_report`, `training_metrics`, `evaluation_report`, `dataset_preview` — pulling from the raw LangGraph state dict. If a field is absent or empty, the returned value is `{}` / `[]` (falsy), so the caller's `if val` guard skips it.

`dataset_preview` is populated by calling `pd.read_csv(state["dataset_path"]).head(10).to_dict("records")` — only when `validation_report` is newly non-empty.

---

## Error Handling

- **Pipeline fails early:** Tabs with no data show their "Waiting..." placeholder. No crashes.
- **Missing keys in report dicts:** All tab rendering uses `.get()` with safe defaults; falls back to `st.json(report)` if expected keys are absent.
- **Empty `dataset_path`:** `extract_panel_data` returns empty `dataset_preview` if path is falsy or file doesn't exist.

---

## Helper Function Contract

```python
# pipeline_helpers.py additions

def extract_panel_data(state: dict) -> dict:
    """Extract displayable panel data from a raw LangGraph state dict.

    Returns a dict with keys:
      validation_report, training_metrics, evaluation_report, dataset_preview
    Values are empty ({} / []) if the corresponding stage hasn't completed.
    """
```

This function is pure (no Streamlit, no I/O except pd.read_csv gated on path existence) and fully unit-testable.

---

## Out of Scope

- Progress bar during Optuna hyperparameter tuning (no per-trial LangGraph events)
- Evidently HTML report embedded in iframe
- Historical run comparison (belongs in Experiments page)
- Multiple concurrent pipeline runs
- Persistent run history across browser sessions

---

## Files Changed

| Action | Path |
|--------|------|
| Modify | `dashboard/pages/01_pipeline.py` |
| Modify | `dashboard/pipeline_helpers.py` |
| Modify | `tests/test_dashboard/test_pipeline_helpers.py` |
