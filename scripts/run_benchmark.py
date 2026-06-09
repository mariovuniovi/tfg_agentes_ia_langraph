"""Offline benchmark runner — seeds the experience pool from public datasets.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --manifest scripts/benchmark_manifest.yaml --trials 8
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*No further splits.*")
try:
    from skforecast.exceptions import IgnoredArgumentWarning, MissingValuesWarning
    warnings.filterwarnings("ignore", category=IgnoredArgumentWarning)
    warnings.filterwarnings("ignore", category=MissingValuesWarning)
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord
from mlops_agents.training.default_plans import default_training_plan
from mlops_agents.training.executor import run_training_plan
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.contracts.training import TrialBudget
from mlops_agents.utils.logging import get_logger
from scripts._dataset_sources import fetch_dataset

logger = get_logger(__name__)

_STATISTICAL_MODELS = frozenset({"naive", "seasonal_naive", "ets", "auto_arima"})
_SUPERVISED_MODELS = frozenset({
    "random_forest_forecaster", "extra_trees_forecaster", "gbm_forecaster",
    "lightgbm_forecaster", "xgboost_forecaster", "svr_forecaster",
})


def _check_family(model_key: str, expected_family: str) -> bool:
    """True if model_key belongs to expected_family."""
    if expected_family == "statistical":
        return model_key in _STATISTICAL_MODELS
    if expected_family == "random_walk":
        return model_key == "naive"
    if expected_family == "supervised":
        return model_key in _SUPERVISED_MODELS
    return False


def _preprocess_benchmark_df(df: "pd.DataFrame", entry: dict) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Label-encode categoricals; drop high-cardinality string columns."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    if entry["problem_type"] == "forecasting":
        return df

    target = entry["target_column"]
    n = len(df)

    # Drop columns that perfectly predict the target (OpenML often duplicates the target)
    try:
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        for col in list(df.columns):
            if col == target:
                continue
            try:
                col_numeric = pd.to_numeric(df[col], errors="coerce")
                if col_numeric.equals(target_numeric) or abs(col_numeric.corr(target_numeric)) >= 1.0:
                    df = df.drop(columns=[col])
            except Exception:
                pass
    except Exception:
        pass

    for col in list(df.columns):
        if col == target:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "category":
            n_unique = df[col].nunique()
            if n_unique > min(50, n * 0.5):
                df = df.drop(columns=[col])
            else:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        elif df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if df[target].dtype == object or str(df[target].dtype) == "category":
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))

    return df


def build_task_metadata(entry: dict) -> dict:
    meta = {"problem_type": entry["problem_type"], "target_column": entry["target_column"]}
    if entry["problem_type"] == "forecasting":
        meta.update({
            "datetime_column": entry["datetime_column"],
            "series_id_columns": entry.get("series_id_columns", []),
            "frequency": entry["frequency"],
            "forecast_horizon": entry["forecast_horizon"],
        })
        if "exogenous_columns" in entry:
            meta["exogenous_columns"] = entry["exogenous_columns"]
        if "expected_drift" in entry:
            meta["expected_drift"] = entry["expected_drift"]
    return meta


def stage_dataset(df: "pd.DataFrame", entry: dict, staged_dir: Path) -> Path:  # type: ignore[name-defined]
    staged_dir.mkdir(parents=True, exist_ok=True)
    target = entry["target_column"]
    if target not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": target})
    csv_path = staged_dir / f"{entry['dataset_id']}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def run_benchmark(
    manifest_path: Path = Path("scripts/benchmark_manifest.yaml"),
    db_path: Path | None = None,
    audit_dir: Path | None = None,
    splits_dir: Path | None = None,
    staged_dir: Path | None = None,
    n_trials_override: int | None = None,
    reset_forecasting: bool = False,
    strict: bool = False,
) -> tuple[int, int]:
    """Seed the experience pool. Returns (n_success, n_fail)."""
    db_path = db_path or settings.experience_db_path
    audit_dir = audit_dir or settings.experience_audit_dir
    splits_dir = splits_dir or Path("data/benchmarks/_splits")
    staged_dir = staged_dir or Path("data/benchmarks")

    manifest = yaml.safe_load(manifest_path.read_text()) or []
    pool = ExperiencePool(db_path, audit_dir=audit_dir)

    if reset_forecasting:
        n_del = pool.reset_forecasting_experiences()
        logger.info(f"[benchmark] reset_forecasting: removed {n_del} stale forecasting experiences")

    family_mismatches: list[dict[str, str]] = []
    n_success = n_fail = 0

    for entry in manifest:
        dataset_id = entry["dataset_id"]
        try:
            logger.info(f"[{dataset_id}] Fetching dataset...")
            df = fetch_dataset(entry)
            df = _preprocess_benchmark_df(df, entry)
            csv_path = stage_dataset(df, entry, staged_dir)

            task_meta = build_task_metadata(entry)
            profile = build_dataset_profile(csv_path, task_meta)
            plan = default_training_plan(entry["problem_type"], profile)

            skip_models: list[str] = entry.get("skip_models") or []
            if skip_models:
                filtered = [c for c in plan.candidates if c.model_key not in skip_models]
                plan = plan.model_copy(update={"candidates": filtered})

            if n_trials_override is not None:
                plan = plan.model_copy(update={
                    "trial_budget": TrialBudget(
                        total_trials=n_trials_override * len(plan.candidates),
                        allocation_strategy="equal",
                        min_trials_per_candidate=max(2, n_trials_override // 2),
                        max_trials_per_candidate=n_trials_override,
                    )
                })

            result = run_training_plan(
                plan=plan,
                processed_dataset_path=csv_path,
                target_column=entry["target_column"],
                task_metadata=task_meta,
                output_dir=splits_dir / dataset_id,
                mlflow_experiment="mlops-agents-benchmark",
            )

            record = ExperienceRecord.model_validate(
                json.loads(Path(result.experience_record_path).read_text())
            )
            pool.insert_from_record(record)
            champion_key = result.champion_candidate["model_key"]
            expected_family = entry.get("expected_family")
            if expected_family and entry.get("problem_type") == "forecasting":
                match = _check_family(champion_key, expected_family)
                symbol = "✓" if match else "✗"
                logger.info(
                    f"[{dataset_id}] family_check {symbol} "
                    f"champion={champion_key} expected_family={expected_family}"
                )
                if not match:
                    family_mismatches.append({
                        "dataset_id": dataset_id,
                        "champion": champion_key,
                        "expected_family": expected_family,
                    })
            n_success += 1
            logger.info(f"[{dataset_id}] champion={champion_key}")

        except Exception as e:
            n_fail += 1
            logger.error(f"[{dataset_id}] FAILED: {e}")

    if family_mismatches:
        logger.warning(f"[benchmark] {len(family_mismatches)} family mismatch(es):")
        for m in family_mismatches:
            logger.warning(
                f"  ✗ {m['dataset_id']}: champion={m['champion']} "
                f"expected={m['expected_family']}"
            )
        if strict:
            n_fail += len(family_mismatches)
    else:
        logger.info("[benchmark] All forecasting family checks passed ✓")

    logger.info(f"Benchmark complete: {n_success} success, {n_fail} failed")
    return n_success, n_fail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("scripts/benchmark_manifest.yaml"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--reset-forecasting", action="store_true",
                        help="Delete all forecasting pool experiences before re-seeding")
    parser.add_argument("--strict", action="store_true",
                        help="Count family mismatches as failures (non-zero exit)")
    args = parser.parse_args()
    n_ok, n_fail = run_benchmark(
        manifest_path=args.manifest,
        n_trials_override=args.trials,
        reset_forecasting=args.reset_forecasting,
        strict=args.strict,
    )
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
