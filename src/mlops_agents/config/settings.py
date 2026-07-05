"""Application configuration via Pydantic Settings (reads from .env)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration is read from environment variables or .env file.

    Never hardcode tokens or URIs — always use this settings object.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — OpenAI API (or any OpenAI-compatible endpoint, e.g. GitHub Models)
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"   # fallback when no YAML config found
    # Point at an OpenAI-compatible endpoint (empty = OpenAI). Setting this
    # switches the factory to plain chat-completions (no Responses/reasoning API).
    openai_base_url: str = ""
    # Force every agent onto one model, ignoring the per-agent prompt YAML.
    # Needed for GitHub Models, whose model names (openai/gpt-4.1-mini) differ.
    openai_model_override: str = ""

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "mlops-agents"

    # Application
    log_level: str = "INFO"
    log_verbosity: int = 2
    data_dir: str = "./data/samples"
    dataset_schema: str = "iris_classification"
    max_attempts_per_agent: int = 3
    imputation_strategy_numeric: Literal["mean", "median", "zero"] = "mean"
    imputation_strategy_categorical: Literal["mode", "unknown", "drop_row"] = "mode"

    # Data Validator — Join Discovery
    data_validator_profile_nrows: int = 10
    data_validator_max_join_candidates: int = 20
    data_validator_row_explosion_medium_threshold: float = 1.25
    data_validator_row_explosion_high_threshold: float = 2.0
    data_validator_min_left_coverage: float = 0.8
    data_validator_min_containment: float = 0.8

    # Training executor (SP3)
    train_test_split_ratio: float = 0.2
    cv_folds: int = 5
    min_rows_for_cv: int = 50
    min_class_count_for_cv: int = 5
    optuna_total_trials: int = 60
    optuna_min_trials_per_candidate: int = 5
    optuna_max_trials_per_candidate: int = 30
    log_non_champion_models: bool = False
    tie_tolerance_relative: float = 0.01
    forecasting_min_train_points: int = 30
    experience_pool_dir: Path = Path("experience_pool")

    # Experience pool (SP4)
    experience_db_path: Path = Path("storage/mlops_metadata.db")
    experience_audit_dir: Path = Path("experience_pool")
    data_benchmarks_dir: Path = Path("data/benchmarks")
    ml_rules_path: Path = Path("src/mlops_agents/knowledge/ml_rules.yaml")
    retrieval_default_k: int = 5
    retrieval_weights_override: dict[str, Any] = Field(default_factory=dict)

    # Planner (SP5)
    planner_max_iterations: int = 10
    planner_max_tool_calls: int = 6
    planner_max_inspect_calls: int = 3
    planner_max_retrieved: int = 20
    planner_timeout_seconds: int = 60  # RESERVED for future wall-clock enforcement; NOT enforced in v2


settings = Settings()
