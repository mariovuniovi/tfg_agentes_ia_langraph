"""Application configuration via Pydantic Settings (reads from .env)."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration is read from environment variables or .env file.

    Never hardcode tokens or URIs — always use this settings object.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — OpenAI API
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"   # default / fallback model

    # Per-agent model overrides
    openai_model_data_validator: str = "gpt-5-mini"
    openai_model_planner: str = "gpt-5-mini"
    openai_model_report_writer: str = "gpt-5-mini"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "mlops-agents"

    # Evidently
    evidently_workspace: str = "./evidently_workspace"

    # LangSmith (optional tracing)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "mlops-multi-agent"

    # Application
    log_level: str = "INFO"
    log_verbosity: int = 2
    data_dir: str = "./data/samples"
    dataset_schema: str = "iris_classification"
    max_attempts_per_agent: int = 3
    imputation_strategy_numeric: Literal["mean", "median", "zero"] = "mean"
    imputation_strategy_categorical: Literal["mode", "unknown", "drop_row"] = "mode"

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
    retrieval_weights_override: dict = Field(default_factory=dict)

    # Planner (SP5)
    planner_max_iterations: int = 10
    planner_max_tool_calls: int = 6
    planner_max_inspect_calls: int = 3
    planner_max_retrieved: int = 20
    planner_timeout_seconds: int = 60  # RESERVED for future wall-clock enforcement; NOT enforced in v2


settings = Settings()
