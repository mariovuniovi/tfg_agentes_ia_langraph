"""Application configuration via Pydantic Settings (reads from .env)."""

from typing import Literal

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
    openai_model_supervisor: str = "gpt-5-mini"
    openai_model_data_validator: str = "gpt-5-mini"
    openai_model_trainer: str = "gpt-5-mini"
    openai_model_evaluator: str = "gpt-5-mini"
    openai_model_deployer: str = "gpt-5.4-nano"

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


settings = Settings()
