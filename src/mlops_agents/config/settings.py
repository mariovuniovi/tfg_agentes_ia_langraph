"""Application configuration via Pydantic Settings (reads from .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration is read from environment variables or .env file.

    Never hardcode tokens or URIs — always use this settings object.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — GitHub Models
    github_token: str = ""
    github_model: str = "openai/gpt-4.1-mini"   # matches your GITHUB_MODEL env var
    # Base URL is fixed for GitHub Models — no env var needed
    github_api_base: str = "https://models.github.ai/inference"

    # Fallback providers (optional — only needed if GitHub rate limit is hit)
    groq_api_key: str = ""
    gemini_api_key: str = ""

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


settings = Settings()
