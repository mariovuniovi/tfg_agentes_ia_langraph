"""Application-wide constants."""

from __future__ import annotations

# MLflow model registry stages / aliases
MLFLOW_ALIAS_CHAMPION = "champion"
MLFLOW_ALIAS_CHALLENGER = "challenger"
MLFLOW_ALIAS_STAGING = "staging"
MLFLOW_TAG_VALIDATION_STATUS = "validation_status"
MLFLOW_REGISTERED_MODEL_NAME = "mlops-agent-model"

# Pipeline thresholds
MIN_ACCURACY_TO_DEPLOY = 0.80
MIN_F1_TO_DEPLOY = 0.75
MAX_DRIFT_SCORE = 0.10  # PSI threshold

# LangGraph limits
GRAPH_RECURSION_LIMIT = 30

# Agent names (must match node names in the graph)
AGENT_DATA_VALIDATOR = "data_validator"
