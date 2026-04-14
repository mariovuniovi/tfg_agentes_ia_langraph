"""Seed MLflow with sample experiment runs for dashboard demo purposes.

Usage:
    uv run python scripts/seed_mlflow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlops_agents.config.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

runs = [
    ("RandomForest-baseline", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("RandomForest-tuned", RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
]

for run_name, model in runs:
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Logged '{run_name}': accuracy={acc:.4f}, f1={f1:.4f}")

print(f"\nDone. View at: {settings.mlflow_tracking_uri}")
