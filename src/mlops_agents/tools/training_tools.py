"""Model training and hyperparameter tuning tools.

The training loop (fit, cross-validation, metrics) is deterministic.
The training agent uses these tools and reasons about the results
(e.g., deciding whether to retune or accept the model).
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("./models")


@tool
def tune_hyperparameters(dataset_path: str, model_type: str, n_trials: int = 20) -> str:
    """Run Optuna hyperparameter search for the specified model type.

    Args:
        dataset_path: Path to training CSV (must have a 'target' column).
        model_type: One of 'random_forest', 'gradient_boosting', 'logistic_regression'.
        n_trials: Number of Optuna trials (default 20).

    Returns:
        JSON with best hyperparameters and cross-validation score.
    """
    try:
        import optuna
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        return json.dumps({"error": f"Missing dependency: {e}"})

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["target"]).select_dtypes(include="number")
    y = df["target"]

    def objective(trial: optuna.Trial) -> float:
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
            }
            model = GradientBoostingClassifier(**params, random_state=42)
        else:
            params = {"C": trial.suggest_float("C", 0.01, 100, log=True)}
            model = LogisticRegression(**params, random_state=42, max_iter=1000)

        scores = cross_val_score(model, X, y, cv=3, scoring="f1_weighted")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    result = {
        "model_type": model_type,
        "best_params": study.best_params,
        "best_cv_f1": round(study.best_value, 4),
        "n_trials": n_trials,
    }
    logger.info(f"Hyperparameter tuning complete: best F1={study.best_value:.4f}")
    return json.dumps(result)


@tool
def train_model(dataset_path: str, model_type: str, hyperparameters_json: str) -> str:
    """Train a model with the given hyperparameters and save it to disk.

    Args:
        dataset_path: Path to training CSV (must have a 'target' column).
        model_type: One of 'random_forest', 'gradient_boosting', 'logistic_regression'.
        hyperparameters_json: JSON string of hyperparameter dict.

    Returns:
        JSON with model path, train/val accuracy, and classification report.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        return json.dumps({"error": f"Missing dependency: {e}"})

    params = json.loads(hyperparameters_json)
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["target"]).select_dtypes(include="number")
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model_map = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }
    if model_type not in model_map:
        return json.dumps({"error": f"Unknown model_type: {model_type}"})

    model = model_map[model_type](**params, random_state=42)
    model.fit(X_train, y_train)

    train_acc = round(float(model.score(X_train, y_train)), 4)
    val_acc = round(float(model.score(X_val, y_val)), 4)
    report = classification_report(y_val, model.predict(X_val), output_dict=True)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = str(MODELS_DIR / f"{model_type}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    result = {
        "model_type": model_type,
        "model_path": model_path,
        "hyperparameters": params,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "classification_report": report,
    }
    logger.info(f"Training complete: {model_type}, val_acc={val_acc}")
    return json.dumps(result, default=str)
