CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    task_id TEXT PRIMARY KEY,
    problem_type TEXT NOT NULL,
    dataset_name TEXT,
    dataset_profile_json TEXT NOT NULL,
    training_plan_json TEXT NOT NULL,
    selected_model_key TEXT,
    metric_to_optimize TEXT,
    metric_direction TEXT,
    validation_score REAL,
    validation_std REAL,
    experience_summary TEXT,
    experience_json_path TEXT,
    mlflow_parent_run_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_problem_type ON experiences(problem_type);
CREATE INDEX IF NOT EXISTS idx_experiences_created_at ON experiences(created_at);

CREATE TABLE IF NOT EXISTS candidate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    status TEXT NOT NULL,
    best_params_json TEXT,
    best_score REAL,
    best_score_std REAL,
    n_trials_used INTEGER,
    duration_s REAL,
    complexity_rank INTEGER,
    mlflow_run_id TEXT,
    error_type TEXT,
    error_message TEXT,
    FOREIGN KEY (task_id) REFERENCES experiences(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_candidate_results_task_id ON candidate_results(task_id);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    mlflow_run_id TEXT,
    artifact_path TEXT,
    artifact_uri TEXT,
    model_uri TEXT,
    is_champion INTEGER NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES experiences(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_task_id ON model_artifacts(task_id);

INSERT OR IGNORE INTO _schema_version (version, applied_at)
VALUES (1, datetime('now'));
