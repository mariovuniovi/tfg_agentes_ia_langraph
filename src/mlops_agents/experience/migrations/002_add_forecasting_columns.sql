-- Migration 002: add five forecasting/leakage-safe-validation columns to experiences
ALTER TABLE experiences ADD COLUMN validation_strategy_json TEXT;
ALTER TABLE experiences ADD COLUMN exog_availability_json TEXT;
ALTER TABLE experiences ADD COLUMN exog_strategies_json TEXT;
ALTER TABLE experiences ADD COLUMN per_fold_metrics_json TEXT;
ALTER TABLE experiences ADD COLUMN exog_fit_failures_json TEXT;

INSERT OR IGNORE INTO _schema_version (version, applied_at)
VALUES (2, datetime('now'));
