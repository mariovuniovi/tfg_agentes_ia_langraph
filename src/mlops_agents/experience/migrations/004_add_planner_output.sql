-- Migration 004: add planner_output column to experiences
ALTER TABLE experiences ADD COLUMN planner_output_json TEXT;

INSERT OR IGNORE INTO _schema_version (version, applied_at)
VALUES (4, datetime('now'));
