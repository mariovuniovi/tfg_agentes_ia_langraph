-- Migration 003: add expected_drift column to experiences
ALTER TABLE experiences ADD COLUMN expected_drift TEXT;

INSERT OR IGNORE INTO _schema_version (version, applied_at)
VALUES (3, datetime('now'));
