"""Tests for SQLite migration runner."""
import sqlite3
from mlops_agents.experience.migrations._runner import apply_pending_migrations


def test_migrations_create_three_tables(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"experiences", "candidate_results", "model_artifacts", "_schema_version"}.issubset(tables)
    conn.close()


def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    apply_pending_migrations(db)  # second call must not raise
    conn = sqlite3.connect(db)
    version = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
    assert version == 2
    conn.close()


def test_migration_sets_schema_version(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
    assert row[0] == 2
    conn.close()
