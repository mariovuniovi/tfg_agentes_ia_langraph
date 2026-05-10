"""Apply pending SQLite migrations in version order."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)
_MIGRATIONS_DIR = Path(__file__).parent


def _read_current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        return row[0] or 0
    except sqlite3.OperationalError:
        return 0


def apply_pending_migrations(db_path: Path) -> None:
    """Apply numbered migrations <NNN>_*.sql in order if their version > current. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        current = _read_current_version(conn)
        sql_files = sorted(f for f in _MIGRATIONS_DIR.glob("*.sql") if f.stem[0].isdigit())
        for mig in sql_files:
            version = int(mig.stem.split("_")[0])
            if version > current:
                logger.info(f"Applying migration {mig.name}")
                with conn:
                    conn.executescript(mig.read_text())
    finally:
        conn.close()
