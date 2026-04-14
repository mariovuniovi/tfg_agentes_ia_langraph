"""Entry-point runners for CLI scripts."""

import subprocess
import sys
from pathlib import Path


def run_dashboard() -> None:
    """Launch the Streamlit dashboard (registered as mlops-dashboard script)."""
    dashboard_path = Path(__file__).parent.parent.parent.parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )
