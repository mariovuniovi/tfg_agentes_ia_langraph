"""CLI script to run the full MLOps agent pipeline.

Usage:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py data/samples/iris.csv
"""

import sys
from pathlib import Path

# Ensure src is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlops_agents.graphs.cli import main

if __name__ == "__main__":
    main()
