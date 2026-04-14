"""Data MCP server — exposes dataset listing and preview as MCP tools.

Run via:
    uv run python -m mlops_agents.mcp_servers.data_server
"""

import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from mlops_agents.config.settings import settings

mcp = FastMCP("data-server")


@mcp.tool()
def list_datasets() -> str:
    """List available CSV datasets in the data directory."""
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return json.dumps({"error": f"Data directory not found: {data_dir}"})
    files = [f.name for f in data_dir.glob("*.csv")]
    return json.dumps({"datasets": files, "directory": str(data_dir)})


@mcp.tool()
def preview_dataset(filename: str, rows: int = 5) -> str:
    """Preview the first N rows of a dataset from the data directory.

    Args:
        filename: CSV filename (just the name, not full path).
        rows: Number of rows to preview (default 5).
    """
    path = Path(settings.data_dir) / filename
    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})
    df = pd.read_csv(path)
    return json.dumps({
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "head": df.head(rows).to_dict(orient="records"),
    }, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
