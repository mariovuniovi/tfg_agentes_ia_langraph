"""Integration test for the data validation agent — requires GITHUB_TOKEN."""
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
@pytest.mark.slow
def test_data_agent_loads_dataset_and_responds(tmp_path: Path) -> None:
    """Real LLM call — verifies the data_validator agent can use load_dataset tool."""
    from mlops_agents.agents.data_agent import build_data_agent

    src = Path("data/samples/iris.csv")
    if not src.exists():
        pytest.skip("data/samples/iris.csv not found")

    dst = tmp_path / "iris.csv"
    shutil.copy(src, dst)

    agent = build_data_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"Load the dataset at {dst} and report its shape, column names, and data types. "
            "Do not perform any further validation steps."
        ))]
    })

    messages = result.get("messages", [])
    assert len(messages) > 1, "Agent must produce at least one response beyond the input"
    last = messages[-1]
    assert hasattr(last, "content") and last.content.strip(), (
        "Last message must have non-empty content"
    )
    content_lower = last.content.lower()
    assert any(kw in content_lower for kw in ("row", "column", "feature", "iris", "shape")), (
        "Response should mention dataset properties"
    )
