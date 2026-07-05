"""Integration test for the data validation agent — requires GITHUB_TOKEN."""
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


@pytest.mark.integration
@pytest.mark.slow
def test_data_agent_loads_dataset_and_responds(tmp_path: Path) -> None:
    """Real LLM call — verifies the data_validator agent can use load_dataset tool."""
    from mlops_agents.data_validation.agent import build_data_agent

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
    content = _message_content_text(getattr(last, "content", ""))
    assert content.strip(), (
        "Last message must have non-empty content"
    )
    content_lower = content.lower()
    assert any(kw in content_lower for kw in ("row", "column", "feature", "iris", "shape")), (
        "Response should mention dataset properties"
    )
