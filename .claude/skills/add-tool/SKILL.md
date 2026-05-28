---
name: add-tool
description: Add a new LangChain @tool to an existing agent in the MLOps pipeline. Use when user wants to add a tool, extend agent capabilities, or says "add a tool to <agent>".
---

# add-tool

## Workflow

- [ ] **1. Read context**
  - Read `src/mlops_agents/agents/<name>_agent.py`
  - Read `src/mlops_agents/state/agent_state.py`

- [ ] **2. Check for duplication**
  - List all function names in `src/mlops_agents/tools/` (Grep for `^def `)
  - If the tool name already exists, stop and tell the user

- [ ] **3. Choose domain file**
  - Ask the user which file the tool belongs in:
    - `data_tools.py` — dataset loading, profiling, splitting
    - `training_tools.py` — model fitting, hyperparameter search
    - `mlflow_tools.py` — run tracking, model registry, artifact logging
    - `evidently_tools.py` — drift detection, data quality reports
    - `memory_tools.py` — experience pool read/write

- [ ] **4. Implement the tool**

  Follow the existing pattern exactly — `@tool` decorator, type hints, docstring, `json.dumps()` return:

  ```python
  from langchain_core.tools import tool
  import json

  @tool
  def tool_name(param: str) -> str:
      """One-line description of what the tool does and returns."""
      result = {"key": "value"}
      return json.dumps(result)
  ```

  Guards:
  - Return type is always `str` — use `json.dumps()`, never return plain text
  - No LLM calls inside the tool
  - Import tool in the agent file from `mlops_agents.tools.<domain>`, never define inline

- [ ] **5. Register the tool**

  In `src/mlops_agents/agents/<name>_agent.py`, add to the `tools=[...]` list in `create_agent(...)`.

- [ ] **6. Write the unit test**

  - Deterministic tools (pure Python / pandas): test with real DataFrames, no mocks
  - Tools calling external services (MLflow, Evidently): mock with `unittest.mock`
  - File: `tests/test_tools/test_<domain>.py` (add to existing file if it exists)

  ```python
  def test_tool_name_returns_expected_json():
      result = json.loads(tool_name.invoke({"param": "value"}))
      assert result["key"] == "value"
  ```

- [ ] **7. Run the test**

  ```bash
  uv run pytest tests/ -k "tool_name" -v
  ```

  Expected: PASSED.
