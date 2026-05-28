---
name: debug-pipeline
description: Debug MLOps pipeline failures by reading logs, identifying the failing node, and proposing a minimal fix. Use when a pipeline run failed, a node threw an error, or output was unexpected.
---

# debug-pipeline

## Workflow

- [ ] **0. Check LangSmith tracing**
  - Open `.env` and check for `LANGCHAIN_TRACING_V2=true`
  - If absent or false, prompt: _"Enable LANGCHAIN_TRACING_V2=true and re-run for full node traces. Continue without it? (y/n)"_
  - Proceed only if the user confirms

- [ ] **1. Read logs**
  - Use the Agent tool to dispatch the `log-reader` subagent defined in `.claude/agents/log-reader.md`
  - Pass the error keyword or run ID as input
  - Wait for the structured error summary

- [ ] **2. Identify the failing node**
  - From the LangSmith trace (preferred) or the log summary (fallback)
  - Node names: `data_validator`, `planner`, `trainer` (deterministic executor), `evaluator`, `deployer`
  - LLM agents: `data_validator`, `evaluator` → prompt issues are likely
  - Deterministic nodes: `planner`, `trainer`, `deployer` → code bugs are likely

- [ ] **3. Read the node source**
  - Agent nodes: `src/mlops_agents/agents/<node>_agent.py`
  - Graph nodes / deterministic: `src/mlops_agents/graphs/mlops_graph.py`
  - Prompt (if LLM node): `src/mlops_agents/prompts/<agent>.yaml`

- [ ] **4. Read relevant state fields**
  - `src/mlops_agents/state/agent_state.py` — find the fields the failing node reads/writes

- [ ] **5. Propose a minimal reproduction**

  ```python
  # inline script or pytest case
  from mlops_agents.agents.<node>_agent import build_<node>_agent
  agent = build_<node>_agent()
  result = agent.invoke({"messages": [HumanMessage(content="<minimal input>")]})
  print(result)
  ```

- [ ] **6. Propose and apply the fix**
  - Explain the root cause
  - Make the minimal change (prompt edit or code fix — not both unless both are broken)

- [ ] **7. Confirm no regressions**

  ```bash
  uv run pytest -m "not integration"
  ```

  Expected: all pass.

## Notes

- For LLM agent failures, check the prompt BEFORE the code
- For deterministic node failures, check the code (they don't use prompts)
- Future: Langfuse / LangSmith full integration for deeper traces
