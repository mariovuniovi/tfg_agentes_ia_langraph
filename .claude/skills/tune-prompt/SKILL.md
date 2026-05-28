---
name: tune-prompt
description: Iterate on a YAML agent prompt to fix wrong outputs, hallucinations, or routing errors. Use when an agent gives wrong answers, routes incorrectly, misses required fields, or its tone/format is off.
---

# tune-prompt

**Cost warning:** Each iteration makes 1 real LLM call against GitHub Models quota (150 RPD free tier). Max 3 iterations = up to 3 calls. Always confirm with user before each round.

## Agent prompt files

| Agent | Prompt file |
|-------|------------|
| data_validator | `src/mlops_agents/prompts/data_agent.yaml` |
| evaluator | `src/mlops_agents/prompts/evaluation_agent.yaml` |
| supervisor | `src/mlops_agents/prompts/supervisor.yaml` |
| planner | `src/mlops_agents/prompts/planner.yaml` |
| deployer | `src/mlops_agents/prompts/deployment_agent.yaml` |

## Workflow

- [ ] **1. Read the current prompt**
  - `src/mlops_agents/prompts/<agent>.yaml`

- [ ] **2. Read recent bad output**
  - From `logs/pipeline.log` or a user-provided sample

- [ ] **3. Diagnose the failure mode**
  - Hallucination — agent invents facts not in context
  - Wrong routing — supervisor sends to wrong node
  - Missing field — required output key absent
  - Wrong tone/format — response not structured as expected

- [ ] **4. Propose 2–3 prompt variants**
  - For each: show the change and explain why it addresses the failure mode
  - Keep changes minimal — one hypothesis per variant

- [ ] **5. Check for integration test**
  - Does `tests/test_agents/test_<agent>_integration.py` exist?
  - If not, offer to create a minimal one (single input → assert expected output fields)

- [ ] **6. Confirm quota use**
  - Tell user: _"Round N will make 1 LLM call. Continue? (y/n)"_
  - Stop immediately if user says no

- [ ] **7. Run integration test with new prompt variant**

  ```bash
  uv run pytest tests/test_agents/test_<agent>_integration.py -v -m integration
  ```

- [ ] **8. Compare and select**
  - Show the output diff between variants
  - Select the variant that best addresses the failure mode

- [ ] **9. Write the winner back**
  - Overwrite `src/mlops_agents/prompts/<agent>.yaml` with the winning variant

## Guards

- Maximum 3 rounds — if not converging after 3, escalate to an architecture question
- Never change `temperature` or `max_tokens` without explicit user approval
- One agent at a time — never edit two agent prompts in the same session
- If no integration test exists and user declines to create one, stop — don't iterate blind
