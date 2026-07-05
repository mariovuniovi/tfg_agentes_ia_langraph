# Data Validation HITL Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HITL gate after data validation that pauses the pipeline, shows the user a dataset preview in the Dataset tab, and lets them approve or reject (with an optional comment) before training begins — retries are orchestrated by the supervisor with a configurable per-agent attempt limit.

**Architecture:** `interrupt()` is added to `data_validator_node` following the same pattern as the existing deployer HITL. The supervisor is updated to increment `agent_attempt_counts` when routing and to force END when a target's count reaches `max_attempts_per_agent`. The frontend's Dataset tab gains a review panel that appears when `hitlPending && interruptValue.type === "data_validation"`.

**Tech Stack:** Python 3.12, LangGraph, FastAPI, asyncio, Next.js 14, Zustand, React Query, Tailwind CSS

---

## File Map

| File | Change |
|------|--------|
| `src/mlops_agents/config/settings.py` | Add `max_attempts_per_agent: int = 3` |
| `src/mlops_agents/state/agent_state.py` | Replace `retry_count: int` → `agent_attempt_counts: dict[str, int]` |
| `src/mlops_agents/agents/supervisor.py` | Check + increment `agent_attempt_counts` before routing |
| `src/mlops_agents/graphs/mlops_graph.py` | Add HITL `interrupt()` to `data_validator_node`; update `initial_state` in `main()` |
| `api/models/run.py` | Add `comment: str = ""` to `HITLDecision` |
| `api/services/run_store.py` | Add `hitl_comment: str = ""` to `RunEntry` |
| `api/routers/runs.py` | Save `body.comment` to `entry.hitl_comment` in `POST /approve` |
| `api/services/pipeline.py` | Change HITL `if` → `while` loop; derive agent label from payload; build proper resume dict |
| `frontend/types/api.ts` | Add `DataValidationInterrupt` interface; add `comment` to `HITLDecision` |
| `frontend/lib/api.ts` | No change (uses `HITLDecision` type, updated transitively) |
| `frontend/hooks/use-approve.ts` | Accept `comment` parameter |
| `frontend/components/pipeline/HITLGate.tsx` | Guard: return null when type is `data_validation` |
| `frontend/components/pipeline/ResultsDashboard.tsx` | Add `DatasetReviewPanel` to Dataset tab |
| `tests/test_agents/test_supervisor.py` | Update `make_state()` + add max-attempts tests |
| `tests/test_graphs/test_mlops_graph.py` | Add data_validator_node HITL tests |
| `api/tests/test_models.py` | Add `HITLDecision` comment field tests |
| `api/tests/test_run_store.py` | Add `hitl_comment` field test |
| `api/tests/test_runs.py` | Update approve test + add comment-saved test |
| `api/tests/test_pipeline.py` | Add while-loop and resume-dict tests |

---

## Task 1: Config — add `max_attempts_per_agent`

**Files:**
- Modify: `src/mlops_agents/config/settings.py`

- [ ] **Step 1: Add the field**

  In `settings.py`, add after `dataset_schema`:

  ```python
  max_attempts_per_agent: int = 3
  ```

- [ ] **Step 2: Verify it loads**

  Run: `uv run python -c "from mlops_agents.config.settings import settings; assert settings.max_attempts_per_agent == 3; print('ok')"`

  Expected: `ok`

- [ ] **Step 3: Commit**

  ```bash
  git add src/mlops_agents/config/settings.py
  git commit -m "feat: add max_attempts_per_agent to settings"
  ```

---

## Task 2: State — replace `retry_count` with `agent_attempt_counts`

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Modify: `tests/test_agents/test_supervisor.py` (update `make_state` helper)

- [ ] **Step 1: Update `agent_state.py`**

  Replace the `retry_count: int` line with:

  ```python
  agent_attempt_counts: dict[str, int]  # {"data_validator": 1, "trainer": 2, …}
  ```

  Remove `retry_count: int` entirely.

- [ ] **Step 2: Update `make_state` in supervisor tests**

  In `tests/test_agents/test_supervisor.py`, in `make_state()`, replace:

  ```python
  "retry_count": 0,
  ```

  with:

  ```python
  "agent_attempt_counts": {},
  ```

- [ ] **Step 3: Update `initial_state` in `mlops_graph.py` `main()`**

  In `src/mlops_agents/graphs/mlops_graph.py`, in the `main()` function's `initial_state` dict, replace:

  ```python
  "error_message": "",
  "retry_count": 0,
  ```

  with:

  ```python
  "error_message": "",
  "agent_attempt_counts": {},
  ```

- [ ] **Step 4: Run tests to verify no breakage**

  Run: `uv run pytest tests/test_agents/test_supervisor.py -v`

  Expected: all existing tests PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/mlops_agents/state/agent_state.py src/mlops_agents/graphs/mlops_graph.py tests/test_agents/test_supervisor.py
  git commit -m "feat: replace retry_count with agent_attempt_counts in state"
  ```

---

## Task 3: Supervisor — increment count and enforce max attempts

**Files:**
- Modify: `src/mlops_agents/agents/supervisor.py`
- Modify: `tests/test_agents/test_supervisor.py`

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_agents/test_supervisor.py`:

  ```python
  @patch("mlops_agents.agents.supervisor._router_llm")
  def test_supervisor_increments_attempt_count_when_routing(mock_llm):
      """Routing to data_validator should increment its count from 0 to 1."""
      mock_structured = MagicMock()
      mock_structured.invoke.return_value = RouterOutput(
          next="data_validator",
          reasoning="Start with validation.",
      )
      mock_llm.with_structured_output.return_value = mock_structured

      from mlops_agents.agents.supervisor import supervisor_node

      state = make_state(agent_attempt_counts={})
      command = supervisor_node(state)

      assert command.goto == "data_validator"
      assert command.update["agent_attempt_counts"] == {"data_validator": 1}


  @patch("mlops_agents.agents.supervisor._router_llm")
  def test_supervisor_forces_end_when_max_attempts_reached(mock_llm):
      """Supervisor must force END when target agent is at max attempts."""
      from langgraph.graph import END

      mock_structured = MagicMock()
      mock_structured.invoke.return_value = RouterOutput(
          next="data_validator",
          reasoning="Try validation again.",
      )
      mock_llm.with_structured_output.return_value = mock_structured

      from mlops_agents.agents.supervisor import supervisor_node

      state = make_state(agent_attempt_counts={"data_validator": 3})
      command = supervisor_node(state)

      assert command.goto == END
      mock_structured.invoke.assert_called_once()  # LLM was called but result overridden
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `uv run pytest tests/test_agents/test_supervisor.py::test_supervisor_increments_attempt_count_when_routing tests/test_agents/test_supervisor.py::test_supervisor_forces_end_when_max_attempts_reached -v`

  Expected: FAIL — `KeyError` or assertion error on `agent_attempt_counts`

- [ ] **Step 3: Update `supervisor.py`**

  Add import at top of `src/mlops_agents/agents/supervisor.py`:

  ```python
  from mlops_agents.config.settings import settings
  ```

  Replace the final lines of `supervisor_node` (the `goto =` and `return Command(...)` lines) with:

  ```python
  goto = END if response.next == "FINISH" else response.next

  if goto != END:
      counts = dict(state.get("agent_attempt_counts") or {})
      if counts.get(goto, 0) >= settings.max_attempts_per_agent:
          logger.warning(f"[supervisor] max attempts reached for {goto} — forcing END")
          return Command(goto=END, update={"next": "FINISH"})
      counts[goto] = counts.get(goto, 0) + 1
      return Command(goto=goto, update={"next": response.next, "agent_attempt_counts": counts})

  return Command(goto=END, update={"next": response.next})
  ```

- [ ] **Step 4: Run all supervisor tests**

  Run: `uv run pytest tests/test_agents/test_supervisor.py -v`

  Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/mlops_agents/agents/supervisor.py tests/test_agents/test_supervisor.py
  git commit -m "feat: supervisor increments agent_attempt_counts and enforces max_attempts_per_agent"
  ```

---

## Task 4: `data_validator_node` — HITL interrupt

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Modify: `tests/test_graphs/test_mlops_graph.py`

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_graphs/test_mlops_graph.py`:

  ```python
  from unittest.mock import MagicMock, patch
  from langchain_core.messages import HumanMessage, ToolMessage


  def _make_validator_state(tmp_path=None):
      from pathlib import Path
      import pandas as pd

      processed = ""
      if tmp_path:
          p = Path(tmp_path) / "processed.csv"
          pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
          processed = str(p)

      return {
          "messages": [HumanMessage(content="Run pipeline.")],
          "next": "",
          "dataset_paths": ["data/samples/iris_measurements.csv"],
          "dataset_path": processed,
          "validation_passed": False,
          "validation_report": {},
          "trained_model_path": "",
          "training_run_id": "",
          "training_metrics": {},
          "evaluation_passed": False,
          "evaluation_report": {},
          "best_model_uri": "",
          "deployment_decision": "pending",
          "deployment_status": "",
          "error_message": "",
          "agent_attempt_counts": {"data_validator": 1},
      }


  def _make_mock_agent(final_content="done", tool_name=None, tool_content="{}"):
      mock_agent = MagicMock()
      messages = [HumanMessage(content="context")]
      if tool_name:
          tm = ToolMessage(content=tool_content, tool_call_id="t1", name=tool_name)
          messages.append(tm)
      messages.append(HumanMessage(content=final_content))
      mock_agent.invoke.return_value = {"messages": messages}
      return mock_agent


  def test_data_validator_node_approved_returns_to_supervisor(tmp_path):
      """On approval, node returns Command(goto='supervisor') with validation fields."""
      state = _make_validator_state(tmp_path)
      mock_agent = _make_mock_agent(
          tool_name="apply_column_mapping",
          tool_content=f'{{"output_path": "{state["dataset_path"]}", "mapped_columns": 2}}',
      )

      with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
           patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
          from mlops_agents.graphs.mlops_graph import data_validator_node
          command = data_validator_node(state)

      assert command.goto == "supervisor"
      assert command.update["validation_passed"] is False  # tool returned empty {}


  def test_data_validator_node_rejected_injects_message(tmp_path):
      """On rejection, node injects a HumanMessage with the comment and sets validation_passed=False."""
      state = _make_validator_state(tmp_path)
      mock_agent = _make_mock_agent()

      with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
           patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": False, "comment": "rename column X"}):
          from mlops_agents.graphs.mlops_graph import data_validator_node
          command = data_validator_node(state)

      assert command.goto == "supervisor"
      assert command.update["validation_passed"] is False
      rejection_msgs = [
          m for m in command.update["messages"]
          if isinstance(m, HumanMessage) and "rename column X" in m.content
      ]
      assert len(rejection_msgs) == 1


  def test_data_validator_node_interrupt_payload_has_type(tmp_path):
      """The interrupt payload must have type='data_validation'."""
      state = _make_validator_state(tmp_path)
      mock_agent = _make_mock_agent()
      captured = {}

      def fake_interrupt(value):
          captured["payload"] = value
          return {"approved": True, "comment": ""}

      with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
           patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fake_interrupt):
          from mlops_agents.graphs.mlops_graph import data_validator_node
          data_validator_node(state)

      assert captured["payload"]["type"] == "data_validation"
      assert "dataset_preview" in captured["payload"]
      assert "validation_summary" in captured["payload"]
      assert "attempt" in captured["payload"]
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `uv run pytest tests/test_graphs/test_mlops_graph.py::test_data_validator_node_approved_returns_to_supervisor tests/test_graphs/test_mlops_graph.py::test_data_validator_node_rejected_injects_message tests/test_graphs/test_mlops_graph.py::test_data_validator_node_interrupt_payload_has_type -v`

  Expected: FAIL — `ImportError` or the node has no `interrupt()` call yet

- [ ] **Step 3: Update `data_validator_node` in `mlops_graph.py`**

  Replace the entire `data_validator_node` function with:

  ```python
  def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
      import pandas as pd
      from pathlib import Path as _Path
      from mlops_agents.config.settings import settings

      schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
      schema_json = schema_file.read_text() if schema_file.exists() else "{}"
      schema_path = str(schema_file.resolve())

      dataset_paths = state.get("dataset_paths", [])
      context_message = HumanMessage(
          content=(
              f"Raw files: {json.dumps(dataset_paths)}\n"
              f"Schema path: {schema_path}\n"
              f"Target schema:\n{schema_json}"
          )
      )

      agent = get_agent("data_validator")
      result = agent.invoke({"messages": list(state["messages"]) + [context_message]})
      final_message = result["messages"][-1].content

      quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
      mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
      validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")

      processed_path = mapping_result.get("output_path", "")
      validation_passed = bool(validation_result.get("passed", False))

      # Build dataset preview (up to 20 rows) for the HITL payload
      preview: dict = {"shape": [0, 0], "columns": [], "sample_rows": []}
      if processed_path:
          try:
              df = pd.read_csv(processed_path)
              preview = {
                  "shape": list(df.shape),
                  "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                  "sample_rows": df.head(20).to_dict(orient="records"),
              }
          except Exception:
              pass

      counts = dict(state.get("agent_attempt_counts") or {})
      attempt = counts.get("data_validator", 1)

      missing_vals: dict = {}
      if isinstance(quality_report, dict):
          missing_vals = quality_report.get("missing_values", {})

      approval = interrupt({
          "type": "data_validation",
          "question": "Review the processed dataset before training begins.",
          "attempt": attempt,
          "dataset_preview": preview,
          "validation_summary": {
              "passed": validation_passed,
              "missing_values": missing_vals,
              "schema_validated": bool(validation_result.get("passed", False)),
          },
      })

      base_update = {
          "messages": [HumanMessage(content=final_message, name="data_validator")],
          "validation_report": quality_report,
          "validation_passed": validation_passed,
          "dataset_path": processed_path,
      }

      if approval.get("approved", False):
          logger.info("[data_validator] approved — routing back to supervisor")
          return Command(update=base_update, goto="supervisor")

      comment = approval.get("comment", "")
      rejection_text = f"Dataset review rejected by human. Comment: {comment}" if comment else "Dataset review rejected by human."
      logger.info(f"[data_validator] rejected — comment: {comment!r}")
      return Command(
          update={
              **base_update,
              "messages": [
                  HumanMessage(content=final_message, name="data_validator"),
                  HumanMessage(content=rejection_text, name="data_validator"),
              ],
              "validation_passed": False,
          },
          goto="supervisor",
      )
  ```

- [ ] **Step 4: Run the new tests**

  Run: `uv run pytest tests/test_graphs/test_mlops_graph.py -v`

  Expected: all tests PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_mlops_graph.py
  git commit -m "feat: add data validation HITL interrupt to data_validator_node"
  ```

---

## Task 5: API — `HITLDecision` comment field, `RunEntry.hitl_comment`, router

**Files:**
- Modify: `api/models/run.py`
- Modify: `api/services/run_store.py`
- Modify: `api/routers/runs.py`
- Modify: `api/tests/test_models.py`
- Modify: `api/tests/test_run_store.py`
- Modify: `api/tests/test_runs.py`

- [ ] **Step 1: Write the failing tests**

  In `api/tests/test_models.py`, append:

  ```python
  def test_hitl_decision_has_comment_field():
      from api.models.run import HITLDecision
      d = HITLDecision(decision="reject", comment="bad column names")
      assert d.comment == "bad column names"

  def test_hitl_decision_comment_defaults_to_empty():
      from api.models.run import HITLDecision
      d = HITLDecision(decision="approve")
      assert d.comment == ""
  ```

  In `api/tests/test_run_store.py`, append:

  ```python
  def test_run_entry_has_hitl_comment_field():
      entry = _make_entry("r-comment")
      assert entry.hitl_comment == ""
  ```

  In `api/tests/test_runs.py`, append:

  ```python
  @pytest.mark.asyncio
  async def test_approve_saves_comment(client):
      with patch("api.routers.runs.pipeline_task"):
          start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
      run_id = start.json()["run_id"]
      entry = run_store_module.get_entry(run_id)
      entry.status = "awaiting_approval"
      resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "reject", "comment": "rename column X"})
      assert resp.status_code == 200
      assert entry.hitl_comment == "rename column X"
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `uv run pytest api/tests/test_models.py::test_hitl_decision_has_comment_field api/tests/test_run_store.py::test_run_entry_has_hitl_comment_field api/tests/test_runs.py::test_approve_saves_comment -v`

  Expected: FAIL — `ValidationError` or `AttributeError`

- [ ] **Step 3: Add `comment` field to `HITLDecision`**

  In `api/models/run.py`, update `HITLDecision`:

  ```python
  class HITLDecision(BaseModel):
      decision: Literal["approve", "reject"]
      reason: str = ""
      comment: str = ""
  ```

- [ ] **Step 4: Add `hitl_comment` to `RunEntry`**

  In `api/services/run_store.py`, add to `RunEntry` dataclass after `hitl_decision`:

  ```python
  hitl_comment: str = ""
  ```

- [ ] **Step 5: Save comment in `POST /approve`**

  In `api/routers/runs.py`, in the `approve_run` handler, add after `entry.hitl_decision = body.decision`:

  ```python
  entry.hitl_comment = body.comment
  ```

- [ ] **Step 6: Run the new tests**

  Run: `uv run pytest api/tests/test_models.py api/tests/test_run_store.py api/tests/test_runs.py -v`

  Expected: all tests PASS

- [ ] **Step 7: Commit**

  ```bash
  git add api/models/run.py api/services/run_store.py api/routers/runs.py api/tests/test_models.py api/tests/test_run_store.py api/tests/test_runs.py
  git commit -m "feat: add comment field to HITLDecision and RunEntry, save on POST /approve"
  ```

---

## Task 6: API pipeline — while loop, agent label, resume dict

**Files:**
- Modify: `api/services/pipeline.py`
- Modify: `api/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

  Append to `api/tests/test_pipeline.py`:

  ```python
  def _data_validation_interrupt_chunk():
      from unittest.mock import MagicMock
      interrupt = MagicMock()
      interrupt.value = {
          "type": "data_validation",
          "question": "Review dataset",
          "attempt": 1,
          "dataset_preview": {"shape": [10, 3], "columns": [], "sample_rows": []},
          "validation_summary": {"passed": True, "missing_values": {}, "schema_validated": True},
      }
      return ("ns", "updates", {"__interrupt__": [interrupt]})


  @pytest.mark.asyncio
  async def test_hitl_request_event_agent_derived_from_payload_type(mock_graph, tmp_path):
      """hitl_request event agent field should be 'data_validation', not hardcoded 'deployer'."""
      csv = tmp_path / "data.csv"
      csv.write_text("a,b\n1,2\n")
      call_count = 0

      async def fake_astream(*a, **kw):
          nonlocal call_count
          call_count += 1
          if call_count == 1:
              yield _data_validation_interrupt_chunk()
          else:
              yield _run_complete_chunk()

      mock_graph.astream = fake_astream

      with patch("api.services.pipeline.graph", mock_graph):
          run_id = "test-label"
          entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

          async def approve_later():
              await asyncio.sleep(0.01)
              entry.hitl_decision = "approve"
              entry.hitl_comment = ""
              entry.hitl_event.set()

          await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

      hitl_events = [e for e in entry.events if e["type"] == "hitl_request"]
      assert hitl_events[0]["agent"] == "data_validation"


  @pytest.mark.asyncio
  async def test_pipeline_resumes_with_dict_containing_approved_and_comment(mock_graph, tmp_path):
      """pipeline_task must resume with {"approved": bool, "comment": str}, not a raw string."""
      from langgraph.types import Command
      csv = tmp_path / "data.csv"
      csv.write_text("a,b\n1,2\n")
      call_count = 0
      resume_value = {}

      async def fake_astream(source, *a, **kw):
          nonlocal call_count
          call_count += 1
          if call_count == 1:
              yield _interrupt_chunk()
          else:
              if isinstance(source, Command):
                  resume_value["value"] = source.resume
              yield _run_complete_chunk()

      mock_graph.astream = fake_astream

      with patch("api.services.pipeline.graph", mock_graph):
          run_id = "test-resume-dict"
          entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

          async def approve_later():
              await asyncio.sleep(0.01)
              entry.hitl_decision = "approve"
              entry.hitl_comment = "looks good"
              entry.hitl_event.set()

          await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

      assert resume_value["value"] == {"approved": True, "comment": "looks good"}


  @pytest.mark.asyncio
  async def test_pipeline_handles_two_hitl_rounds(mock_graph, tmp_path):
      """pipeline_task while loop must handle two consecutive HITL interrupts."""
      csv = tmp_path / "data.csv"
      csv.write_text("a,b\n1,2\n")
      call_count = 0

      async def fake_astream(*a, **kw):
          nonlocal call_count
          call_count += 1
          if call_count == 1:
              yield _data_validation_interrupt_chunk()
          elif call_count == 2:
              yield _interrupt_chunk()
          else:
              yield _run_complete_chunk()

      mock_graph.astream = fake_astream

      with patch("api.services.pipeline.graph", mock_graph):
          run_id = "test-two-hitl"
          entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})
          approval_count = 0

          async def approve_later():
              nonlocal approval_count
              while True:
                  await asyncio.sleep(0.01)
                  if entry.status == "awaiting_approval":
                      entry.hitl_decision = "approve"
                      entry.hitl_comment = ""
                      entry.hitl_event.set()
                      approval_count += 1
                      if approval_count >= 2:
                          break

          await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

      assert entry.status == "complete"
      assert approval_count == 2
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `uv run pytest api/tests/test_pipeline.py::test_hitl_request_event_agent_derived_from_payload_type api/tests/test_pipeline.py::test_pipeline_resumes_with_dict_containing_approved_and_comment api/tests/test_pipeline.py::test_pipeline_handles_two_hitl_rounds -v`

  Expected: FAIL

- [ ] **Step 3: Update `pipeline.py`**

  In `api/services/pipeline.py`, inside `_stream()`, replace:

  ```python
  if "__interrupt__" in data:
      interrupt_list = data["__interrupt__"]
      interrupt_val = interrupt_list[0].value if interrupt_list else {}
      entry.status = "awaiting_approval"
      entry.interrupt_value = interrupt_val
      hitl_event: dict = {
          "type": "hitl_request",
          "agent": "deployer",
          "timestamp_ms": time.time() * 1000,
          "data": interrupt_val,
      }
      entry.events.append(hitl_event)
      await entry.queue.put(hitl_event)
      return  # exit loop; wait for approval below
  ```

  with:

  ```python
  if "__interrupt__" in data:
      interrupt_list = data["__interrupt__"]
      interrupt_val = interrupt_list[0].value if interrupt_list else {}
      entry.status = "awaiting_approval"
      entry.interrupt_value = interrupt_val
      hitl_agent = interrupt_val.get("type", "deployer")
      hitl_event: dict = {
          "type": "hitl_request",
          "agent": hitl_agent,
          "timestamp_ms": time.time() * 1000,
          "data": interrupt_val,
      }
      entry.events.append(hitl_event)
      await entry.queue.put(hitl_event)
      return  # exit loop; wait for approval below
  ```

  Then replace the post-`_stream` HITL block (the `if entry.status == "awaiting_approval":` block) with:

  ```python
  while entry.status == "awaiting_approval":
      entry.hitl_event.clear()
      await entry.hitl_event.wait()
      entry.status = "running"
      resume = {
          "approved": entry.hitl_decision == "approve",
          "comment": entry.hitl_comment,
      }
      await asyncio.wait_for(
          _stream(Command(resume=resume)), timeout=_STREAM_TIMEOUT
      )
  ```

- [ ] **Step 4: Run all pipeline tests**

  Run: `uv run pytest api/tests/test_pipeline.py -v`

  Expected: all tests PASS

- [ ] **Step 5: Commit**

  ```bash
  git add api/services/pipeline.py api/tests/test_pipeline.py
  git commit -m "feat: pipeline while loop for multi-round HITL, derive agent label, fix resume dict"
  ```

---

## Task 7: Frontend — types and `use-approve` hook

**Files:**
- Modify: `frontend/types/api.ts`
- Modify: `frontend/hooks/use-approve.ts`

- [ ] **Step 1: Add `DataValidationInterrupt` and update `HITLDecision` in `types/api.ts`**

  In `frontend/types/api.ts`, replace the existing `HITLDecision` interface and add after it:

  ```ts
  export interface HITLDecision {
    decision: 'approve' | 'reject'
    reason?: string
    comment?: string
  }

  export interface DataValidationInterrupt {
    type: 'data_validation'
    question: string
    attempt: number
    dataset_preview: {
      shape: [number, number]
      columns: { name: string; dtype: string }[]
      sample_rows: Record<string, unknown>[]
    }
    validation_summary: {
      passed: boolean
      missing_values: Record<string, number>
      schema_validated: boolean
    }
  }
  ```

- [ ] **Step 2: Update `use-approve.ts` to accept a comment**

  Replace the entire contents of `frontend/hooks/use-approve.ts` with:

  ```ts
  import { useMutation } from '@tanstack/react-query'
  import { approveRun } from '@/lib/api'
  import { useRunStore } from '@/stores/run-store'

  export function useApprove(runId: string | null) {
    const clearHITL = useRunStore((s) => s.clearHITL)

    const mutation = useMutation({
      mutationFn: ({ decision, comment }: { decision: 'approve' | 'reject'; comment?: string }) => {
        if (!runId) throw new Error('no run id')
        return approveRun(runId, { decision, comment: comment ?? '' })
      },
      onSuccess: () => clearHITL(),
    })

    return {
      approve: (decision: 'approve' | 'reject', comment?: string) =>
        mutation.mutateAsync({ decision, comment }),
      isPending: mutation.isPending,
      isError: mutation.isError,
    }
  }
  ```

- [ ] **Step 3: Verify TypeScript compiles**

  Run: `cd frontend && npx tsc --noEmit`

  Expected: no errors

- [ ] **Step 4: Commit**

  ```bash
  git add frontend/types/api.ts frontend/hooks/use-approve.ts
  git commit -m "feat: add DataValidationInterrupt type and comment param to useApprove"
  ```

---

## Task 8: Frontend — `HITLGate` guard for deployment-only

**Files:**
- Modify: `frontend/components/pipeline/HITLGate.tsx`

- [ ] **Step 1: Add `data_validation` guard**

  In `frontend/components/pipeline/HITLGate.tsx`, replace:

  ```tsx
  if (!hitlPending) return null
  ```

  with:

  ```tsx
  if (!hitlPending) return null
  if ((interruptValue as { type?: string })?.type === 'data_validation') return null
  ```

- [ ] **Step 2: Verify TypeScript compiles**

  Run: `cd frontend && npx tsc --noEmit`

  Expected: no errors

- [ ] **Step 3: Commit**

  ```bash
  git add frontend/components/pipeline/HITLGate.tsx
  git commit -m "feat: HITLGate skips rendering for data_validation HITL type"
  ```

---

## Task 9: Frontend — `DatasetReviewPanel` in `ResultsDashboard`

**Files:**
- Modify: `frontend/components/pipeline/ResultsDashboard.tsx`

- [ ] **Step 1: Add `DatasetReviewPanel` component and wire it into the Dataset tab**

  In `frontend/components/pipeline/ResultsDashboard.tsx`:

  1. Update the existing React import line to add `useRef`:

     ```tsx
     // before
     import { useMemo, useState } from 'react'
     // after
     import { useMemo, useRef, useState } from 'react'
     ```

     Then add these two new imports after the existing import block:

     ```tsx
     import type { DataValidationInterrupt } from '@/types/api'
     import { useApprove } from '@/hooks/use-approve'
     ```

  2. Add the `DatasetReviewPanel` component above the `ResultsDashboard` export:

     ```tsx
     function DatasetReviewPanel({
       runId,
       interruptValue,
     }: {
       runId: string | null
       interruptValue: DataValidationInterrupt
     }) {
       const commentRef = useRef<HTMLTextAreaElement>(null)
       const { approve, isPending } = useApprove(runId)
       const maxAttempts = 3
       const attempt = interruptValue.attempt ?? 1

       return (
         <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
           <div className="mb-2 flex items-center gap-2">
             <span className="text-sm font-semibold text-blue-900">Dataset Review</span>
             <span className="rounded bg-blue-100 px-2 py-0.5 text-xs text-blue-700">
               awaiting approval
             </span>
             <span className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
               Attempt {attempt} of {maxAttempts}
               {Array.from({ length: maxAttempts }).map((_, i) => (
                 <span
                   key={i}
                   className={`inline-block h-2 w-2 rounded-full ${
                     i < attempt ? 'bg-amber-400' : 'bg-slate-200'
                   }`}
                 />
               ))}
             </span>
           </div>
           <p className="mb-3 text-xs text-slate-500">
             Approve to proceed to training, or reject with a comment so the data agent can fix
             the issue and reprocess.
           </p>
           <label className="mb-1 block text-xs font-medium text-slate-500">
             Comment (optional)
           </label>
           <textarea
             ref={commentRef}
             rows={2}
             placeholder="e.g. rename column X, drop rows where value < 0…"
             className="mb-3 w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-1 focus:ring-blue-300"
           />
           <div className="flex gap-2">
             <button
               onClick={() => approve('approve', '')}
               disabled={isPending}
               className="rounded bg-emerald-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
             >
               ✓ Approve dataset
             </button>
             <button
               onClick={() => approve('reject', commentRef.current?.value ?? '')}
               disabled={isPending}
               className="rounded border border-red-200 bg-red-50 px-4 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
             >
               ✗ Reject &amp; retry
             </button>
           </div>
         </div>
       )
     }
     ```

  3. In the `ResultsDashboard` component, add two new selectors after `const status = ...`:

     ```tsx
     const runId = useRunStore((s) => s.runId)
     const hitlPending = useRunStore((s) => s.hitlPending)
     const interruptValue = useRunStore((s) => s.interruptValue)
     ```

  4. Inside the dataset tab section (after `<DatasetPanel ... />`), add:

     ```tsx
     {hitlPending && (interruptValue as { type?: string })?.type === 'data_validation' && (
       <DatasetReviewPanel
         runId={runId}
         interruptValue={interruptValue as unknown as DataValidationInterrupt}
       />
     )}
     ```

- [ ] **Step 2: Verify TypeScript compiles**

  Run: `cd frontend && npx tsc --noEmit`

  Expected: no errors

- [ ] **Step 3: Run full Python test suite to check for regressions**

  Run: `uv run pytest -m "not integration" -v`

  Expected: all tests PASS

- [ ] **Step 4: Commit**

  ```bash
  git add frontend/components/pipeline/ResultsDashboard.tsx
  git commit -m "feat: add DatasetReviewPanel to ResultsDashboard Dataset tab for data validation HITL"
  ```

---

## Task 10: Full regression check

- [ ] **Step 1: Run all unit tests**

  Run: `uv run pytest -m "not integration" -v`

  Expected: all tests PASS, no failures

- [ ] **Step 2: Run TypeScript check**

  Run: `cd frontend && npx tsc --noEmit`

  Expected: no errors

- [ ] **Step 3: Run linter**

  Run: `uv run ruff check . && uv run ruff format --check .`

  Expected: no errors (fix any issues found before proceeding)

- [ ] **Step 4: Final commit if any lint fixes were needed**

  ```bash
  git add -p
  git commit -m "fix: lint corrections after data validation HITL implementation"
  ```
