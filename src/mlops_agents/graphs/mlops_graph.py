"""Main LangGraph StateGraph — the MLOps pipeline topology.

Architecture:
  START → supervisor → [data_validator | planner | executor | evaluator | deployer] → supervisor → … → END

The supervisor (LLM with structured output) decides routing at every step.
Worker nodes wrap create_react_agent sub-graphs and return Command(goto="supervisor").
The deployer node includes a HITL interrupt() before the champion promotion step.

Run with:
    uv run python scripts/run_pipeline.py
"""

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from mlops_agents.agents.planner import PlannerError, planner_node
from mlops_agents.agents.registry import get_agent
from mlops_agents.agents.supervisor import supervisor_node
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.state.agent_state import AgentState
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Worker node wrappers
# Each wrapper invokes its react agent, extracts the final message,
# appends it to the shared message history, and returns to the supervisor.
# =============================================================================

def _extract_tool_json(messages: list, tool_name: str) -> Any:
    """Return the parsed JSON content of the last ToolMessage matching tool_name.

    Returns {} if no matching message is found or JSON parsing fails.
    Returns a list when the tool responded with a JSON array (e.g. get_best_run).
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return {}
    return {}


def _build_data_validator_context(
    state: AgentState,
    *,
    schema_json: str = "{}",
    schema_path: str = "",
) -> HumanMessage:
    paths: list[str] = state.get("dataset_paths") or []
    single_file_note = (
        "\nNOTE: Only ONE file was uploaded. "
        "Do NOT call merge_datasets. "
        "After load_dataset, go directly to apply_column_mapping on this single file."
        if len(paths) == 1 else ""
    )
    return HumanMessage(content=(
        f"Raw files: {json.dumps(paths)}\n"
        f"Schema path: {schema_path}\n"
        f"Target schema:\n{schema_json}"
        f"{single_file_note}"
    ))



def _build_evaluator_context(state: AgentState) -> HumanMessage:
    metrics = state.get("training_metrics") or {}
    problem_type = state.get("problem_type", "")

    if problem_type == "classification":
        f1_val = metrics.get("f1_score") or metrics.get("macro_f1") or metrics.get("weighted_f1")
        normalised = {**metrics}
        if f1_val is not None:
            normalised["f1_score"] = f1_val
        notes = (
            "NOTE: MLflow stores F1 under the key 'macro_f1'. "
            "Call get_best_run with metric='macro_f1' (ascending=False) to find the champion."
        )
    else:
        normalised = {**metrics}
        notes = (
            "NOTE: This is a forecasting/regression problem. "
            "Primary metric is 'rmse' (lower is better). "
            "Call get_best_run with metric='rmse' and ascending=True to find the run with the lowest RMSE. "
            "Do NOT apply accuracy or macro_f1 thresholds — those are for classification. "
            "Promotion criterion: candidate RMSE must exist and be lower than (or tie with) the current best RMSE."
        )

    return HumanMessage(content=(
        f"Problem type: {problem_type}\n"
        f"Task metadata: {json.dumps(state.get('task_metadata') or {})}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Trained model path: {state.get('trained_model_path', '')}\n"
        f"Training metrics: {json.dumps(normalised)}\n"
        f"{notes}"
    ))



def _validate_schema_contract(schema_data: dict) -> None:
    """Validate ML dataset contract fields. Raises ValueError on any violation."""
    column_names = {c["name"] for c in schema_data.get("columns", [])}

    problem_type = schema_data.get("problem_type")
    if problem_type not in ("classification", "regression", "forecasting"):
        raise ValueError(
            f"Schema missing or invalid 'problem_type'. Got: {problem_type!r}. "
            "Must be 'classification', 'regression', or 'forecasting'."
        )

    target_column = schema_data.get("target_column")
    if not target_column or target_column not in column_names:
        raise ValueError(
            f"'target_column' must be declared and exist in columns. Got: {target_column!r}."
        )

    if problem_type == "forecasting":
        required = ["datetime_column", "forecast_horizon", "frequency"]
        missing = [f for f in required if schema_data.get(f) is None]
        if missing:
            raise ValueError(f"Forecasting schema missing required fields: {missing}")

        if not isinstance(schema_data["forecast_horizon"], int) or schema_data["forecast_horizon"] <= 0:
            raise ValueError(
                f"'forecast_horizon' must be a positive integer. Got: {schema_data['forecast_horizon']!r}."
            )

        if schema_data["datetime_column"] not in column_names:
            raise ValueError(
                f"'datetime_column' '{schema_data['datetime_column']}' not found in columns."
            )

        for col in schema_data.get("series_id_columns", []):
            if col not in column_names:
                raise ValueError(f"'series_id_columns' entry '{col}' not found in columns.")


def data_validator_node(state: AgentState) -> Command[Literal["supervisor", "data_validator"]]:
    import pandas as pd

    schema_json: str = state.get("schema_json") or ""
    schema_data = json.loads(schema_json) if schema_json else {}

    if schema_json:
        schema_dir = Path("data/schema")
        schema_dir.mkdir(parents=True, exist_ok=True)
        schema_file = schema_dir / "uploaded_schema.json"
        schema_file.write_text(schema_json)
        schema_path = str(schema_file)
    else:
        schema_path = "(none)"

    if not schema_json:
        error_msg = "No schema uploaded. Upload a schema JSON before running the pipeline."
        logger.error(f"[data_validator] {error_msg}")
        return Command(
            update={
                "messages": [HumanMessage(content=error_msg, name="data_validator")],
                "validation_passed": False,
                "error_message": error_msg,
                "problem_type": "",
                "task_metadata": {},
                "dataset_summary": {},
                "validation_report": {},
                "processed_dataset_path": "",
                "schema_json": "",
            },
            goto="supervisor",
        )

    try:
        _validate_schema_contract(schema_data)
    except ValueError as exc:
        error_msg = f"Schema contract violation: {exc}"
        logger.error(f"[data_validator] {error_msg}")
        return Command(
            update={
                "messages": [HumanMessage(content=error_msg, name="data_validator")],
                "validation_passed": False,
                "error_message": error_msg,
                "problem_type": "",
                "task_metadata": {},
                "dataset_summary": {},
                "validation_report": {},
                "processed_dataset_path": "",
                "schema_json": schema_json,
            },
            goto="supervisor",
        )

    _max_attempts = 3
    counts = dict(state.get("agent_attempt_counts") or {})
    attempt = counts.get("data_validator", 1)

    # Build agent messages; on retry, prepend prior rejection feedback so the
    # agent knows what the human reviewer objected to and can try a different approach.
    context_msg = _build_data_validator_context(state, schema_json=schema_json, schema_path=schema_path)
    agent_messages: list[HumanMessage] = [context_msg]
    if attempt > 1:
        for msg in reversed(state.get("messages") or []):
            if getattr(msg, "name", "") == "data_validator" and "rejected" in str(msg.content).lower():
                agent_messages.append(HumanMessage(
                    content=(
                        f"Your previous attempt was rejected by a human reviewer. "
                        f"Their feedback: {msg.content}. "
                        "Please try a different approach to address this feedback."
                    )
                ))
                break

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": agent_messages})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    processed_path = (
        imputation_result.get("output_path", "")
        or mapping_result.get("output_path", "")
        or validation_result.get("output_path", "")
    )
    validation_passed = bool(validation_result.get("passed", False))

    dataset_summary: dict = {}
    # Validation passed — build preview and surface HITL for human sign-off.
    # Use df.to_json + json.loads so NaN/Inf become null — plain to_dict() leaves
    # Python float('nan') which serialises to the bare token NaN, invalid JSON.
    preview: dict = {"shape": [0, 0], "columns": [], "sample_rows": []}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            dataset_summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
            }
            preview = {
                "shape": list(df.shape),
                "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "sample_rows": json.loads(df.head(20).to_json(orient="records")),
            }
        except Exception:
            pass

    problem_type: str = schema_data.get("problem_type", "")
    task_metadata: dict[str, Any] = {"target_column": schema_data.get("target_column", "")}
    if problem_type == "forecasting":
        task_metadata.update({
            "datetime_column": schema_data.get("datetime_column", ""),
            "series_id_columns": schema_data.get("series_id_columns", []),
            "forecast_horizon": schema_data.get("forecast_horizon"),
            "frequency": schema_data.get("frequency", ""),
            "exogenous_columns": schema_data.get("exogenous_columns"),  # [{name, future_availability}]
        })

    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "processed_dataset_path": processed_path,
        "dataset_summary": dataset_summary,
        "problem_type": problem_type,
        "task_metadata": task_metadata,
        "schema_json": schema_json,
    }

    if not validation_passed:
        # Validation failed after agent's auto-fix attempt — abort without HITL.
        # The supervisor will see error_message set and select FINISH.
        error_msg = f"Data validation failed after auto-fix attempt: {final_message}"
        logger.warning("[data_validator] validation failed — aborting without HITL")
        return Command(
            update={**base_update, "error_message": error_msg},
            goto="supervisor",
        )

    missing_vals: dict = {}
    if isinstance(quality_report, dict):
        missing_vals = quality_report.get("missing_values", {})

    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "dataset_preview": preview,
        "validation_summary": {
            "passed": True,
            "missing_values": missing_vals,
            "schema_validated": True,
        },
        "imputation_applied": imputation_result,
    })

    if approval.get("approved", False):
        logger.info("[data_validator] approved — routing back to supervisor")
        return Command(update=base_update, goto="supervisor")

    comment = approval.get("comment", "")
    rejection_text = (
        f"Dataset rejected by human reviewer. Comment: {comment}"
        if comment
        else "Dataset rejected by human reviewer."
    )
    counts["data_validator"] = attempt + 1
    logger.info(f"[data_validator] rejected (attempt {attempt}/{_max_attempts}) — comment: {comment!r}")

    if attempt < _max_attempts:
        # Route back to data_validator for another attempt; rejection feedback
        # is stored in messages so the next run can inject it into agent context.
        return Command(
            update={
                "messages": [
                    HumanMessage(content=final_message, name="data_validator"),
                    HumanMessage(content=rejection_text, name="data_validator"),
                ],
                "agent_attempt_counts": counts,
                "validation_passed": False,
                "error_message": "",
            },
            goto="data_validator",
        )

    # Max attempts exhausted — abort.
    return Command(
        update={
            **base_update,
            "messages": [
                HumanMessage(content=final_message, name="data_validator"),
                HumanMessage(content=rejection_text, name="data_validator"),
            ],
            "validation_passed": False,
            "error_message": f"Dataset rejected after {_max_attempts} attempts. Last comment: {comment}",
        },
        goto="supervisor",
    )


def executor_node(state: AgentState) -> Command[Literal["supervisor"]]:
    from mlops_agents.training.executor import run_training_plan

    processed_path = Path(state["processed_dataset_path"])
    task_meta = {**(state.get("task_metadata") or {}), "problem_type": state.get("problem_type", "")}

    raw_plan = state.get("training_plan")
    if raw_plan is None:
        raise RuntimeError(
            "executor_node expected a planner-generated training_plan, but none was found. "
            "Ensure the planner node ran successfully before executor."
        )
    plan = TrainingPlan.model_validate(raw_plan)

    planner_out = state.get("_planner_output_record")

    result = run_training_plan(
        plan=plan,
        processed_dataset_path=processed_path,
        target_column=task_meta.get("target_column", "target"),
        task_metadata=task_meta,
        output_dir=Path("data/processed"),
        mlflow_experiment=settings.mlflow_experiment_name,
        planner_output=planner_out,
    )

    logger.info("[executor] completed — routing back to supervisor")
    return Command(
        goto="supervisor",
        update={
            "training_plan": plan.model_dump(),
            "train_pool_path": result.train_pool_path,
            "test_path": result.test_path,
            "split_metadata_path": result.split_metadata_path,
            "trained_model_path": result.champion_model_path,
            "training_run_id": result.mlflow_parent_run_id,
            "training_metrics": result.champion_metrics,
            "champion_candidate": result.champion_candidate,
            "experience_record_path": result.experience_record_path,
        },
    )


def _planner_node_with_error_handling(
    state: AgentState,
) -> Command[Literal["supervisor"]]:
    """Wrap planner_node to catch PlannerError and route to supervisor gracefully."""
    try:
        return planner_node(state)
    except PlannerError as exc:
        logger.error(f"[planner] failed after retry: {exc}")
        return Command(
            goto="supervisor",
            update={
                "planner_status": "failed",
                "planner_retry_used": True,
                "error_message": f"Model planner failed: {exc}",
                "messages": [HumanMessage(content=f"Planner failed: {exc}", name="planner")],
            },
        )


def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("evaluator")
    result = agent.invoke({"messages": [_build_evaluator_context(state)]})
    final_message = result["messages"][-1].content

    best_runs_raw = _extract_tool_json(result["messages"], "get_best_run")
    runs_list: list = best_runs_raw if isinstance(best_runs_raw, list) else []
    candidate = runs_list[0] if runs_list else {}
    baseline = runs_list[1] if len(runs_list) > 1 else {}

    evaluation_report = {
        "candidate_metrics": candidate.get("metrics", {}),
        "candidate_run_id": candidate.get("run_id", ""),
        "baseline_metrics": baseline.get("metrics", {}),
    }

    lower = final_message.lower()
    promoted = (
        "promot" in lower
        and "do not promot" not in lower
        and "not promot" not in lower
        and "retrain" not in lower
        and "reject" not in lower
    )

    logger.info(f"[evaluator] completed — promoted={promoted} — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="evaluator")],
            "evaluation_report": evaluation_report,
            "evaluation_passed": promoted,
        },
        goto="supervisor",
    )


def deployer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """Deployment node with HITL interrupt() before champion promotion.

    Execution flow:
    1. Register the model and assign 'challenger' alias (deterministic).
    2. Interrupt for human approval.
    3. On resume: if approved, assign 'champion' alias; if rejected, abort.
    """
    import json as _json
    from mlops_agents.tools.mlflow_tools import register_model, set_model_alias

    run_id = state.get("training_run_id", "")

    # Step 1: register + challenger alias (deterministic, no LLM)
    reg_raw = register_model.invoke({"run_id": run_id})
    reg = _json.loads(reg_raw)
    version = reg.get("version")
    model_name = reg.get("model_name", "")
    alias_raw = set_model_alias.invoke({"model_name": model_name, "alias": "challenger", "version": version})
    registration_summary = (
        f"Registered '{model_name}' version {version} from run {run_id}. "
        f"Alias 'challenger' assigned."
    )
    logger.info(f"[deployer] {registration_summary}")

    # Step 2: HITL — pause and wait for human approval
    approval = interrupt({
        "question": "Approve promotion of this model to 'champion' in the MLflow Model Registry?",
        "registration_summary": registration_summary,
        "instructions": "Reply with {'approved': true} to promote, or {'approved': false} to reject.",
    })

    # Step 3: handle approval decision
    if approval.get("approved", False):
        set_model_alias.invoke({"model_name": model_name, "alias": "champion", "version": version})
        outcome = f"APPROVED. '{model_name}' v{version} promoted to champion."
        logger.info(f"[deployer] {outcome}")
    else:
        reason = approval.get("reason", "No reason provided.")
        outcome = f"REJECTED. '{model_name}' v{version} NOT promoted. Reason: {reason}"
        logger.warning(f"[deployer] {outcome}")

    return Command(
        update={
            "messages": [HumanMessage(content=outcome, name="deployer")],
            "deployment_decision": "approved" if approval.get("approved") else "rejected",
        },
        goto="supervisor",
    )


# =============================================================================
# Graph construction
# =============================================================================

def _build_graph(checkpointer=None) -> StateGraph:
    """Build the MLOps StateGraph. Optionally attach a checkpointer for HITL.

    Nodes: supervisor, data_validator, planner, executor, evaluator, deployer.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_validator", data_validator_node)
    builder.add_node("planner", _planner_node_with_error_handling)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("deployer", deployer_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # All workers route back to supervisor via Command — no explicit edges needed

    return builder.compile(checkpointer=checkpointer)


# Default compiled graph with in-memory checkpointer (required for HITL interrupt)
_checkpointer = InMemorySaver()
graph = _build_graph(checkpointer=_checkpointer)


def main() -> None:
    """Run the full MLOps pipeline from the CLI, including HITL approval."""
    import sys

    dataset_paths = sys.argv[1:] if len(sys.argv) > 1 else ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    paths_display = ", ".join(dataset_paths)

    config = {"configurable": {"thread_id": "pipeline-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    initial_state: dict = {
        "messages": [
            HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")
        ],
        "next": "",
        "dataset_paths": dataset_paths,
        "processed_dataset_path": "",
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
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
        "agent_attempt_counts": {},
        "schema_json": "",
    }

    print(f"\n{'='*60}")
    print(f"MLOps Pipeline — files: {paths_display}")
    print(f"{'='*60}\n")

    for event in graph.stream(initial_state, config=config):
        if "__interrupt__" in event:
            interrupt_value = event["__interrupt__"][0].value
            _handle_hitl(graph, config, interrupt_value)
        else:
            for node_name in event:
                print(f"  [{node_name}] completed")

    print(f"\n{'='*60}")
    print("Pipeline finished.")
    print(f"{'='*60}\n")


def _handle_hitl(graph: Any, config: dict[str, Any], interrupt_value: dict[str, Any]) -> None:
    """Prompt the operator for HITL approval and resume the graph."""
    from langgraph.types import Command

    print(f"\n{'='*60}")
    print("HUMAN APPROVAL REQUIRED")
    print(f"{'='*60}")
    print(interrupt_value.get("question", "Approve this action?"))
    summary = interrupt_value.get("registration_summary", "")
    if summary:
        print(f"\nDetails:\n{summary}")
    print(f"{'='*60}")

    raw = input("\nApprove? (y/n): ").strip().lower()
    approved = raw == "y"
    resume: dict[str, Any] = {"approved": approved}
    if not approved:
        reason = input("Rejection reason (optional, press Enter to skip): ").strip()
        resume["reason"] = reason or "Rejected by operator"

    print()
    for event in graph.stream(Command(resume=resume), config=config):
        for node_name in event:
            print(f"  [{node_name}] completed")


if __name__ == "__main__":
    main()
