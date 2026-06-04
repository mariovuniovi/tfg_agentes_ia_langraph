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
from langgraph.types import Command

from mlops_agents.planning.node import PlannerError, planner_node
from mlops_agents.agents.registry import get_agent
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.deployment.deployer import run_deployer as run_deployer_module
from mlops_agents.evaluation.promotion import evaluate_promotion
from mlops_agents.evaluation.report_writer import run_report_writer
from mlops_agents.graphs.approval_nodes import dataset_approval_node, deployment_approval_node
from mlops_agents.graphs.workflow_controller import workflow_controller
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
    from mlops_agents.tools.join_discovery_tools import profile_raw_datasets as _profile

    paths: list[str] = state.get("dataset_paths") or []

    # Build name → path mapping; use filename stem as dataset name
    raw_paths = {Path(p).stem: p for p in paths}

    profiles_section = ""
    if len(paths) > 1:
        try:
            profiles = _profile(raw_paths)
            profiles_section = "\nRaw dataset profiles:\n" + json.dumps(
                [p.model_dump() for p in profiles], default=str, indent=2
            )
        except Exception as exc:
            profiles_section = f"\n(Could not pre-profile raw files: {exc})"

    single_file_note = (
        "\nNOTE: Only ONE file was uploaded. "
        "Do NOT call merge_datasets or execute_join_plan. "
        "After load_dataset, go directly to apply_column_mapping on this single file."
        if len(paths) == 1 else ""
    )
    return HumanMessage(content=(
        f"Raw files: {json.dumps(paths)}\n"
        f"Schema path: {schema_path}\n"
        f"Target schema:\n{schema_json}"
        f"{profiles_section}"
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


def data_validator_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
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
            goto="workflow_controller",
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
            goto="workflow_controller",
        )

    # Build agent messages; on retry, prepend prior rejection feedback so the
    # agent knows what the human reviewer objected to and can try a different approach.
    context_msg = _build_data_validator_context(state, schema_json=schema_json, schema_path=schema_path)
    agent_messages: list[HumanMessage] = [context_msg]
    rejection_comment = state.get("dataset_rejection_comment") or ""
    if rejection_comment:
        agent_messages.append(HumanMessage(
            content=(
                f"Your previous attempt was rejected by a human reviewer. "
                f"Their feedback: {rejection_comment}. "
                "Please try a different approach to address this feedback."
            )
        ))

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": agent_messages})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    join_exec_result: dict = _extract_tool_json(result["messages"], "execute_join_plan")
    eval_result: dict = _extract_tool_json(result["messages"], "evaluate_join_candidates")

    data_join_plan = join_exec_result.get("join_plan")          # echoed by execute_join_plan
    data_join_evaluations = eval_result.get("evaluations", [])

    # Capture base dataset row count for audit
    data_join_base_nrows: int | None = None
    if data_join_plan:
        base_name = data_join_plan.get("base_dataset", {}).get("dataset_name")
        if base_name:
            raw_paths = {Path(p).stem: p for p in (state.get("dataset_paths") or [])}
            base_path = raw_paths.get(base_name)
            if base_path and Path(base_path).exists():
                try:
                    data_join_base_nrows = len(pd.read_csv(base_path))
                except Exception:
                    pass

    processed_path = (
        imputation_result.get("output_path", "")
        or mapping_result.get("output_path", "")
        or validation_result.get("output_path", "")
    )
    validation_passed = bool(validation_result.get("passed", False))

    dataset_summary: dict = {}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            dataset_summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
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
        "data_join_plan": data_join_plan,
        "data_join_base_nrows": data_join_base_nrows,
        "data_join_evaluations": data_join_evaluations,
    }

    if not validation_passed:
        # Validation failed after agent's auto-fix attempt — abort without HITL.
        # The workflow_controller will see error_message set and select FINISH.
        error_msg = f"Data validation failed after auto-fix attempt: {final_message}"
        logger.warning("[data_validator] validation failed — aborting without HITL")
        return Command(
            update={**base_update, "error_message": error_msg},
            goto="workflow_controller",
        )

    return Command(
        update={
            **base_update,
            "dataset_rejection_comment": "",
        },
        goto="workflow_controller",
    )


def executor_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
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

    logger.info("[executor] completed — routing back to workflow_controller")
    return Command(
        goto="workflow_controller",
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
) -> Command[Literal["workflow_controller"]]:
    """Wrap planner_node to catch PlannerError and route to workflow_controller gracefully."""
    try:
        return planner_node(state)
    except PlannerError as exc:
        logger.error(f"[planner] failed after retry: {exc}")
        return Command(
            goto="workflow_controller",
            update={
                "planner_status": "failed",
                "planner_retry_used": True,
                "error_message": f"Model planner failed: {exc}",
                "messages": [HumanMessage(content=f"Planner failed: {exc}", name="planner")],
            },
        )


def evaluation_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic promotion decision — no LLM."""
    result = evaluate_promotion(state)
    logger.info(f"[evaluation] passed={result['evaluation_passed']}")
    return Command(update=result, goto="workflow_controller")


def report_writer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Audit LLM node — produces structured EvaluationReport."""
    result = run_report_writer(state)
    return Command(update=result, goto="workflow_controller")


def deployer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic deployment — Gate 2 has already approved upstream."""
    result = run_deployer_module(state)
    return Command(update=result, goto="workflow_controller")


# =============================================================================
# Graph construction
# =============================================================================

def _build_graph(checkpointer=None) -> StateGraph:
    """Build the refactored MLOps StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("workflow_controller", workflow_controller)
    builder.add_node("data_validator", data_validator_node)
    builder.add_node("dataset_approval", dataset_approval_node)
    builder.add_node("planner", _planner_node_with_error_handling)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("report_writer", report_writer_node)
    builder.add_node("deployment_approval", deployment_approval_node)
    builder.add_node("deployer", deployer_node)

    builder.add_edge(START, "workflow_controller")
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
        "evaluation_passed": None,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
        "schema_json": "",
        "dataset_approved": None,
        "dataset_rejection_comment": "",
        "deployment_approved": None,
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "evaluation_report_audit": None,
        "evaluation_report_audit_status": "",
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
