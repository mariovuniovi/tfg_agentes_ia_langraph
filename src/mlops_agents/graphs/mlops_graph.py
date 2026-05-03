"""Main LangGraph StateGraph — the MLOps pipeline topology.

Architecture:
  START → supervisor → [data_validator | trainer | evaluator | deployer] → supervisor → … → END

The supervisor (LLM with structured output) decides routing at every step.
Worker nodes wrap create_react_agent sub-graphs and return Command(goto="supervisor").
The deployer node includes a HITL interrupt() before the champion promotion step.

Run with:
    uv run python scripts/run_pipeline.py
"""

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from mlops_agents.agents.registry import get_agent
from mlops_agents.agents.supervisor import supervisor_node
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
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
    return HumanMessage(content=(
        f"Raw files: {json.dumps(state.get('dataset_paths', []))}\n"
        f"Schema path: {schema_path}\n"
        f"Target schema:\n{schema_json}"
    ))


def _build_trainer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Canonical dataset: {state.get('dataset_path', '')}\n"
        f"Dataset summary: {json.dumps(state.get('dataset_summary') or {})}"
    ))


def _build_evaluator_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Trained model path: {state.get('trained_model_path', '')}\n"
        f"Training metrics: {json.dumps(state.get('training_metrics') or {})}"
    ))


def _build_deployer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Best model URI: {state.get('best_model_uri', '')}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Evaluation report: {json.dumps(state.get('evaluation_report') or {})}"
    ))


def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    from pathlib import Path as _Path

    import pandas as pd

    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    schema_path = str(schema_file.resolve())

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": [_build_data_validator_context(state, schema_json=schema_json, schema_path=schema_path)]})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    processed_path = (
        mapping_result.get("output_path", "")
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

    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "dataset_path": processed_path,
        "dataset_summary": dataset_summary,
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

    # Validation passed — build preview and surface HITL for human sign-off.
    # Use df.to_json + json.loads so NaN/Inf become null — plain to_dict() leaves
    # Python float('nan') which serialises to the bare token NaN, invalid JSON.
    preview: dict = {"shape": [0, 0], "columns": [], "sample_rows": []}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            preview = {
                "shape": list(df.shape),
                "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "sample_rows": json.loads(df.head(20).to_json(orient="records")),
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
            "passed": True,
            "missing_values": missing_vals,
            "schema_validated": True,
        },
        "imputation_applied": imputation_result,
    })

    if approval.get("approved", False):
        logger.info("[data_validator] approved — routing back to supervisor")
        return Command(update=base_update, goto="supervisor")

    # Human rejected a validated+imputed dataset. Abort — retrying cannot help
    # because tools and strategy are deterministic.
    comment = approval.get("comment", "")
    rejection_text = (
        f"Dataset rejected by human reviewer. Comment: {comment}"
        if comment
        else "Dataset rejected by human reviewer."
    )
    logger.info(f"[data_validator] rejected — comment: {comment!r}")
    return Command(
        update={
            **base_update,
            "messages": [
                HumanMessage(content=final_message, name="data_validator"),
                HumanMessage(content=rejection_text, name="data_validator"),
            ],
            "validation_passed": False,
            "error_message": rejection_text,
        },
        goto="supervisor",
    )


def trainer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("trainer")
    result = agent.invoke({"messages": list(state["messages"])})
    final_message = result["messages"][-1].content

    train_result: dict = _extract_tool_json(result["messages"], "train_model")
    mlflow_result: dict = _extract_tool_json(result["messages"], "log_experiment")

    training_metrics = {
        "model_type": train_result.get("model_type", ""),
        "train_accuracy": train_result.get("train_accuracy", 0.0),
        "val_accuracy": train_result.get("val_accuracy", 0.0),
    }

    logger.info("[trainer] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="trainer")],
            "training_metrics": training_metrics,
            "training_run_id": mlflow_result.get("run_id", ""),
            "trained_model_path": train_result.get("model_path", ""),
        },
        goto="supervisor",
    )


def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("evaluator")
    result = agent.invoke({"messages": list(state["messages"])})
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

    logger.info("[evaluator] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="evaluator")],
            "evaluation_report": evaluation_report,
            "evaluation_passed": bool(candidate),
        },
        goto="supervisor",
    )


def deployer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """Deployment node with HITL interrupt() before champion promotion.

    Execution flow:
    1. Run the deployment agent (registers model, sets 'challenger' alias).
    2. Interrupt for human approval.
    3. On resume: if approved, set 'champion' alias; if rejected, abort.
    """
    # Step 1: register model and set challenger alias
    agent = get_agent("deployer")
    result = agent.invoke({"messages": list(state["messages"])})
    registration_summary = result["messages"][-1].content

    # Step 2: HITL — pause and wait for human approval
    logger.info("[deployer] Pausing for human approval...")
    approval = interrupt({
        "question": "Approve promotion of this model to 'champion' in the MLflow Model Registry?",
        "registration_summary": registration_summary,
        "instructions": "Reply with {'approved': true} to promote, or {'approved': false} to reject.",
    })

    # Step 3: handle approval decision
    if approval.get("approved", False):
        outcome = f"APPROVED. Model promoted to champion.\n\nRegistration details:\n{registration_summary}"
        logger.info("[deployer] Deployment approved by human reviewer.")
    else:
        reason = approval.get("reason", "No reason provided.")
        outcome = f"REJECTED by human reviewer. Reason: {reason}\n\nModel NOT promoted to champion."
        logger.warning(f"[deployer] Deployment rejected: {reason}")

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
    """Build the MLOps StateGraph. Optionally attach a checkpointer for HITL."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_validator", data_validator_node)
    builder.add_node("trainer", trainer_node)
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
        "dataset_path": "",
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
