"""Main LangGraph StateGraph — the MLOps pipeline topology.

Architecture:
  START → workflow_controller → [data_validator | dataset_approval | planner |
  executor | evaluation | report_writer | deployment_approval | deployer]
  → workflow_controller → … → END

workflow_controller is a deterministic router (no LLM): it reads state and
returns Command(goto=...), writing only routing-control updates inline. Every
other node returns its state slice via a typed contract from
mlops_agents.contracts.outputs (`.to_update()`), then routes back to
workflow_controller. HITL interrupts live in the two approval gate nodes.

Run with:
    uv run python scripts/run_pipeline.py
"""

from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from mlops_agents.config.settings import settings
from mlops_agents.contracts.outputs import (
    AuditStateUpdate,
    DeploymentStateUpdate,
    EvaluationStateUpdate,
    PlannerErrorStateUpdate,
    TrainingStateUpdate,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.data_validation.node import data_validator_node
from mlops_agents.deployment.deployer import run_deployer as run_deployer_module
from mlops_agents.evaluation.promotion import evaluate_promotion
from mlops_agents.evaluation.report_writer import run_report_writer
from mlops_agents.graphs.approval_nodes import dataset_approval_node, deployment_approval_node
from mlops_agents.graphs.workflow_controller import workflow_controller
from mlops_agents.planning.node import PlannerError, planner_node
from mlops_agents.state.agent_state import AgentState
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Worker node wrappers
# Each wrapper delegates to its stage's domain package and routes the typed
# state update back to the workflow_controller.
# =============================================================================

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
    output = TrainingStateUpdate.from_training_result(result, training_plan=plan.model_dump())
    return Command(goto="workflow_controller", update=output.to_update())


def _planner_node_with_error_handling(
    state: AgentState,
) -> Command[Literal["workflow_controller"]]:
    """Wrap planner_node to catch PlannerError and route to workflow_controller gracefully."""
    try:
        return planner_node(state)
    except PlannerError as exc:
        logger.error(f"[planner] failed after retry: {exc}")
        output = PlannerErrorStateUpdate(error_message=f"Model planner failed: {exc}")
        return Command(
            goto="workflow_controller",
            update=output.to_update(
                messages=[HumanMessage(content=f"Planner failed: {exc}", name="planner")]
            ),
        )


def evaluation_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic promotion decision — no LLM."""
    result = evaluate_promotion(state)
    logger.info(f"[evaluation] passed={result['evaluation_passed']}")
    return Command(
        update=EvaluationStateUpdate(**result).to_update(), goto="workflow_controller"
    )


def report_writer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Audit LLM node — produces structured EvaluationReport."""
    result = run_report_writer(state)
    return Command(
        update=AuditStateUpdate(**result).to_update(), goto="workflow_controller"
    )


def deployer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic deployment — Gate 2 has already approved upstream."""
    result = run_deployer_module(state)
    return Command(
        update=DeploymentStateUpdate(**result).to_update(), goto="workflow_controller"
    )


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
