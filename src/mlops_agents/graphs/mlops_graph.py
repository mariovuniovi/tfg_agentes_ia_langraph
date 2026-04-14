"""Main LangGraph StateGraph — the MLOps pipeline topology.

Architecture:
  START → supervisor → [data_validator | trainer | evaluator | deployer] → supervisor → … → END

The supervisor (LLM with structured output) decides routing at every step.
Worker nodes wrap create_react_agent sub-graphs and return Command(goto="supervisor").
The deployer node includes a HITL interrupt() before the champion promotion step.

Run with:
    uv run python scripts/run_pipeline.py
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.managed.is_last_step import RemainingSteps
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

def _wrap_agent(agent_name: str, state: AgentState) -> Command[Literal["supervisor"]]:
    """Generic wrapper: invoke a react agent and route back to supervisor."""
    agent = get_agent(agent_name)
    result = agent.invoke({"messages": list(state["messages"])})
    final_message = result["messages"][-1].content

    logger.info(f"[{agent_name}] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name=agent_name)],
        },
        goto="supervisor",
    )


def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    return _wrap_agent("data_validator", state)


def trainer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    return _wrap_agent("trainer", state)


def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    return _wrap_agent("evaluator", state)


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
    """Run the full MLOps pipeline from the CLI."""
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./data/samples/iris.csv"

    config = {"configurable": {"thread_id": "pipeline-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    initial_state: dict = {
        "messages": [
            HumanMessage(content=f"Run the full MLOps pipeline on dataset: {dataset_path}")
        ],
        "next": "",
        "dataset_path": dataset_path,
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
        "retry_count": 0,
    }

    print(f"\n{'='*60}")
    print(f"MLOps Pipeline — dataset: {dataset_path}")
    print(f"{'='*60}\n")

    for event in graph.stream(initial_state, config=config):
        for node_name, node_output in event.items():
            print(f"[{node_name}] completed")

    print(f"\n{'='*60}")
    print("Pipeline finished.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
