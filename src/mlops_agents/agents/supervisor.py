"""Supervisor node — LLM-based router that orchestrates the 4 specialist agents.

Uses structured output (RouterOutput) so every routing decision is auditable.
The supervisor uses a cheaper model (gpt-4.1-nano) to conserve rate limit budget.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.types import Command

from mlops_agents.config.constants import AGENT_SUPERVISOR
from mlops_agents.state.agent_state import AgentState
from mlops_agents.state.schemas import RouterOutput
from mlops_agents.utils.llm import get_router_llm
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

_router_llm = get_router_llm()

SUPERVISOR_SYSTEM_PROMPT = """You are the MLOps Pipeline Supervisor coordinating a team of 4 specialist agents.

Your team:
- data_validator: Validates dataset schema, checks for missing values, and detects data drift using Evidently AI.
- trainer: Tunes hyperparameters with Optuna and trains scikit-learn models, logging everything to MLflow.
- evaluator: Evaluates trained models on test data, compares against the production baseline, and recommends promotion or rejection.
- deployer: Registers the best model in the MLflow Model Registry and requests human approval before promotion to production.

PIPELINE RULES (follow these strictly):
1. Always start with data_validator — never skip data validation.
2. Only proceed to trainer if the data_validator reports validation_passed=True.
3. Only proceed to evaluator after training is complete (training_run_id is set in state).
4. Only proceed to deployer if the evaluator recommends 'promote'.
5. If any stage reports an error or failure, select FINISH and report the failure clearly.
6. Select FINISH when the full pipeline completes successfully.
7. Do not route to the same agent twice in a row unless recovering from a transient error.

Always include a brief reasoning for your routing decision."""


def supervisor_node(
    state: AgentState,
) -> Command[Literal["data_validator", "trainer", "evaluator", "deployer", "__end__"]]:
    """Supervisor node: reads state and decides which agent to call next.

    Uses structured output to enforce a valid routing decision.
    The reasoning field is logged for thesis analysis.
    """
    # Graceful exit if approaching recursion limit
    remaining: RemainingSteps | None = state.get("remaining_steps")  # type: ignore[assignment]
    if remaining is not None and remaining <= 2:
        logger.warning("Approaching recursion limit — forcing FINISH")
        return Command(goto=END, update={"next": "FINISH"})

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + list(state["messages"])
    response: RouterOutput = _router_llm.with_structured_output(RouterOutput).invoke(messages)

    logger.info(f"[{AGENT_SUPERVISOR}] → {response.next} | reason: {response.reasoning}")

    goto = END if response.next == "FINISH" else response.next
    return Command(goto=goto, update={"next": response.next})
