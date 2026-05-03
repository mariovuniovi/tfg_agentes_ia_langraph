"""Supervisor node — LLM-based router that orchestrates the 4 specialist agents.

Uses structured output (RouterOutput) so every routing decision is auditable.
The supervisor uses a cheaper model (gpt-4.1-nano) to conserve rate limit budget.
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.types import Command

from mlops_agents.config.constants import AGENT_SUPERVISOR
from mlops_agents.config.settings import settings
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.state.schemas import RouterOutput
from mlops_agents.utils.llm import get_router_llm
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

_router_llm = get_router_llm()
_supervisor_prompt = get_prompt("supervisor").template


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

    dv_has_run = bool((state.get("agent_attempt_counts") or {}).get("data_validator", 0))
    snapshot_data = {
        "validation_passed": state.get("validation_passed") if dv_has_run else None,
        "evaluation_passed": state.get("evaluation_passed"),
        "deployment_decision": state.get("deployment_decision", "pending"),
        "error_message": state.get("error_message", ""),
        "training_run_id": state.get("training_run_id", ""),
    }
    state_snapshot = HumanMessage(content=f"Pipeline state:\n{json.dumps(snapshot_data)}")
    messages = [SystemMessage(content=_supervisor_prompt)] + list(state["messages"]) + [state_snapshot]
    response: RouterOutput = _router_llm.with_structured_output(RouterOutput).invoke(messages)

    logger.info(f"[{AGENT_SUPERVISOR}] → {response.next} | reason: {response.reasoning}")

    goto = END if response.next == "FINISH" else response.next

    if goto != END:
        counts = dict(state.get("agent_attempt_counts") or {})
        if counts.get(goto, 0) >= settings.max_attempts_per_agent:
            logger.warning(f"[supervisor] max attempts reached for {goto} — forcing END")
            return Command(goto=END, update={"next": "FINISH"})
        counts[goto] = counts.get(goto, 0) + 1
        return Command(goto=goto, update={"next": response.next, "agent_attempt_counts": counts})

    return Command(goto=END, update={"next": response.next})
