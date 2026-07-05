"""build_planner_agent — wraps langchain's create_agent with response_format=PlannerOutput."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


def build_planner_agent(tools: list[BaseTool]) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the planner ReAct agent. Tools must be closure-bound by build_planner_tools(...)
    in the caller so the agent never sees raw profile/task_metadata in tool args."""
    return create_agent(
        model=get_llm("planner"),
        tools=tools,
        system_prompt=get_prompt("planner").template,
        response_format=PlannerOutput,
        name="planner",
    )
