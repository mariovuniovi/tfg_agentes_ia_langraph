"""build_planner_agent — wraps langchain's create_agent with response_format=PlannerOutput."""
from langchain.agents import create_agent

from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm


def build_planner_agent(tools: list):
    """Build the planner ReAct agent. Tools must be closure-bound by build_planner_tools(...)
    in the caller so the agent never sees raw profile/task_metadata in tool args."""
    return create_agent(
        model=get_llm("planner"),
        tools=tools,
        system_prompt=get_prompt("planner").template,
        response_format=PlannerOutput,
        name="planner",
    )
