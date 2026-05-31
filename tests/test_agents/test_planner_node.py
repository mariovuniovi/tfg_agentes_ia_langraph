"""Shim smoke-tests: agents/planner.py re-exports planner_node and PlannerError
from mlops_agents.planning.node. Full planner_node tests live in
tests/test_planning/test_node.py.
"""
from mlops_agents.agents.planner import PlannerError, planner_node
from mlops_agents.planning.node import PlannerError as PlannerErrorDirect
from mlops_agents.planning.node import planner_node as planner_node_direct


def test_shim_planner_node_is_same_object():
    assert planner_node is planner_node_direct


def test_shim_planner_error_is_same_class():
    assert PlannerError is PlannerErrorDirect
