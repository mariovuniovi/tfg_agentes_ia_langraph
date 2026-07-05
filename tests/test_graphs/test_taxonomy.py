from mlops_agents.graphs.taxonomy import NODE_CATEGORIES, is_agent, is_deterministic, is_hitl, is_llm_node


def test_categories_are_disjoint_and_cover_all_nodes():
    agents = set(NODE_CATEGORIES["agents"])
    llm = set(NODE_CATEGORIES["llm_nodes"])
    det = set(NODE_CATEGORIES["deterministic"])
    hitl = set(NODE_CATEGORIES["hitl"])
    assert agents.isdisjoint(llm)
    assert agents.isdisjoint(det)
    assert llm.isdisjoint(det)
    assert hitl.isdisjoint(agents)
    assert hitl.isdisjoint(llm)
    assert hitl.isdisjoint(det)

def test_hitl_nodes():
    assert is_hitl("dataset_approval")
    assert is_hitl("deployment_approval")
    assert not is_hitl("planner")
    assert not is_hitl("executor")

def test_planner_is_agent_post_refactor():
    assert is_agent("planner")
    assert not is_llm_node("planner")

def test_report_writer_is_llm_node():
    assert is_llm_node("report_writer")

def test_executor_is_deterministic():
    assert is_deterministic("executor")

def test_unknown_node_is_none_of_the_above():
    assert not is_agent("foo")
    assert not is_llm_node("foo")
    assert not is_deterministic("foo")
