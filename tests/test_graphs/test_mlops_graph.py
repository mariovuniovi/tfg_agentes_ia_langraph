"""Tests for the main MLOps graph structure."""

import pytest


def test_graph_compiles_without_error():
    """The main graph should compile successfully on import."""
    from mlops_agents.graphs.mlops_graph import graph

    assert graph is not None


def test_graph_has_expected_nodes():
    """Graph should contain all 5 expected nodes."""
    from mlops_agents.graphs.mlops_graph import graph

    node_names = set(graph.nodes.keys())
    assert "supervisor" in node_names
    assert "data_validator" in node_names
    assert "trainer" in node_names
    assert "evaluator" in node_names
    assert "deployer" in node_names
