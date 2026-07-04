"""Single source of truth for node categorization. Imported by api/services/pipeline.py
for the run_info SSE event and anywhere else that needs to classify a node by behavior."""
from __future__ import annotations

NODE_CATEGORIES: dict[str, list[str]] = {
    "agents":        ["data_validator", "planner"],
    "llm_nodes":     ["report_writer"],
    "deterministic": ["controller", "executor", "evaluation", "deployer"],
    "hitl":          ["dataset_approval", "deployment_approval"],
}


def is_agent(name: str) -> bool:
    return name in NODE_CATEGORIES["agents"]


def is_llm_node(name: str) -> bool:
    return name in NODE_CATEGORIES["llm_nodes"]


def is_deterministic(name: str) -> bool:
    return name in NODE_CATEGORIES["deterministic"]


def is_hitl(name: str) -> bool:
    return name in NODE_CATEGORIES["hitl"]
