"""LangChain @tool retrieval functions for experience pool and ML knowledge."""
from __future__ import annotations
import json
from langchain_core.tools import tool
from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.knowledge.reader import match_rules


@tool
def retrieve_similar_experiences(
    dataset_profile_json: str,
    problem_type: str,
    k: int = 5,
) -> str:
    """Retrieve up to k past experience records with the most similar dataset_profile.

    Returns JSON list of RetrievalView objects ordered by similarity_score descending.
    Empty list if no experiences match the problem_type.

    Args:
        dataset_profile_json: JSON string of the DatasetProfile dict.
        problem_type: One of "classification", "regression", "forecasting".
        k: Maximum number of results to return.
    """
    profile = json.loads(dataset_profile_json)
    pool = ExperiencePool(settings.experience_db_path)
    views = pool.find_similar(profile, problem_type, k)
    return json.dumps([v.model_dump() for v in views], default=str)


@tool
def retrieve_ml_knowledge(
    dataset_profile_json: str,
    problem_type: str,
) -> str:
    """Retrieve all curated ML rules whose applies_when conditions are satisfied by the profile.

    Returns JSON list of MLRule objects in YAML file order (curated order is meaningful).

    Args:
        dataset_profile_json: JSON string of the DatasetProfile dict.
        problem_type: One of "classification", "regression", "forecasting".
    """
    profile = json.loads(dataset_profile_json)
    profile["problem_type"] = problem_type
    rules = match_rules(profile)
    return json.dumps([r.model_dump() for r in rules], default=str)
