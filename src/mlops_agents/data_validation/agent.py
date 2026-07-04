"""Data Validation Agent — validates datasets before they enter the pipeline."""

from functools import cache
from typing import Any

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_data_quality,
    check_missing_values,
    detect_temporal_gaps,
    impute_missing_values,
    load_dataset,
    merge_datasets,
    parse_datetime_column,
    validate_against_schema,
)
from mlops_agents.tools.join_discovery_tools import (
    evaluate_join_candidates,
    execute_join_plan,
)
from mlops_agents.utils.llm import get_llm


@cache
def get_data_agent() -> Any:
    """Return the data validation agent, built lazily on first use and cached."""
    return build_data_agent()


def build_data_agent() -> Any:
    """Build and return the data validation react agent."""
    return create_agent(
        model=get_llm("data_agent"),
        tools=[
            load_dataset,
            merge_datasets,
            evaluate_join_candidates,
            execute_join_plan,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            impute_missing_values,
            parse_datetime_column,
            detect_temporal_gaps,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
