"""Data Validation Agent — validates datasets before they enter the pipeline."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    impute_missing_values,
    load_dataset,
    merge_datasets,
    validate_against_schema,
)
from mlops_agents.tools.evidently_tools import check_data_drift, check_data_quality
from mlops_agents.utils.llm import get_llm


def build_data_agent():
    """Build and return the data validation react agent."""
    return create_agent(
        model=get_llm("data_validator"),
        tools=[
            load_dataset,
            merge_datasets,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            check_data_drift,
            impute_missing_values,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
