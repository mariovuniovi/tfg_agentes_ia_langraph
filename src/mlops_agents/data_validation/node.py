"""Data validation graph node — orchestrates the data validation react agent.

Validates the uploaded schema contract deterministically, builds the agent's
context message, invokes the agent (with soft-failure handling for timeouts
and malformed tool calls), and extracts the validation results from the
agent's tool messages into a typed DataValidationStateUpdate.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from openai import APITimeoutError

from mlops_agents.contracts.outputs import DataValidationStateUpdate
from mlops_agents.data_validation.agent import get_data_agent
from mlops_agents.data_validation.context import build_data_validator_context, extract_tool_json
from mlops_agents.data_validation.schema_contract import validate_schema_contract
from mlops_agents.state.agent_state import AgentState
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def data_validator_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    import pandas as pd

    schema_json: str = state.get("schema_json") or ""
    schema_data = json.loads(schema_json) if schema_json else {}

    if schema_json:
        schema_dir = Path("data/schema")
        schema_dir.mkdir(parents=True, exist_ok=True)
        schema_file = schema_dir / "uploaded_schema.json"
        schema_file.write_text(schema_json)
        schema_path = str(schema_file)
    else:
        schema_path = "(none)"

    if not schema_json:
        error_msg = "No schema uploaded. Upload a schema JSON before running the pipeline."
        logger.error(f"[data_validator] {error_msg}")
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, error_message=error_msg, schema_json=""
            ).to_update(messages=[HumanMessage(content=error_msg, name="data_validator")]),
            goto="workflow_controller",
        )

    try:
        validate_schema_contract(schema_data)
    except ValueError as exc:
        error_msg = f"Schema contract violation: {exc}"
        logger.error(f"[data_validator] {error_msg}")
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, error_message=error_msg, schema_json=schema_json
            ).to_update(messages=[HumanMessage(content=error_msg, name="data_validator")]),
            goto="workflow_controller",
        )

    # Build agent messages; on retry, prepend prior rejection feedback so the
    # agent knows what the human reviewer objected to and can try a different approach.
    context_msg = build_data_validator_context(state, schema_json=schema_json, schema_path=schema_path)
    agent_messages: list[HumanMessage] = [context_msg]
    rejection_comment = state.get("dataset_rejection_comment") or ""
    if rejection_comment:
        agent_messages.append(HumanMessage(
            content=(
                f"Your previous attempt was rejected by a human reviewer. "
                f"Their feedback: {rejection_comment}. "
                "Please try a different approach to address this feedback."
            )
        ))

    agent = get_data_agent()
    try:
        result = agent.invoke({"messages": agent_messages})
    except APITimeoutError as exc:
        # Transient LLM timeout (the client already retried max_retries times). Return a
        # soft failure WITHOUT error_message so the workflow_controller re-dispatches the
        # validator within its attempt budget, instead of crashing the whole run.
        logger.warning(f"[data_validator] LLM timed out — routing back for retry: {exc}")
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, schema_json=schema_json
            ).to_update(messages=[HumanMessage(
                content="Data validation timed out; retrying.", name="data_validator"
            )]),
            goto="workflow_controller",
        )
    except Exception as exc:
        # The agent runtime failed to produce a usable result — e.g. the LLM emitted a tool
        # call with malformed/empty JSON arguments (common after a long reasoning trace),
        # which langchain cannot parse and surfaces as a JSONDecodeError. Don't crash the
        # whole run: return a soft failure WITHOUT error_message so the workflow_controller
        # re-dispatches the validator within its attempt budget (mirrors the timeout path).
        logger.warning(
            f"[data_validator] agent invocation failed — routing back for retry: "
            f"{type(exc).__name__}: {exc}"
        )
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, schema_json=schema_json
            ).to_update(messages=[HumanMessage(
                content="Data validation failed to produce a usable result; retrying.",
                name="data_validator",
            )]),
            goto="workflow_controller",
        )
    final_message = result["messages"][-1].content

    quality_report: dict[str, Any] = extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict[str, Any] = extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict[str, Any] = extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict[str, Any] = extract_tool_json(result["messages"], "impute_missing_values")
    join_exec_result: dict[str, Any] = extract_tool_json(result["messages"], "execute_join_plan")
    eval_result: dict[str, Any] = extract_tool_json(result["messages"], "evaluate_join_candidates")

    data_join_plan = join_exec_result.get("join_plan")
    data_join_evaluations = eval_result.get("evaluations", [])

    data_join_base_nrows: int | None = None
    if data_join_plan:
        base_name = data_join_plan.get("base_dataset", {}).get("dataset_name")
        if base_name:
            raw_paths = {Path(p).stem: p for p in (state.get("dataset_paths") or [])}
            base_path = raw_paths.get(base_name)
            if base_path and Path(base_path).exists():
                with contextlib.suppress(Exception):
                    data_join_base_nrows = len(pd.read_csv(base_path))

    processed_path = (
        imputation_result.get("output_path", "")
        or mapping_result.get("output_path", "")
        or validation_result.get("output_path", "")
    )
    validation_passed = bool(validation_result.get("passed", False))

    dataset_summary: dict[str, Any] = {}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            dataset_summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
            }
        except Exception:
            pass

    problem_type: str = schema_data.get("problem_type", "")
    # Identifier columns (schema unique:true) are row keys, not predictors. Exclude the
    # structural columns (target / datetime / series id) so the datetime index is never
    # treated as a droppable id. The executor drops these from tabular feature matrices.
    _structural = {
        schema_data.get("target_column", ""),
        schema_data.get("datetime_column", ""),
        *(schema_data.get("series_id_columns", []) or []),
    }
    id_columns = [
        c["name"] for c in schema_data.get("columns", [])
        if c.get("unique") is True and c.get("name") not in _structural
    ]
    task_metadata: dict[str, Any] = {
        "target_column": schema_data.get("target_column", ""),
        "name": schema_data.get("name", "unknown"),
        "id_columns": id_columns,
    }
    if problem_type == "forecasting":
        task_metadata.update({
            "datetime_column": schema_data.get("datetime_column", ""),
            "series_id_columns": schema_data.get("series_id_columns", []),
            "forecast_horizon": schema_data.get("forecast_horizon"),
            "frequency": schema_data.get("frequency", ""),
            "exogenous_columns": schema_data.get("exogenous_columns"),
        })

    error_message = (
        "" if validation_passed
        else f"Data validation failed after auto-fix attempt: {final_message}"
    )
    if not validation_passed:
        logger.warning("[data_validator] validation failed — aborting without HITL")

    output = DataValidationStateUpdate(
        validation_report=quality_report,
        validation_passed=validation_passed,
        processed_dataset_path=processed_path,
        dataset_summary=dataset_summary,
        problem_type=problem_type,
        task_metadata=task_metadata,
        schema_json=schema_json,
        data_join_plan=data_join_plan,
        data_join_base_nrows=data_join_base_nrows,
        data_join_evaluations=data_join_evaluations,
        error_message=error_message,
    )
    return Command(
        goto="workflow_controller",
        update=output.to_update(
            messages=[HumanMessage(content=final_message, name="data_validator")]
        ),
    )
