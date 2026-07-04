"""CLI entry point — runs the full MLOps pipeline with interactive HITL approval.

Run with:
    uv run python scripts/run_pipeline.py
or via the installed console script:
    mlops-pipeline
"""

from typing import Any

from langchain_core.messages import HumanMessage

from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
from mlops_agents.graphs.mlops_graph import graph


def main() -> None:
    """Run the full MLOps pipeline from the CLI, including HITL approval."""
    import sys

    dataset_paths = sys.argv[1:] if len(sys.argv) > 1 else ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    paths_display = ", ".join(dataset_paths)

    config = {"configurable": {"thread_id": "pipeline-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    initial_state: dict = {
        "messages": [
            HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")
        ],
        "dataset_paths": dataset_paths,
        "processed_dataset_path": "",
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": None,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
        "schema_json": "",
        "dataset_approved": None,
        "dataset_rejection_comment": "",
        "deployment_approved": None,
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "evaluation_report_audit": None,
        "evaluation_report_audit_status": "",
    }

    print(f"\n{'='*60}")
    print(f"MLOps Pipeline — files: {paths_display}")
    print(f"{'='*60}\n")

    for event in graph.stream(initial_state, config=config):
        if "__interrupt__" in event:
            interrupt_value = event["__interrupt__"][0].value
            _handle_hitl(graph, config, interrupt_value)
        else:
            for node_name in event:
                print(f"  [{node_name}] completed")

    print(f"\n{'='*60}")
    print("Pipeline finished.")
    print(f"{'='*60}\n")


def _handle_hitl(graph: Any, config: dict[str, Any], interrupt_value: dict[str, Any]) -> None:
    """Prompt the operator for HITL approval and resume the graph."""
    from langgraph.types import Command

    print(f"\n{'='*60}")
    print("HUMAN APPROVAL REQUIRED")
    print(f"{'='*60}")
    print(interrupt_value.get("question", "Approve this action?"))
    summary = interrupt_value.get("registration_summary", "")
    if summary:
        print(f"\nDetails:\n{summary}")
    print(f"{'='*60}")

    raw = input("\nApprove? (y/n): ").strip().lower()
    approved = raw == "y"
    resume: dict[str, Any] = {"approved": approved}
    if not approved:
        reason = input("Rejection reason (optional, press Enter to skip): ").strip()
        resume["reason"] = reason or "Rejected by operator"

    print()
    for event in graph.stream(Command(resume=resume), config=config):
        for node_name in event:
            print(f"  [{node_name}] completed")


if __name__ == "__main__":
    main()
