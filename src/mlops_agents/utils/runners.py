"""Entry-point runners for CLI scripts."""


def run_pipeline() -> None:
    from mlops_agents.graphs.mlops_graph import main
    main()
