"""Optional training sub-workflow for iterative retrain loops.

This subgraph can be used in place of the single trainer_node when
the supervisor determines that multiple training iterations are needed
(e.g., first model doesn't meet evaluation criteria).

Not wired into the main graph by default — add when needed.
"""

# TODO: implement if iterative training loop is required
