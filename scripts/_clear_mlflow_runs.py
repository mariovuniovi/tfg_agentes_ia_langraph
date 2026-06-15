"""One-off: archive the old mlops-agents experiment so a fresh, empty one is created.

The promotion gate (_fetch_current_champion) ranks runs by the logged `rmse`. Old runs
hold VALIDATION-based rmse; new runs hold honest TEST rmse. Renaming the old experiment
to an archive name lets the app auto-create a fresh empty `mlops-agents` on the next run,
so the gate compares test-vs-test from a clean baseline. Old runs/artifacts are preserved
under the archive name (just out of the gate's view, which searches by name 'mlops-agents').
"""
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

ARCHIVE = "mlops-agents-archived-validation"

c = MlflowClient("http://127.0.0.1:5000")
e = next((x for x in c.search_experiments(view_type=ViewType.ALL) if x.name == "mlops-agents"), None)
if e is None:
    print("no experiment named mlops-agents (a fresh one will be created on next run)")
else:
    if e.lifecycle_stage == "deleted":
        c.restore_experiment(e.experiment_id)  # must be active to rename
    c.rename_experiment(e.experiment_id, ARCHIVE)
    print(f"renamed experiment {e.experiment_id}: 'mlops-agents' -> '{ARCHIVE}'")
    print("a fresh empty 'mlops-agents' will be created on your next pipeline run")
