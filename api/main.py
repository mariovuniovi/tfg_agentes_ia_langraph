"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import runs, experiments, monitoring, uploads

app = FastAPI(title="MLOps Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(experiments.router)
app.include_router(monitoring.router)
app.include_router(uploads.router)


@app.get("/health")
async def health():
    mlflow_ok = True
    graph_ok = True
    try:
        from mlops_agents.config.settings import settings
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments()
    except Exception:
        mlflow_ok = False
    try:
        from mlops_agents.graphs.mlops_graph import graph  # noqa: F401
    except Exception:
        graph_ok = False
    return {"status": "ok", "mlflow": mlflow_ok, "graph": graph_ok}
