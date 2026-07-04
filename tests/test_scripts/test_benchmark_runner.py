"""Smoke test for the benchmark runner — iris classification only (fast)."""
import yaml
import pytest
from mlops_agents.experience.pool import ExperiencePool


@pytest.mark.slow
def test_benchmark_runner_iris_smoke(tmp_path, monkeypatch):
    """Run the benchmark runner on iris only and verify the pool is populated."""
    from mlops_agents.config.settings import settings as _settings
    import mlops_agents.training.executor as _executor

    monkeypatch.setattr(_settings, "experience_db_path", tmp_path / "bench.db")
    monkeypatch.setattr(_settings, "experience_audit_dir", tmp_path / "pool")
    monkeypatch.setattr(_executor.settings, "experience_pool_dir", tmp_path / "pool")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.dump([{
        "dataset_id": "iris",
        "source": "sklearn",
        "source_id": "load_iris",
        "problem_type": "classification",
        "target_column": "target",
    }]))

    from scripts.run_benchmark import run_benchmark
    n_ok, n_fail = run_benchmark(
        manifest_path=manifest_path,
        db_path=tmp_path / "bench.db",
        audit_dir=tmp_path / "pool",
        splits_dir=tmp_path / "splits",
        staged_dir=tmp_path / "staged",
    )

    assert n_ok == 1
    assert n_fail == 0
    pool = ExperiencePool(tmp_path / "bench.db")
    assert pool.count("classification") >= 1
