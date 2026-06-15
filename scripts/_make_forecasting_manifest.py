import yaml
from pathlib import Path

m = yaml.safe_load(Path("scripts/benchmark_manifest.yaml").read_text())
entries = [e for e in m if e.get("problem_type") == "forecasting"]
Path("scripts/_manifest_forecasting.yaml").write_text(yaml.dump(entries, allow_unicode=True))
print(f"Written {len(entries)} forecasting entries")
