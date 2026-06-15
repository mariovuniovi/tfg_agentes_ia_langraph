import yaml
from pathlib import Path

m = yaml.safe_load(Path("scripts/benchmark_manifest.yaml").read_text())
entry = next(e for e in m if e.get("dataset_id") == "vic_elec")
Path("scripts/_manifest_vic_elec.yaml").write_text(yaml.dump([entry]))
print("Written 1 entry")
