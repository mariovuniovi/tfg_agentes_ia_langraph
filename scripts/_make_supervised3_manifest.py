import yaml
from pathlib import Path

TARGET = {"bike_sharing_daily", "vic_elec", "appliances_energy"}

m = yaml.safe_load(Path("scripts/benchmark_manifest.yaml").read_text())
entries = [e for e in m if e.get("dataset_id") in TARGET]
Path("scripts/_manifest_supervised3.yaml").write_text(yaml.dump(entries, allow_unicode=True))
print(f"Written {len(entries)} entries: {[e['dataset_id'] for e in entries]}")
