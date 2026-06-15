import yaml
from pathlib import Path

m = yaml.safe_load(Path("scripts/benchmark_manifest.yaml").read_text())
remaining = {
    "bike_sharing_daily", "metro_traffic_volume", "beijing_pm25",
    "appliances_energy", "vic_elec", "sp500_weekly", "oil_weekly",
    "gold_macro_weekly", "crypto_weekly", "fx_weekly",
}
entries = [e for e in m if e.get("dataset_id") in remaining]
Path("scripts/_manifest_remaining.yaml").write_text(yaml.dump(entries, allow_unicode=True))
print(f"Written {len(entries)} entries")
