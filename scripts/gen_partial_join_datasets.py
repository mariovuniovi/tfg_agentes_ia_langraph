"""Generate raw datasets to stress-test join discovery with NON-trivial,
partial-coverage FK->PK relationships (orphan FKs and FK-subset-of-PK).

Run: uv run python scripts/gen_partial_join_datasets.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = Path("data/samples/join_partial_coverage")
OUT.mkdir(parents=True, exist_ok=True)

N = 320

# ── Base fact table: regression on revenue_eur ──────────────────────────────
# FK columns reference three dimension tables with DIFFERENT key names.
products = [f"SKU{i:03d}" for i in range(1, 41)]        # base uses 40 SKUs
stores = [f"ST{i:02d}" for i in range(1, 31)]           # base references 30 stores
segments = ["A", "B", "C", "D"]                          # base uses 4 segments

base = pd.DataFrame({
    "txn_id": range(1, N + 1),
    "product_id": rng.choice(products, N),              # -> product_catalog.sku
    "store_id":   rng.choice(stores, N),                # -> store_directory.store_code
    "segment_code": rng.choice(segments, N),            # -> customer_segments.segment_code
    "units": rng.integers(1, 20, N),
    "promo_flag": rng.integers(0, 2, N),
})
base["revenue_eur"] = (base["units"] * rng.uniform(8, 90, N)).round(2)
base.to_csv(OUT / "retail_sales.csv", index=False)

# ── Dim 1: product_catalog — FK SUBSET OF PK (clean) ────────────────────────
# 60 SKUs in catalog, base only references 40 -> right_coverage < 100%,
# left_coverage = 100%. Key name differs (sku vs product_id). many_to_one.
catalog = pd.DataFrame({
    "sku": [f"SKU{i:03d}" for i in range(1, 61)],
    "category": rng.choice(["bakery", "dairy", "drinks", "produce", "frozen"], 60),
    "unit_price_eur": rng.uniform(1, 100, 60).round(2),
    "brand": rng.choice(["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli"], 60),
})
catalog.to_csv(OUT / "product_catalog.csv", index=False)

# ── Dim 2: store_directory — ORPHAN FKs (the hard case) ─────────────────────
# Directory only has ST01..ST22; base references ST01..ST30 -> 8 orphan stores
# -> left_coverage = 22/30 = 73% (< 0.80 threshold). Key name differs.
# Plus a DUPLICATE PK row (ST05) to exercise the dedup path.
directory = pd.DataFrame({
    "store_code": [f"ST{i:02d}" for i in range(1, 23)],
    "region": rng.choice(["north", "south", "east", "west"], 22),
    "store_size_sqm": rng.integers(200, 2000, 22),
    "store_type": rng.choice(["flagship", "standard", "express"], 22),
})
dup = directory[directory["store_code"] == "ST05"].copy()
dup["region"] = "east"  # conflicting attribute on the duplicate key
directory = pd.concat([directory, dup], ignore_index=True)
directory.to_csv(OUT / "store_directory.csv", index=False)

# ── Dim 3: customer_segments — FK SUBSET OF PK (small categorical) ───────────
# 6 segments A..F; base uses only A..D -> right_coverage = 4/6, left = 100%.
seg = pd.DataFrame({
    "segment_code": ["A", "B", "C", "D", "E", "F"],
    "segment_name": ["budget", "mainstream", "premium", "luxury", "wholesale", "online"],
    "avg_income_eur": [18000, 32000, 55000, 90000, 120000, 40000],
})
seg.to_csv(OUT / "customer_segments.csv", index=False)

print("Wrote:")
for p in sorted(OUT.glob("*.csv")):
    print(f"  {p}  ({len(pd.read_csv(p))} rows)")
