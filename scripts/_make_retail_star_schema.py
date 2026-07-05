"""Generate a 5-table retail star schema for join-discovery case study.

Tables and join graph:
  sales_transactions.csv  (~500 rows)  — fact table
  dim_customers.csv       (100 rows)   — customer_id  (int FK)
  dim_products.csv         (50 rows)   — product_id   (int FK)
  dim_stores.csv           (20 rows)   — store_id     (int FK)
  dim_brands.csv           (10 rows)   — brand_id     (int FK via dim_products)

Five distinct joins:
  1. sales_transactions.customer_id  → dim_customers.customer_id   (int FK)
  2. sales_transactions.product_id   → dim_products.product_id     (int FK)
  3. sales_transactions.store_id     → dim_stores.store_id         (int FK)
  4. dim_products.brand_id           → dim_brands.brand_id         (int FK, indirect)
  5. dim_customers.city_code         → dim_stores.city_code        (string code, shared attribute)

Join #4 is indirect (products→brands, not through fact table).
Join #5 is a non-obvious geographic match: customers live in the same cities as stores.
No join key is a date column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
Path("data/samples/retail_star_schema").mkdir(parents=True, exist_ok=True)

# ── Dimension: brands (10 rows) ───────────────────────────────────────────

brands = pd.DataFrame({
    "brand_id": range(1, 11),
    "brand_name": [
        "ArcaVerde", "NordPeak", "SolMare", "TerraNova", "Brixon",
        "Lumino", "CedarLine", "Ravello", "Fortico", "Zephyr",
    ],
    "country_of_origin": ["ES", "DE", "ES", "IT", "FR", "DE", "US", "IT", "ES", "FR"],
    "market_segment":    rng.choice(["premium", "mid-range", "budget"], 10,
                                     p=[0.3, 0.5, 0.2]),
    "founded_year":      rng.integers(1950, 2010, 10),
})
brands.to_csv("data/samples/retail_star_schema/dim_brands.csv", index=False)
print(f"dim_brands.csv            :  {len(brands):>4} rows")

# ── Dimension: products (50 rows) ─────────────────────────────────────────

categories   = ["Food", "Electronics", "Clothing", "Home", "Sports"]
product_words = ["Pro", "Ultra", "Eco", "Max", "Lite", "Prime", "Select", "Pure"]
product_names = [
    f"{rng.choice(['Alpha', 'Bravo', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet', 'Kilo'])} "
    f"{rng.choice(product_words)}"
    for _ in range(50)
]
products = pd.DataFrame({
    "product_id":   range(1, 51),
    "product_name": product_names,
    "brand_id":     rng.integers(1, 11, 50),           # FK → dim_brands
    "category":     rng.choice(categories, 50),
    "subcategory":  rng.choice(["Standard", "Premium", "Budget"], 50),
    "base_price":   rng.uniform(4.99, 499.99, 50).round(2),
    "weight_kg":    rng.uniform(0.1, 15.0, 50).round(2),
})
products.to_csv("data/samples/retail_star_schema/dim_products.csv", index=False)
print(f"dim_products.csv          :  {len(products):>4} rows")

# ── Dimension: stores (20 rows) ───────────────────────────────────────────

city_codes  = ["MAD", "BCN", "VAL", "SEV", "BIL", "ZGZ", "MLG", "ALC"]
city_names  = {
    "MAD": "Madrid", "BCN": "Barcelona", "VAL": "Valencia",
    "SEV": "Sevilla", "BIL": "Bilbao",   "ZGZ": "Zaragoza",
    "MLG": "Malaga",  "ALC": "Alicante",
}
regions = {
    "MAD": "Centro", "BCN": "Este", "VAL": "Este", "SEV": "Sur",
    "BIL": "Norte",  "ZGZ": "Centro", "MLG": "Sur", "ALC": "Este",
}
store_city_codes = rng.choice(city_codes, 20)
formats = rng.choice(["express", "supermarket", "hypermarket"], 20, p=[0.4, 0.4, 0.2])
stores = pd.DataFrame({
    "store_id":    range(1, 21),
    "store_name":  [f"{city_names[c]} {fmt.capitalize()} {i+1}"
                    for i, (c, fmt) in enumerate(zip(store_city_codes, formats, strict=False))],
    "city_code":   store_city_codes,                   # shared attr → dim_customers
    "region":      [regions[c] for c in store_city_codes],
    "format":      formats,
    "sq_meters":   rng.integers(150, 5_001, 20),
    "open_year":   rng.integers(2000, 2022, 20),
})
stores.to_csv("data/samples/retail_star_schema/dim_stores.csv", index=False)
print(f"dim_stores.csv            :  {len(stores):>4} rows")

# ── Dimension: customers (100 rows) ───────────────────────────────────────

first_names = ["Ana", "Luis", "María", "Carlos", "Elena", "Pedro", "Laura",
               "Javier", "Carmen", "Miguel", "Isabel", "David", "Sofía",
               "Jorge", "Nuria", "Álvaro", "Lucía", "Raúl", "Marta", "Pablo"]
last_names  = ["García", "Martínez", "López", "Sánchez", "González", "Fernández",
               "Rodríguez", "Pérez", "Romero", "Torres", "Díaz", "Vázquez",
               "Moreno", "Muñoz", "Alonso", "Ruiz", "Navarro", "Molina", "Jiménez", "Iglesias"]

customer_cities = rng.choice(city_codes, 100)     # overlaps with store city_codes (join #5)
customers = pd.DataFrame({
    "customer_id":  range(1, 101),
    "full_name":    [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(100)],
    "age_group":    rng.choice(["18-25", "26-35", "36-50", "51+"], 100, p=[0.15, 0.3, 0.35, 0.2]),
    "city_code":    customer_cities,               # FK-like → dim_stores.city_code (shared attr)
    "loyalty_tier": rng.choice(["bronze", "silver", "gold", "platinum"], 100,
                                p=[0.4, 0.35, 0.2, 0.05]),
    "since_year":   rng.integers(2018, 2024, 100),
    "email_domain": rng.choice(["gmail.com", "hotmail.com", "empresa.es", "outlook.com"], 100),
})
customers.to_csv("data/samples/retail_star_schema/dim_customers.csv", index=False)
print(f"dim_customers.csv         :  {len(customers):>4} rows")

# ── Fact: sales_transactions (~500 rows) ──────────────────────────────────

n = 500
cust_ids    = rng.integers(1, 101, n)
prod_ids    = rng.integers(1, 51,  n)
store_ids   = rng.integers(1, 21,  n)
sale_dates  = pd.date_range("2023-01-01", "2024-12-31", periods=n).normalize()
quantities  = rng.integers(1, 11, n)
base_prices = products.set_index("product_id")["base_price"]
unit_prices = np.array([base_prices[pid] for pid in prod_ids])
discounts   = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
totals      = (quantities * unit_prices * (1 - discounts)).round(2)
channels    = rng.choice(["in-store", "online", "click-and-collect"], n, p=[0.55, 0.30, 0.15])

transactions = pd.DataFrame({
    "transaction_id": [f"T{i:04d}" for i in range(1, n + 1)],
    "customer_id":    cust_ids,              # FK → dim_customers
    "product_id":     prod_ids,              # FK → dim_products
    "store_id":       store_ids,             # FK → dim_stores
    "sale_date":      sale_dates.strftime("%Y-%m-%d"),
    "quantity":       quantities,
    "unit_price":     unit_prices,
    "discount_rate":  discounts,
    "total_amount":   totals,
    "channel":        channels,
})
transactions.to_csv("data/samples/retail_star_schema/sales_transactions.csv", index=False)
print(f"sales_transactions.csv    :  {len(transactions):>4} rows")

# ── Summary ───────────────────────────────────────────────────────────────

print("""
Join map (5 joins, no date keys):
  [1] sales_transactions.customer_id  --> dim_customers.customer_id   (int FK, 100 distinct)
  [2] sales_transactions.product_id   --> dim_products.product_id     (int FK, 50 distinct)
  [3] sales_transactions.store_id     --> dim_stores.store_id         (int FK, 20 distinct)
  [4] dim_products.brand_id           --> dim_brands.brand_id         (int FK, indirect, 10 distinct)
  [5] dim_customers.city_code         --> dim_stores.city_code        (string code, shared attr, 8 cities)
""")
