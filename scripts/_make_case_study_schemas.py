"""Generate schema JSON files for all Chapter 6 case-study datasets.

Writes one *_schema.json next to each CSV in:
  data/samples/size_test/
  data/samples/join_discovery/
  data/samples/retail_star_schema/
  data/samples/broken/
"""
from __future__ import annotations

import json
from pathlib import Path


def write(path: str, schema: dict) -> None:
    Path(path).write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {path}")

print("=== size_test ===")

write("data/samples/size_test/small_monthly_revenue_schema.json", {
    "problem_type": "forecasting",
    "name": "small_monthly_revenue",
    "description": "10 years of monthly retail store revenue (2014–2023). ~120 rows. Used as the SMALL dataset in the validator size stress-test.",
    "target_column": "revenue_eur",
    "datetime_column": "month",
    "forecast_horizon": 12,
    "frequency": "MS",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month (YYYY-MM-DD).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "revenue_eur",
            "dtype": "int",
            "description": "Total monthly revenue in euros.",
            "required": True, "nullable": False, "min": 0
        }
    ]
})

write("data/samples/size_test/medium_daily_bakery_schema.json", {
    "problem_type": "forecasting",
    "name": "medium_daily_bakery",
    "description": "Daily units sold by a bakery chain over ~1,000 days (2021–2023). Trend + weekly seasonality. Used as the MEDIUM dataset in the validator size stress-test.",
    "target_column": "units_sold",
    "datetime_column": "date",
    "forecast_horizon": 14,
    "frequency": "D",
    "exogenous_columns": [
        {"name": "is_weekend",   "future_availability": "known_future"},
        {"name": "day_of_week",  "future_availability": "known_future"}
    ],
    "columns": [
        {
            "name": "date",
            "dtype": "datetime",
            "description": "Calendar date (YYYY-MM-DD).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "units_sold",
            "dtype": "int",
            "description": "Total bread/pastry units sold that day.",
            "required": True, "nullable": False, "min": 0
        },
        {
            "name": "is_weekend",
            "dtype": "int",
            "description": "1 if Saturday or Sunday, 0 otherwise.",
            "required": True, "nullable": False,
            "allowed_values": [0, 1]
        },
        {
            "name": "day_of_week",
            "dtype": "int",
            "description": "ISO weekday index: 0 = Monday … 6 = Sunday.",
            "required": True, "nullable": False, "min": 0, "max": 6
        }
    ]
})

write("data/samples/size_test/large_hourly_factory_schema.json", {
    "problem_type": "forecasting",
    "name": "large_hourly_factory",
    "description": "Hourly electricity consumption of a manufacturing plant over ~14 months (~10,000 rows). Daily and weekly cycles plus a slow upward trend. Used as the LARGE dataset in the validator size stress-test.",
    "target_column": "consumption_kwh",
    "datetime_column": "timestamp",
    "forecast_horizon": 24,
    "frequency": "H",
    "exogenous_columns": [
        {"name": "temperature_c", "future_availability": "unknown_future"},
        {"name": "is_weekend",    "future_availability": "known_future"},
        {"name": "hour_of_day",   "future_availability": "known_future"}
    ],
    "columns": [
        {
            "name": "timestamp",
            "dtype": "datetime",
            "description": "UTC timestamp at the start of the hour (YYYY-MM-DD HH:MM:SS).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "consumption_kwh",
            "dtype": "float",
            "description": "Total electricity consumed by the plant during that hour in kilowatt-hours.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "temperature_c",
            "dtype": "float",
            "description": "Outdoor temperature at the factory site in degrees Celsius.",
            "required": True, "nullable": False
        },
        {
            "name": "hour_of_day",
            "dtype": "int",
            "description": "Hour of day (0–23).",
            "required": True, "nullable": False, "min": 0, "max": 23
        },
        {
            "name": "is_weekend",
            "dtype": "int",
            "description": "1 if Saturday or Sunday, 0 otherwise.",
            "required": True, "nullable": False,
            "allowed_values": [0, 1]
        }
    ]
})

print("\n=== join_discovery ===")

write("data/samples/join_discovery/air_passengers_base_schema.json", {
    "problem_type": "forecasting",
    "name": "air_passengers_base",
    "description": "Monthly international airline passengers 1949–1960 (Box & Jenkins). This is the base series — upload alongside aviation_context.csv to trigger join discovery on the 'month' key. When merged, the combined dataset exactly matches the pool entry for air_passengers.",
    "target_column": "passengers",
    "datetime_column": "month",
    "forecast_horizon": 12,
    "frequency": "MS",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month (YYYY-MM-DD).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "passengers",
            "dtype": "int",
            "description": "Number of international airline passengers (thousands).",
            "required": True, "nullable": False, "min": 0
        }
    ]
})

write("data/samples/join_discovery/aviation_context_schema.json", {
    "name": "aviation_context",
    "description": "Synthetic monthly aviation context variables (1949–1960) joinable to air_passengers_base.csv on 'month'. Provides exogenous features: fuel price index and average load factor.",
    "join_key": "month",
    "joins_to": "air_passengers_base.csv",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month (YYYY-MM-DD). Primary join key with air_passengers_base.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "fuel_price_idx",
            "dtype": "float",
            "description": "Synthetic aviation fuel price index (base 80, 1949). Proxy for operating cost pressure.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "avg_load_factor_pct",
            "dtype": "float",
            "description": "Average seat occupancy rate across all routes, as a percentage (45–95).",
            "required": True, "nullable": False, "min": 0.0, "max": 100.0
        }
    ]
})

write("data/samples/join_discovery/product_catalog_schema.json", {
    "name": "product_catalog",
    "description": "Static product reference table — 50 products with pricing and inventory. Contains NO date or temporal column. Used as a negative control in the join-discovery test: the data_validator should return zero join candidates for this table.",
    "columns": [
        {
            "name": "product_id",
            "dtype": "str",
            "description": "Alphanumeric product identifier (P001–P050).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "product_name",
            "dtype": "str",
            "description": "Human-readable product name.",
            "required": True, "nullable": False
        },
        {
            "name": "category",
            "dtype": "str",
            "description": "Product category.",
            "required": True, "nullable": False,
            "allowed_values": ["Electronics", "Food", "Clothing", "Tools", "Sports"]
        },
        {
            "name": "unit_price_eur",
            "dtype": "float",
            "description": "Retail unit price in euros.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "stock_units",
            "dtype": "int",
            "description": "Current warehouse stock in units.",
            "required": True, "nullable": False, "min": 0
        },
        {
            "name": "supplier_code",
            "dtype": "str",
            "description": "Internal supplier reference code (e.g. SUP-42).",
            "required": True, "nullable": False
        }
    ]
})

print("\n=== retail_star_schema ===")

write("data/samples/retail_star_schema/sales_transactions_schema.json", {
    "problem_type": "regression",
    "name": "sales_transactions",
    "description": "Retail point-of-sale transactions (500 rows, 2023–2024). Fact table in a 5-table star schema. Joinable to dim_customers (customer_id), dim_products (product_id), dim_stores (store_id). Used to stress-test join discovery across diverse key types.",
    "target_column": "total_amount",
    "columns": [
        {
            "name": "transaction_id",
            "dtype": "str",
            "description": "Unique transaction identifier (T0001–T0500).",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "customer_id",
            "dtype": "int",
            "description": "FK → dim_customers.customer_id (1–100).",
            "required": True, "nullable": False
        },
        {
            "name": "product_id",
            "dtype": "int",
            "description": "FK → dim_products.product_id (1–50).",
            "required": True, "nullable": False
        },
        {
            "name": "store_id",
            "dtype": "int",
            "description": "FK → dim_stores.store_id (1–20).",
            "required": True, "nullable": False
        },
        {
            "name": "sale_date",
            "dtype": "datetime",
            "description": "Date of the transaction (YYYY-MM-DD). Not a join key.",
            "required": True, "nullable": False
        },
        {
            "name": "quantity",
            "dtype": "int",
            "description": "Number of units purchased.",
            "required": True, "nullable": False, "min": 1
        },
        {
            "name": "unit_price",
            "dtype": "float",
            "description": "Price per unit at time of sale in euros.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "discount_rate",
            "dtype": "float",
            "description": "Fractional discount applied (0.0 = no discount, 0.20 = 20% off).",
            "required": True, "nullable": False, "min": 0.0, "max": 1.0
        },
        {
            "name": "total_amount",
            "dtype": "float",
            "description": "Final transaction value in euros: quantity × unit_price × (1 − discount_rate). Target column.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "channel",
            "dtype": "str",
            "description": "Sales channel through which the purchase was made.",
            "required": True, "nullable": False,
            "allowed_values": ["in-store", "online", "click-and-collect"]
        }
    ]
})

write("data/samples/retail_star_schema/dim_customers_schema.json", {
    "name": "dim_customers",
    "description": "Customer dimension table (100 rows). Joins to sales_transactions on customer_id (int FK). Also shares city_code with dim_stores — a non-obvious geographic link used as join #5.",
    "joins": [
        {"key": "customer_id", "to": "sales_transactions.customer_id", "type": "int_fk"},
        {"key": "city_code",   "to": "dim_stores.city_code",           "type": "shared_attribute"}
    ],
    "columns": [
        {
            "name": "customer_id",
            "dtype": "int",
            "description": "Unique customer identifier. PK.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "full_name",
            "dtype": "str",
            "description": "Customer full name.",
            "required": True, "nullable": False
        },
        {
            "name": "age_group",
            "dtype": "str",
            "description": "Age bracket of the customer.",
            "required": True, "nullable": False,
            "allowed_values": ["18-25", "26-35", "36-50", "51+"]
        },
        {
            "name": "city_code",
            "dtype": "str",
            "description": "3-letter city code of the customer's home city (MAD, BCN, VAL, SEV, BIL, ZGZ, MLG, ALC). Shared with dim_stores.city_code.",
            "required": True, "nullable": False
        },
        {
            "name": "loyalty_tier",
            "dtype": "str",
            "description": "Customer loyalty programme tier.",
            "required": True, "nullable": False,
            "allowed_values": ["bronze", "silver", "gold", "platinum"]
        },
        {
            "name": "since_year",
            "dtype": "int",
            "description": "Year the customer first registered.",
            "required": True, "nullable": False
        },
        {
            "name": "email_domain",
            "dtype": "str",
            "description": "Email provider domain of the customer.",
            "required": True, "nullable": False
        }
    ]
})

write("data/samples/retail_star_schema/dim_products_schema.json", {
    "name": "dim_products",
    "description": "Product dimension table (50 rows). Joins to sales_transactions on product_id (int FK). Also joins to dim_brands on brand_id (int FK) — an indirect join not present in the fact table.",
    "joins": [
        {"key": "product_id", "to": "sales_transactions.product_id", "type": "int_fk"},
        {"key": "brand_id",   "to": "dim_brands.brand_id",           "type": "int_fk_indirect"}
    ],
    "columns": [
        {
            "name": "product_id",
            "dtype": "int",
            "description": "Unique product identifier. PK.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "product_name",
            "dtype": "str",
            "description": "Human-readable product name.",
            "required": True, "nullable": False
        },
        {
            "name": "brand_id",
            "dtype": "int",
            "description": "FK → dim_brands.brand_id (1–10).",
            "required": True, "nullable": False
        },
        {
            "name": "category",
            "dtype": "str",
            "description": "Top-level product category.",
            "required": True, "nullable": False,
            "allowed_values": ["Food", "Electronics", "Clothing", "Home", "Sports"]
        },
        {
            "name": "subcategory",
            "dtype": "str",
            "description": "Product tier within the category.",
            "required": True, "nullable": False,
            "allowed_values": ["Standard", "Premium", "Budget"]
        },
        {
            "name": "base_price",
            "dtype": "float",
            "description": "Recommended retail price in euros.",
            "required": True, "nullable": False, "min": 0.0
        },
        {
            "name": "weight_kg",
            "dtype": "float",
            "description": "Product weight in kilograms.",
            "required": True, "nullable": False, "min": 0.0
        }
    ]
})

write("data/samples/retail_star_schema/dim_stores_schema.json", {
    "name": "dim_stores",
    "description": "Store dimension table (20 rows). Joins to sales_transactions on store_id (int FK). Also shares city_code with dim_customers — join #5, a geographic shared-attribute relationship.",
    "joins": [
        {"key": "store_id",  "to": "sales_transactions.store_id", "type": "int_fk"},
        {"key": "city_code", "to": "dim_customers.city_code",     "type": "shared_attribute"}
    ],
    "columns": [
        {
            "name": "store_id",
            "dtype": "int",
            "description": "Unique store identifier. PK.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "store_name",
            "dtype": "str",
            "description": "Human-readable store name including city and format.",
            "required": True, "nullable": False
        },
        {
            "name": "city_code",
            "dtype": "str",
            "description": "3-letter city code where the store is located (MAD, BCN, VAL, SEV, BIL, ZGZ, MLG, ALC). Shared with dim_customers.city_code.",
            "required": True, "nullable": False
        },
        {
            "name": "region",
            "dtype": "str",
            "description": "Spanish geographic region.",
            "required": True, "nullable": False,
            "allowed_values": ["Norte", "Sur", "Este", "Oeste", "Centro"]
        },
        {
            "name": "format",
            "dtype": "str",
            "description": "Store format by size.",
            "required": True, "nullable": False,
            "allowed_values": ["express", "supermarket", "hypermarket"]
        },
        {
            "name": "sq_meters",
            "dtype": "int",
            "description": "Sales floor area in square metres.",
            "required": True, "nullable": False, "min": 100
        },
        {
            "name": "open_year",
            "dtype": "int",
            "description": "Year the store first opened.",
            "required": True, "nullable": False
        }
    ]
})

write("data/samples/retail_star_schema/dim_brands_schema.json", {
    "name": "dim_brands",
    "description": "Brand dimension table (10 rows). Joins to dim_products on brand_id (int FK). This is an indirect join — brand_id does not appear in the fact table, so the agent must traverse products→brands.",
    "joins": [
        {"key": "brand_id", "to": "dim_products.brand_id", "type": "int_fk"}
    ],
    "columns": [
        {
            "name": "brand_id",
            "dtype": "int",
            "description": "Unique brand identifier. PK.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "brand_name",
            "dtype": "str",
            "description": "Commercial brand name.",
            "required": True, "nullable": False
        },
        {
            "name": "country_of_origin",
            "dtype": "str",
            "description": "ISO 2-letter country code of the brand's founding country.",
            "required": True, "nullable": False
        },
        {
            "name": "market_segment",
            "dtype": "str",
            "description": "Price positioning of the brand.",
            "required": True, "nullable": False,
            "allowed_values": ["premium", "mid-range", "budget"]
        },
        {
            "name": "founded_year",
            "dtype": "int",
            "description": "Year the brand was established.",
            "required": True, "nullable": False
        }
    ]
})

print("\n=== broken ===")

write("data/samples/broken/all_nan_target_schema.json", {
    "problem_type": "forecasting",
    "name": "all_nan_target",
    "description": "ERROR SCENARIO: 120-row monthly series where the target column 'revenue' contains entirely NaN values. Expected: data_validator detects 100% missing target and raises a validation error before training is attempted.",
    "target_column": "revenue",
    "datetime_column": "month",
    "forecast_horizon": 12,
    "frequency": "MS",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "revenue",
            "dtype": "float",
            "description": "Monthly revenue in euros. ALL VALUES ARE NaN — this is the injected error.",
            "required": True, "nullable": False
        }
    ]
})

write("data/samples/broken/missing_target_col_schema.json", {
    "problem_type": "forecasting",
    "name": "missing_target_col",
    "description": "ERROR SCENARIO: Schema declares target_column='sales', but the CSV only contains 'month', 'feature_a', 'feature_b'. Expected: data_validator detects the missing column and raises a schema-mismatch error.",
    "target_column": "sales",
    "datetime_column": "month",
    "forecast_horizon": 12,
    "frequency": "MS",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "sales",
            "dtype": "float",
            "description": "Monthly sales volume. COLUMN IS ABSENT FROM THE CSV — this is the injected error.",
            "required": True, "nullable": False
        },
        {
            "name": "feature_a",
            "dtype": "float",
            "description": "Auxiliary numeric feature.",
            "required": False, "nullable": True
        },
        {
            "name": "feature_b",
            "dtype": "int",
            "description": "Auxiliary integer feature.",
            "required": False, "nullable": True
        }
    ]
})

write("data/samples/broken/too_small_for_horizon_schema.json", {
    "problem_type": "forecasting",
    "name": "too_small_for_horizon",
    "description": "ERROR SCENARIO: Only 15 rows of monthly data but forecast_horizon=12. After an 80/20 train-validation split there are ~3 training rows — far too few for any model. Expected: executor or validator detects this and routes to a graceful stop.",
    "target_column": "sales",
    "datetime_column": "month",
    "forecast_horizon": 12,
    "frequency": "MS",
    "columns": [
        {
            "name": "month",
            "dtype": "datetime",
            "description": "First day of the calendar month.",
            "required": True, "nullable": False, "unique": True
        },
        {
            "name": "sales",
            "dtype": "int",
            "description": "Monthly unit sales.",
            "required": True, "nullable": False, "min": 0
        }
    ]
})

print("\nAll schemas written.")
