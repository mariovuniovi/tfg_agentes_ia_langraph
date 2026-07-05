"""Generate benchmark forecasting CSVs for the SP4 benchmark runner.

Synthetic (no internet): air_passengers, m4_monthly_sample
Real financial (requires internet + yfinance): gold_macro_monthly,
    commodity_panel_weekly, crypto_weekly, fx_exog_weekly

Run:
    uv run python scripts/_generate_benchmarks.py

Each dataset is designed to exercise a specific rule in ml_rules.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/benchmarks")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. AirPassengers — univariate monthly, 144 obs, strong trend + seasonality
#    Exercises: forecasting_strong_seasonality_prefers_seasonal_models
#               forecasting_single_series_no_exogenous_prefers_statistical
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
dates = pd.date_range("1949-01-01", periods=144, freq="MS")
trend = np.arange(144) * 1.8
seasonal = 60 * np.sin(np.arange(144) * 2 * np.pi / 12)
noise = rng.normal(scale=8, size=144)
pd.DataFrame({"month": dates, "passengers": np.round(112 + trend + seasonal + noise, 1)}) \
    .to_csv(OUT / "air_passengers.csv", index=False)
print("OK air_passengers.csv")

# ---------------------------------------------------------------------------
# 2. M4 monthly panel — 5 series × 120 obs, no exogenous
#    Exercises: forecasting_many_series_prefers_global_ml
#               forecasting_short_history_prefers_statistical (varies by series)
# ---------------------------------------------------------------------------
rows = []
for i, sid in enumerate(["M1", "M2", "M3", "M4", "M5"]):
    base = 200 + i * 50
    t = np.arange(120)
    y = base + t * 0.5 + 30 * np.sin(t * 2 * np.pi / 12) + rng.normal(scale=10, size=120)
    for d, v in zip(pd.date_range("2000-01-01", periods=120, freq="MS"), y, strict=False):
        rows.append({"series_id": sid, "date": d, "value": round(v, 2)})
pd.DataFrame(rows).to_csv(OUT / "m4_monthly_sample.csv", index=False)
print("OK m4_monthly_sample.csv")

# ---------------------------------------------------------------------------
# Real financial data via yfinance
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "yfinance not installed. Run:  uv add yfinance  or  pip install yfinance"
    ) from exc


def _fetch(ticker: str, start: str, end: str, interval: str) -> pd.Series:
    """Download Close price series, strip timezone, forward-fill."""
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker!r}")
    s = raw["Close"].squeeze()
    s.index = s.index.tz_localize(None).normalize()  # strip tz, keep date
    return s.ffill().dropna()


# ---------------------------------------------------------------------------
# 3. Gold with macro exogenous — monthly, ~9 years
#    Target: GLD (gold ETF price)
#    Exog:   USO (oil), SLV (silver), SPY (S&P 500), UUP (USD index)
#    Exercises: forecasting_long_history_with_exogenous_prefers_supervised
#               (single series, medium-long history, exogenous=True)
# ---------------------------------------------------------------------------
START_MACRO, END_MACRO = "2015-01-01", "2024-06-01"
print("Downloading gold_macro_monthly …")
tickers_macro = {
    "gold_close": "GLD",
    "oil_close":   "USO",
    "silver_close": "SLV",
    "sp500_close":  "SPY",
    "usd_close":    "UUP",
}
frames = {}
for col, ticker in tickers_macro.items():
    frames[col] = _fetch(ticker, START_MACRO, END_MACRO, "1mo")

df_macro = pd.DataFrame(frames).dropna()
df_macro.index.name = "date"
df_macro = df_macro.reset_index()
df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.to_period("M").dt.to_timestamp()
df_macro = df_macro.round(2)
df_macro.to_csv(OUT / "gold_macro_monthly.csv", index=False)
print(f"OK gold_macro_monthly.csv  ({len(df_macro)} rows, {len(df_macro.columns)-1} features)")

# ---------------------------------------------------------------------------
# 4. Commodity panel — weekly, ~6 years, 5 commodities as separate series
#    Target: close_price    Series: Gold, Oil, Silver, Copper, Wheat
#    Exercises: forecasting_many_series_prefers_global_ml (moderate series count)
#               forecasting_short_history_prefers_statistical (each series, weekly)
# ---------------------------------------------------------------------------
START_PANEL, END_PANEL = "2018-01-01", "2024-06-01"
PANEL_TICKERS = {
    "Gold":   "GLD",
    "Oil":    "USO",
    "Silver": "SLV",
    "Copper": "CPER",
    "Wheat":  "WEAT",
}
print("Downloading commodity_panel_weekly …")
panel_rows = []
for name, ticker in PANEL_TICKERS.items():
    s = _fetch(ticker, START_PANEL, END_PANEL, "1wk")
    for d, v in s.items():
        panel_rows.append({"commodity": name, "week": d.strftime("%Y-%m-%d"), "close_price": round(float(v), 4)})

df_panel = pd.DataFrame(panel_rows)
df_panel.to_csv(OUT / "commodity_panel_weekly.csv", index=False)
print(f"OK commodity_panel_weekly.csv  ({len(df_panel)} rows, 5 commodities)")

# ---------------------------------------------------------------------------
# 5. Bitcoin weekly — strongly trending, non-stationary, high volatility
#    Target: btc_close
#    Exercises: forecasting_non_stationary_prefers_differencing_models
#               forecasting_single_series_no_exogenous_prefers_statistical
# ---------------------------------------------------------------------------
START_CRYPTO, END_CRYPTO = "2018-01-01", "2024-06-01"
print("Downloading crypto_weekly …")
btc = _fetch("BTC-USD", START_CRYPTO, END_CRYPTO, "1wk")
df_crypto = btc.rename("btc_close").reset_index()
df_crypto.columns = ["week", "btc_close"]
df_crypto["week"] = pd.to_datetime(df_crypto["week"]).dt.strftime("%Y-%m-%d")
df_crypto["btc_close"] = df_crypto["btc_close"].round(2)
df_crypto.to_csv(OUT / "crypto_weekly.csv", index=False)
print(f"OK crypto_weekly.csv  ({len(df_crypto)} rows)")

# ---------------------------------------------------------------------------
# 6. FX with exogenous — EUR/USD weekly, multiple correlated FX + Gold as exog
#    Target: eurusd_close
#    Exog:   gbpusd_close, jpyusd_close (inverse of JPY), gold_close
#    Exercises: forecasting_long_history_with_exogenous_prefers_supervised
#               (single FX series, exogenous=True — different domain from gold_macro)
# ---------------------------------------------------------------------------
START_FX, END_FX = "2018-01-01", "2024-06-01"
print("Downloading fx_exog_weekly …")
fx_tickers = {
    "eurusd_close": "EURUSD=X",
    "gbpusd_close": "GBPUSD=X",
    "jpyusd_close": "JPY=X",   # JPY per USD — we keep as-is (inverse convention)
    "gold_close":   "GLD",
}
fx_frames = {col: _fetch(t, START_FX, END_FX, "1wk") for col, t in fx_tickers.items()}
df_fx = pd.DataFrame(fx_frames).dropna()
df_fx.index.name = "week"
df_fx = df_fx.reset_index()
df_fx["week"] = pd.to_datetime(df_fx["week"]).dt.strftime("%Y-%m-%d")
df_fx = df_fx.round(4)
df_fx.to_csv(OUT / "fx_exog_weekly.csv", index=False)
print(f"OK fx_exog_weekly.csv  ({len(df_fx)} rows, 3 exogenous features)")

print("\nAll benchmark CSVs generated successfully in data/benchmarks/")
