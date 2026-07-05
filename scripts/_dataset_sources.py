"""Dataset fetchers for the benchmark runner."""
from __future__ import annotations

from typing import Any

import pandas as pd


def fetch_dataset(entry: dict[str, Any]) -> pd.DataFrame:
    src = entry["source"]
    if src == "sklearn":
        from sklearn import datasets
        loader = getattr(datasets, entry["source_id"])
        bunch = loader()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names).copy()
        df["target"] = bunch.target
        return df
    if src == "openml":
        from sklearn.datasets import fetch_openml
        target_col = entry.get("target_column", "target")
        bunch = fetch_openml(data_id=int(entry["source_id"]), as_frame=True, parser="auto")
        df = bunch.frame.copy() if bunch.frame is not None else pd.DataFrame(bunch.data)
        if target_col not in df.columns:
            if bunch.target is not None:
                df[target_col] = bunch.target.values
            elif len(df.columns) > 0:
                df = df.rename(columns={df.columns[-1]: target_col})
        # OpenML strips string datetime columns; reconstruct from instant (1-based day index)
        dt_col = entry.get("datetime_column")
        if dt_col and dt_col not in df.columns and "instant" in df.columns:
            origin = pd.Timestamp("2011-01-01")
            df = df.sort_values("instant")
            df[dt_col] = (
                origin + pd.to_timedelta(df["instant"].astype(int) - 1, unit="D")
            ).dt.strftime("%Y-%m-%d")
        return df
    if src == "local":
        return pd.read_csv(entry["source_id"])
    if src == "yfinance":
        import yfinance as yf
        ticker = entry["source_id"]
        interval = entry.get("interval", "1wk")
        start = entry.get("start", "2018-01-01")
        end = entry.get("end", "2024-06-01")
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned empty data for {ticker!r}")
        s = raw["Close"].squeeze()
        s.index = s.index.tz_localize(None).normalize()
        df = s.ffill().dropna().reset_index()
        date_col = entry.get("datetime_column", "date")
        target_col = entry.get("target_column", "close")
        df.columns = [date_col, target_col]
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        return df
    if src == "yfinance_multi":
        import yfinance as yf
        interval = entry.get("interval", "1wk")
        start = entry.get("start", "2005-01-01")
        end = entry.get("end", "2024-06-01")
        target_ticker = entry["source_id"]
        exog_tickers: list[str] = entry.get("exog_tickers", [])
        all_tickers = [target_ticker] + exog_tickers
        col_names: dict[str, str] = entry.get("col_names", {})

        frames = {}
        for ticker in all_tickers:
            raw = yf.download(ticker, start=start, end=end, interval=interval,
                              auto_adjust=True, progress=False)
            if raw.empty:
                raise RuntimeError(f"yfinance returned empty data for {ticker!r}")
            s = raw["Close"].squeeze()
            s.index = s.index.tz_localize(None).normalize()
            s = s.ffill().dropna()
            frames[ticker] = s

        df = pd.DataFrame(frames).dropna()
        df.index.name = entry.get("datetime_column", "date")
        df = df.reset_index()

        rename = {entry.get("datetime_column", "date"): entry.get("datetime_column", "date")}
        rename[target_ticker] = col_names.get(target_ticker, entry["target_column"])
        for t in exog_tickers:
            rename[t] = col_names.get(t, t.replace("=", "_").replace("^", "").replace("-", "_").lower())
        df = df.rename(columns=rename)

        dt_col = entry.get("datetime_column", "date")
        df[dt_col] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d")
        return df

    if src == "fred":
        return _fetch_fred(entry)

    if src == "statsmodels":
        return _fetch_statsmodels(entry)

    if src == "uci_url":
        return _fetch_uci_url(entry)

    if src == "dominicks":
        return _fetch_dominicks(entry)

    raise ValueError(f"Unknown source: {src!r}. Valid: sklearn, openml, local, yfinance, yfinance_multi, fred, statsmodels, uci_url, dominicks")


# ---------------------------------------------------------------------------
# fred source — FRED economic series via the keyless public CSV endpoint
# (https://fred.stlouisfed.org/graph/fredgraph.csv?id=...). No API key, no extra
# dependency. Supports one target series plus optional companion series joined on
# the shared date index (for unknown-future exogenous columns).
# ---------------------------------------------------------------------------


def _fetch_fred(entry: dict[str, Any]) -> pd.DataFrame:
    series_ids = str(entry["source_id"])  # e.g. "PCU325325" or "PCU325325,IPG325S"
    colmap = entry.get("fred_columns", {})  # FRED code -> friendly column name
    date_col = entry.get("datetime_column", "date")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_ids}"
    raw = pd.read_csv(url)
    if raw.empty or raw.shape[1] < 2:
        raise RuntimeError(f"FRED returned no usable data for id={series_ids!r}")
    # First column is the date ("DATE" or "observation_date" depending on FRED version).
    raw = raw.rename(columns={raw.columns[0]: date_col})
    value_cols = [c for c in raw.columns if c != date_col]
    # FRED encodes missing observations as ".".
    for c in value_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.rename(columns=colmap)
    value_cols = [colmap.get(c, c) for c in value_cols]
    raw[date_col] = pd.to_datetime(raw[date_col])
    if entry.get("start"):
        raw = raw[raw[date_col] >= pd.Timestamp(entry["start"])]
    if entry.get("end"):
        raw = raw[raw[date_col] <= pd.Timestamp(entry["end"])]
    raw = raw.sort_values(date_col)
    # Forward-fill isolated gaps (e.g. business-day series), then drop any leading
    # rows that have no observation yet for one of the joined series.
    raw[value_cols] = raw[value_cols].ffill()
    raw = raw.dropna(subset=value_cols)
    raw[date_col] = raw[date_col].dt.strftime("%Y-%m-%d")
    return raw[[date_col, *value_cols]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# statsmodels source — static public datasets (co2, sunspots, nile,
# air_passengers). No network after statsmodels is installed.
# ---------------------------------------------------------------------------

def _fetch_statsmodels(entry: dict[str, Any]) -> pd.DataFrame:
    import statsmodels.api as sm
    sid = entry["source_id"]

    if sid == "co2":
        raw = sm.datasets.co2.load_pandas().data  # weekly, DatetimeIndex, col 'co2'
        monthly = raw.resample("MS").mean().interpolate("linear").dropna()
        df = monthly.reset_index()
        df.columns = ["date", "co2"]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    if sid == "sunspots":
        raw = sm.datasets.sunspots.load_pandas().data.copy()
        raw.columns = [c.upper() for c in raw.columns]  # normalise capitalisation
        return pd.DataFrame({
            "year": raw["YEAR"].astype(int).apply(lambda y: f"{y}-01-01"),
            "sunspots": raw["SUNACTIVITY"].values,
        })

    if sid == "nile":
        raw = sm.datasets.nile.load_pandas().data.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        return pd.DataFrame({
            "year": raw["year"].astype(int).apply(lambda y: f"{y}-01-01"),
            "volume": raw["volume"].values,
        })

    if sid == "air_passengers":
        data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        # 'time' is a float year (1949.0, 1949.0833, ...); 'value' is the count
        def _fy(y: float) -> str:
            year = int(y)
            month = round((y - year) * 12) + 1
            return f"{year}-{month:02d}-01"
        return pd.DataFrame({
            "month": data["time"].apply(_fy),
            "passengers": data["value"].astype(int),
        })

    raise ValueError(f"Unknown statsmodels dataset source_id: {sid!r}. "
                     f"Valid: co2, sunspots, nile, air_passengers")


# ---------------------------------------------------------------------------
# uci_url source — downloads a CSV from a UCI archive URL and applies
# dataset-specific resampling. Identified by entry['dataset_id'].
# ---------------------------------------------------------------------------

def _fetch_uci_url(entry: dict[str, Any]) -> pd.DataFrame:
    dataset_id = entry["dataset_id"]
    url = entry["source_id"]

    if dataset_id == "bike_sharing_daily":
        # UCI 275 — daily Bike Sharing packaged as zip; day.csv has dteday + all exog
        import io as _io
        import urllib.request
        import zipfile
        with urllib.request.urlopen(url) as resp, zipfile.ZipFile(_io.BytesIO(resp.read())) as zf:
            df = pd.read_csv(zf.open("day.csv"), parse_dates=["dteday"])
        df["dteday"] = df["dteday"].dt.strftime("%Y-%m-%d")
        return df

    if dataset_id == "metro_traffic_volume":
        # UCI 492 — hourly Metro Interstate Traffic Volume → resample to daily
        df = pd.read_csv(url, parse_dates=["date_time"])
        df = df.set_index("date_time")
        df["is_holiday"] = (df["holiday"] != "None").astype(int)
        agg = {"traffic_volume": "mean", "temp": "mean", "rain_1h": "mean",
               "snow_1h": "mean", "clouds_all": "mean", "is_holiday": "max"}
        daily = df[list(agg)].resample("D").agg(agg).dropna().reset_index()
        daily = daily.rename(columns={"date_time": "date"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    if dataset_id == "beijing_pm25":
        # UCI 381 — hourly Beijing PM2.5 → resample to daily
        df = pd.read_csv(url)
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        keep = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        daily = df.set_index("datetime")[keep].resample("D").mean().dropna().reset_index()
        daily = daily.rename(columns={"datetime": "date", "pm2.5": "pm25"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    if dataset_id == "appliances_energy":
        # UCI 374 — 10-min Appliances Energy → resample to hourly (~3,300 rows)
        df = pd.read_csv(url, parse_dates=["date"])
        df = df.set_index("date")
        keep = ["Appliances", "T6", "RH_6", "T_out", "RH_out",
                "Windspeed", "Visibility", "Press_mm_hg"]
        hourly = df[keep].resample("h").mean().dropna().reset_index()
        hourly["date"] = hourly["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return hourly

    if dataset_id == "vic_elec":
        # VIC2015 half-hourly demand + temperature (3 separate files, Excel serial dates)
        # url is the base directory, e.g. .../VIC2015
        import urllib.request as _req
        excel_origin = pd.Timestamp("1899-12-30")

        demand = pd.read_csv(f"{url}/demand.csv")
        demand["datetime"] = (
            excel_origin
            + pd.to_timedelta(demand["Date"], unit="D")
            + pd.to_timedelta((demand["Period"] - 1) * 30, unit="min")
        )
        demand["demand"] = demand["OperationalLessIndustrial"] + demand["Industrial"]

        temp = pd.read_csv(f"{url}/temperature.csv")
        temp["datetime"] = (
            excel_origin
            + pd.to_timedelta(temp["Date"], unit="D")
            + pd.to_timedelta((temp["Period"] - 1) * 30, unit="min")
        )

        with _req.urlopen(f"{url}/holidays.txt") as resp:
            holiday_dates = {
                pd.to_datetime(line.strip(), dayfirst=True).date()
                for line in resp.read().decode().splitlines()
                if line.strip()
            }

        df = demand[["datetime", "demand"]].merge(
            temp[["datetime", "Temp"]].rename(columns={"Temp": "temperature"}),
            on="datetime", how="inner",
        ).set_index("datetime")

        daily = df.resample("D").agg({"demand": "sum", "temperature": "mean"}).dropna()
        daily["holiday"] = [int(d in holiday_dates) for d in daily.index.date]
        daily = daily.reset_index().rename(columns={"datetime": "date"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    raise ValueError(f"No uci_url handler for dataset_id: {dataset_id!r}. "
                     f"Valid: metro_traffic_volume, beijing_pm25, "
                     f"appliances_energy, vic_elec")


# ---------------------------------------------------------------------------
# dominicks source — Dominick's Finer Foods scanner data (UChicago Booth).
# Requires local movement CSV (wcoo.csv or similar) downloaded manually.
# entry["source_id"]  : path to raw movement CSV
# entry["store_id"]   : store number (default 62)
# entry["upc"]        : UPC code (default 4400000720 — top-selling cookie)
# Week 1 = 1989-09-14 per the Dominick's data manual.
# ---------------------------------------------------------------------------

def _fetch_dominicks(entry: dict[str, Any]) -> pd.DataFrame:
    raw_path = entry["source_id"]
    store_id = int(entry.get("store_id", 62))
    upc = int(entry.get("upc", 4400000720))

    chunks = []
    for chunk in pd.read_csv(raw_path, chunksize=500_000,
                             usecols=["STORE", "UPC", "WEEK", "MOVE", "PRICE", "SALE", "OK"]):
        sub = chunk[(chunk["OK"] == 1) & (chunk["STORE"] == store_id) & (chunk["UPC"] == upc)]
        if len(sub):
            chunks.append(sub)

    df = pd.concat(chunks).sort_values("WEEK").reset_index(drop=True)

    # Convert numeric week to date (week 1 = 1989-09-14)
    origin = pd.Timestamp("1989-09-14")
    df["week"] = (origin + pd.to_timedelta((df["WEEK"].astype(int) - 1) * 7, unit="D")).dt.strftime("%Y-%m-%d")

    df["promo"] = df["SALE"].notna().astype(int)
    # Replace price=0 (data error) with forward-fill
    df["PRICE"] = df["PRICE"].replace(0, float("nan")).ffill()

    return df[["week", "MOVE", "PRICE", "promo"]].rename(columns={"MOVE": "sales", "PRICE": "price"})
