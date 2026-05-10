"""Dataset fetchers for the benchmark runner."""
from __future__ import annotations
import pandas as pd


def fetch_dataset(entry: dict) -> pd.DataFrame:
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
        bunch = fetch_openml(data_id=int(entry["source_id"]), as_frame=True, parser="auto")
        return bunch.frame
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
    raise ValueError(f"Unknown source: {src!r}. Valid: sklearn, openml, local, yfinance")
