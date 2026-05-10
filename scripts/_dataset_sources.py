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
        target_col = entry.get("target_column", "target")
        bunch = fetch_openml(data_id=int(entry["source_id"]), as_frame=True, parser="auto")
        df = bunch.frame.copy() if bunch.frame is not None else pd.DataFrame(bunch.data)
        if target_col not in df.columns:
            if bunch.target is not None:
                df[target_col] = bunch.target.values
            elif len(df.columns) > 0:
                df = df.rename(columns={df.columns[-1]: target_col})
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

    raise ValueError(f"Unknown source: {src!r}. Valid: sklearn, openml, local, yfinance, yfinance_multi")
