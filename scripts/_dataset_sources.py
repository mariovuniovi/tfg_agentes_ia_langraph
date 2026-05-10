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
    raise ValueError(f"Unknown source: {src!r}. Valid: sklearn, openml, local")
