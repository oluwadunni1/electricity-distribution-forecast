"""
data.py — loading and the chronological train/test split.
"""
import pandas as pd


def load_raw_data(path: str = "../data/interim/df_core_features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    purge_days: int = 7,
    test_start_central: str = "2025-08-13 00:00:00",
):
    """
    Splits on a purged boundary so no leakage occurs across the gap.
    Returns (train, test, train_end, test_start) — the boundary timestamps are
    returned too since several downstream steps (holiday calendar range, MLflow
    param logging) need them.
    """
    test_start_c = pd.Timestamp(test_start_central, tz="America/Chicago")
    test_start = test_start_c.tz_convert("UTC")
    train_end = test_start - pd.Timedelta(days=purge_days)

    train = df[df["timestamp"] <= train_end].reset_index(drop=True)
    test = df[df["timestamp"] >= test_start].reset_index(drop=True)
    return train, test, train_end, test_start
