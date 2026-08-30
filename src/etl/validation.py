"""
validation.py — explicit schema checks at every ETL stage boundary.

Each pipeline stage's output is checked immediately after it's produced:
column names, dtypes, and sanity bounds. This exists so a bad row (a null
from a flaky weather response, a GridStatus lookup that silently returned
the wrong hour, a unit mismatch) fails loudly at the stage that produced it
— with a specific column and value in the error — instead of propagating
into feature engineering or, worse, into a live prediction.
"""
from __future__ import annotations

import pandas as pd

# {column: {"dtype": "numeric"|"datetime"|"string", "min": float|None,
#           "max": float|None, "allow_nan": bool}}
Spec = dict


def validate_dataframe(df: pd.DataFrame, spec: Spec, stage: str) -> None:
    """Raises ValueError listing every problem found, or returns silently if clean."""
    problems: list[str] = []

    missing = [c for c in spec if c not in df.columns]
    if missing:
        problems.append(f"missing column(s): {missing}")

    if df.empty:
        problems.append("dataframe is empty")

    for col, rules in spec.items():
        if col not in df.columns:
            continue
        series = df[col]
        dtype_kind = rules.get("dtype", "numeric")

        if dtype_kind == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                problems.append(f"'{col}': expected numeric dtype, got {series.dtype}")
                continue
        elif dtype_kind == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                problems.append(f"'{col}': expected datetime dtype, got {series.dtype}")
                continue
            if series.dt.tz is None:
                problems.append(f"'{col}': datetime column must be tz-aware UTC, got tz-naive")
            continue
        elif dtype_kind == "string":
            continue
        else:
            raise ValueError(f"validate_dataframe: unknown dtype_kind '{dtype_kind}' in spec for '{col}'")

        if not rules.get("allow_nan", False) and series.isna().any():
            n_nan = int(series.isna().sum())
            problems.append(f"'{col}': {n_nan} NaN value(s)")

        finite = series.dropna()
        if finite.empty:
            continue
        lo, hi = rules.get("min"), rules.get("max")
        if lo is not None and (finite < lo).any():
            problems.append(f"'{col}': value(s) below minimum {lo} (min found: {finite.min()})")
        if hi is not None and (finite > hi).any():
            problems.append(f"'{col}': value(s) above maximum {hi} (max found: {finite.max()})")

    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(f"[{stage}] schema validation failed:\n  - {detail}")


# ---------------------------------------------------------------------------
# Per-stage specs
# ---------------------------------------------------------------------------

# extract.fetch_weather_features() output. Bounds are generous sanity checks
# (South Central Texas doesn't see -40C or 55C), not tight physical limits —
# the point is to catch a unit error or a garbage API response, not to
# reject a genuine heat wave.
WEATHER_SPEC: Spec = {
    "timestamp": {"dtype": "datetime"},
    "temp_c": {"dtype": "numeric", "min": -40, "max": 55},
    "humidity_pct": {"dtype": "numeric", "min": 0, "max": 100},
    "precip_mm": {"dtype": "numeric", "min": 0, "max": 500},
    "tmax": {"dtype": "numeric", "min": -40, "max": 55},
    "tmin": {"dtype": "numeric", "min": -40, "max": 55},
    "tavg": {"dtype": "numeric", "min": -40, "max": 55},
}

# extract.fetch_load_lags() output (as a one-row DataFrame). Upper bound is
# well above ERCOT South Central's historical peak — a sanity ceiling, not a
# forecast of the true max.
LOAD_LAG_SPEC: Spec = {
    "load_lag_24": {"dtype": "numeric", "min": 0, "max": 50_000},
    "load_lag_168": {"dtype": "numeric", "min": 0, "max": 50_000},
}

# src/common/timestamps.py output, merged with weather — checked right
# before the raw row is handed to feature_engineering.transform_for_inference().
CALENDAR_SPEC: Spec = {
    "hour": {"dtype": "numeric", "min": 0, "max": 23},
    "dayofweek": {"dtype": "numeric", "min": 0, "max": 6},
    "month": {"dtype": "numeric", "min": 1, "max": 12},
    "is_weekend": {"dtype": "numeric", "min": 0, "max": 1},
    "is_holiday": {"dtype": "numeric", "min": 0, "max": 1},
    "holiday_name": {"dtype": "string"},
    "hour_x_temp": {"dtype": "numeric"},
    "temp_c_roll_std_72": {"dtype": "numeric", "min": 0, "allow_nan": True},
    "temp_change_vs_lag24": {"dtype": "numeric", "allow_nan": True},
}
