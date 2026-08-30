"""
test_etl_pipeline.py — src/etl/pipeline.py's build_raw_feature_row(), which
wires extract.py's output through src/common/timestamps.py into the exact
raw-row shape src/inference/feature_engineering.py expects.

extract.fetch_weather_features / extract.fetch_load_lags are monkeypatched
to avoid real network calls (Open-Meteo, GridStatus) — this tests the
ASSEMBLY logic (stage-boundary validation, target-row selection, the
warm-up/NaN guard), not the HTTP layer, which test_etl_extract.py already
covers piece by piece.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl import extract, pipeline
from src.inference.feature_engineering import RAW_REQUIRED_COLUMNS

# A known-good row from data/interim/pipeline_test_sample.parquet — see
# test_timestamps.py::test_hour_dayofweek_diverge_from_utc_for_midnight_crossing_row
# for why this specific timestamp is a strong regression case (Central and
# UTC disagree on hour/dayofweek/date for it).
TARGET_TIMESTAMP = pd.Timestamp("2026-06-06 00:00:00", tz="UTC")
EXPECTED_HOUR = 19
EXPECTED_DAYOFWEEK = 4
EXPECTED_MONTH = 6
EXPECTED_IS_WEEKEND = 0


def _synthetic_weather(target_timestamp: pd.Timestamp, lookback_hours: int = 96) -> pd.DataFrame:
    """A contiguous, deterministic hourly weather series covering
    [target - lookback_hours, end of target's Central day] — enough history
    for the 72h rolling window and 24h lag to be fully populated at the
    target row."""
    start = target_timestamp - pd.Timedelta(hours=lookback_hours)
    end = target_timestamp + pd.Timedelta(hours=6)  # pad past target, inside its Central day
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    temp_c = 20 + rng.normal(scale=3.0, size=len(idx))
    return pd.DataFrame({
        "timestamp": idx,
        "temp_c": temp_c,
        "humidity_pct": np.full(len(idx), 55.0),
        "precip_mm": np.zeros(len(idx)),
        "tmax": np.full(len(idx), temp_c.max()),
        "tmin": np.full(len(idx), temp_c.min()),
        "tavg": np.full(len(idx), temp_c.mean()),
    })


def test_build_raw_feature_row_shape_and_calendar_values(monkeypatch):
    weather = _synthetic_weather(TARGET_TIMESTAMP)
    monkeypatch.setattr(extract, "fetch_weather_features", lambda ts, **kw: weather)
    monkeypatch.setattr(
        extract, "fetch_load_lags", lambda ts: {"load_lag_24": 8000.0, "load_lag_168": 8500.0}
    )

    raw = pipeline.build_raw_feature_row(TARGET_TIMESTAMP)

    assert len(raw) == 1
    assert set(raw.columns) == set(RAW_REQUIRED_COLUMNS + ["timestamp"])

    row = raw.iloc[0]
    assert row["timestamp"] == TARGET_TIMESTAMP
    # The exact regression this module exists to prevent: hour/dayofweek/month
    # must come from timestamp_central, not the UTC `timestamp` column.
    assert row["hour"] == EXPECTED_HOUR
    assert row["dayofweek"] == EXPECTED_DAYOFWEEK
    assert row["month"] == EXPECTED_MONTH
    assert row["is_weekend"] == EXPECTED_IS_WEEKEND

    assert row["load_lag_24"] == 8000.0
    assert row["load_lag_168"] == 8500.0

    # hour_x_temp must equal this row's own hour * temp_c, independent of the
    # rolling/lag computation.
    assert row["hour_x_temp"] == pytest.approx(row["hour"] * row["temp_c"])
    assert not pd.isna(row["temp_c_roll_std_72"])
    assert not pd.isna(row["temp_change_vs_lag24"])


def test_build_raw_feature_row_raises_on_insufficient_lookback(monkeypatch):
    """A weather fetch with too little history leaves the target row's
    rolling/lag features NaN — this must fail loudly, not propagate."""
    weather = _synthetic_weather(TARGET_TIMESTAMP, lookback_hours=10)  # far short of the 72h/24h needed
    monkeypatch.setattr(extract, "fetch_weather_features", lambda ts, **kw: weather)
    monkeypatch.setattr(
        extract, "fetch_load_lags", lambda ts: {"load_lag_24": 8000.0, "load_lag_168": 8500.0}
    )

    with pytest.raises(RuntimeError, match="didn't provide enough lookback history"):
        pipeline.build_raw_feature_row(TARGET_TIMESTAMP)


def test_build_raw_feature_row_requires_utc_target():
    with pytest.raises(ValueError, match="tz-aware UTC"):
        pipeline.build_raw_feature_row(pd.Timestamp("2026-06-06 00:00:00"))
