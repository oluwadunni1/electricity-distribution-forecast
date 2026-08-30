"""
test_etl_validation.py — src/etl/validation.py's stage-boundary schema checks.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.etl import validation


def test_valid_dataframe_passes():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "temp_c": [10.0, 12.0, 14.0],
        "humidity_pct": [50.0, 55.0, 60.0],
        "precip_mm": [0.0, 1.0, 0.0],
        "tmax": [15.0, 15.0, 15.0],
        "tmin": [8.0, 8.0, 8.0],
        "tavg": [11.0, 11.0, 11.0],
    })
    validation.validate_dataframe(df, validation.WEATHER_SPEC, stage="test")  # must not raise


def test_missing_column_raises():
    df = pd.DataFrame({"temp_c": [10.0]})
    with pytest.raises(ValueError, match="missing column"):
        validation.validate_dataframe(df, validation.WEATHER_SPEC, stage="test")


def test_empty_dataframe_raises():
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in validation.LOAD_LAG_SPEC})
    with pytest.raises(ValueError, match="empty"):
        validation.validate_dataframe(df, validation.LOAD_LAG_SPEC, stage="test")


def test_out_of_bounds_value_raises():
    df = pd.DataFrame({"load_lag_24": [-5.0], "load_lag_168": [1000.0]})
    with pytest.raises(ValueError, match="below minimum"):
        validation.validate_dataframe(df, validation.LOAD_LAG_SPEC, stage="test")


def test_nan_raises_by_default():
    df = pd.DataFrame({"load_lag_24": [float("nan")], "load_lag_168": [100.0]})
    with pytest.raises(ValueError, match="NaN"):
        validation.validate_dataframe(df, validation.LOAD_LAG_SPEC, stage="test")


def test_nan_allowed_when_spec_says_so():
    df = pd.DataFrame({"hour_x_temp": [1.0, 2.0]})
    spec = {"hour_x_temp": {"dtype": "numeric", "allow_nan": True}}
    df.loc[0, "hour_x_temp"] = float("nan")
    validation.validate_dataframe(df, spec, stage="test")  # must not raise


def test_wrong_dtype_raises():
    df = pd.DataFrame({"temp_c": ["not-a-number", "also-not"]})
    spec = {"temp_c": {"dtype": "numeric"}}
    with pytest.raises(ValueError, match="expected numeric dtype"):
        validation.validate_dataframe(df, spec, stage="test")


def test_tz_naive_datetime_raises():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:00:00"])})
    spec = {"timestamp": {"dtype": "datetime"}}
    with pytest.raises(ValueError, match="tz-aware UTC"):
        validation.validate_dataframe(df, spec, stage="test")


def test_multiple_problems_all_reported():
    df = pd.DataFrame({"load_lag_24": [-1.0]})  # missing load_lag_168, out-of-bounds load_lag_24
    with pytest.raises(ValueError) as exc_info:
        validation.validate_dataframe(df, validation.LOAD_LAG_SPEC, stage="test")
    message = str(exc_info.value)
    assert "load_lag_168" in message
    assert "below minimum" in message
