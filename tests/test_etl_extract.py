"""
test_etl_extract.py — pure-function pieces of src/etl/extract.py, with no
network calls. The actual Open-Meteo/GridStatus HTTP calls aren't exercised
here (that would make tests flaky and dependent on external services and
API keys) — instead this tests the request-shaping logic
(_openmeteo_day_windows) and the response-parsing logic
(_fetch_gridstatus_hour) against a fake client, which is where real bugs
(off-by-one day counts, picking the wrong row) would actually live.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.etl import extract


class _FakeGridStatusClient:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_dataset(self, dataset, start, end):
        return self._df.copy()


def test_openmeteo_day_windows_requires_utc():
    with pytest.raises(ValueError, match="tz-aware UTC"):
        extract._openmeteo_day_windows(pd.Timestamp("2026-01-01"), lookback_hours=96)


def test_openmeteo_day_windows_target_is_now():
    """target_timestamp == now: no forecast days needed, ~1 day of lookback padding."""
    now = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    past_days, forecast_days = extract._openmeteo_day_windows(now, lookback_hours=96, now=now)
    assert past_days >= 5  # 96h lookback (4 days) + padding
    assert forecast_days >= 1  # target's own remaining calendar day


def test_openmeteo_day_windows_target_far_in_future():
    now = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    target = now + pd.Timedelta(days=5)
    past_days, forecast_days = extract._openmeteo_day_windows(target, lookback_hours=96, now=now)
    assert forecast_days >= 6  # must reach 5 days ahead plus its own calendar day
    assert past_days >= 1


def test_openmeteo_day_windows_clamped_to_api_limits():
    now = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    target = now + pd.Timedelta(days=400)  # absurdly far ahead
    past_days, forecast_days = extract._openmeteo_day_windows(target, lookback_hours=96, now=now)
    assert forecast_days == extract.config.OPENMETEO_MAX_FORECAST_DAYS


def test_fetch_gridstatus_hour_happy_path():
    target_hour = pd.Timestamp("2026-01-01 05:00:00", tz="UTC")
    df = pd.DataFrame({
        "interval_start_utc": [target_hour],
        "south_central": [9123.45],
    })
    client = _FakeGridStatusClient(df)
    result = extract._fetch_gridstatus_hour(client, target_hour)
    assert result == pytest.approx(9123.45)


def test_fetch_gridstatus_hour_empty_response_raises():
    client = _FakeGridStatusClient(pd.DataFrame(columns=["interval_start_utc", "south_central"]))
    with pytest.raises(RuntimeError, match="no rows"):
        extract._fetch_gridstatus_hour(client, pd.Timestamp("2026-01-01 05:00:00", tz="UTC"))


def test_fetch_gridstatus_hour_missing_load_column_raises():
    df = pd.DataFrame({"interval_start_utc": [pd.Timestamp("2026-01-01 05:00:00", tz="UTC")]})
    client = _FakeGridStatusClient(df)
    with pytest.raises(RuntimeError, match="south_central"):
        extract._fetch_gridstatus_hour(client, pd.Timestamp("2026-01-01 05:00:00", tz="UTC"))


def test_fetch_gridstatus_hour_wrong_hour_returned_raises():
    """The response contains a row, but not for the hour we asked about — must not silently use it."""
    target_hour = pd.Timestamp("2026-01-01 05:00:00", tz="UTC")
    wrong_hour = pd.Timestamp("2026-01-01 06:00:00", tz="UTC")
    df = pd.DataFrame({"interval_start_utc": [wrong_hour], "south_central": [1000.0]})
    client = _FakeGridStatusClient(df)
    with pytest.raises(RuntimeError, match="Expected exactly 1"):
        extract._fetch_gridstatus_hour(client, target_hour)


def test_fetch_gridstatus_hour_multiple_rows_raises():
    target_hour = pd.Timestamp("2026-01-01 05:00:00", tz="UTC")
    df = pd.DataFrame({"interval_start_utc": [target_hour, target_hour], "south_central": [1000.0, 1001.0]})
    client = _FakeGridStatusClient(df)
    with pytest.raises(RuntimeError, match="Expected exactly 1"):
        extract._fetch_gridstatus_hour(client, target_hour)
