"""
test_timestamps.py

Covers src/common/timestamps.py — the single shared implementation of
"derive calendar/temp features from a timestamp" used by both training
(src/experiments/features.py) and live ETL (src/etl/pipeline.py).

The most important test here is test_hour_dayofweek_diverge_from_utc_*:
it's an explicit regression guard for the exact incident that motivated this
module — hour/dayofweek/month were once assumed derivable from the UTC
`timestamp` column, when they actually come from `timestamp_central`. That
assumption produced no error, just a silently wrong local hour. This test
picks a real row where the two disagree and asserts the module gets it
right.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.common import timestamps

PIPELINE_SAMPLE_PATH = "data/interim/pipeline_test_sample.parquet"
FEATURES_FIXTURE_PATH = "tests/fixtures/before_attach_holiday_train.parquet"


def _normalize_missing(series: pd.Series) -> list:
    """NaN and None both round-trip as "missing" through different paths
    (a fresh pandas .map() miss vs. a parquet-loaded null) — normalize
    before comparing so this isn't a false mismatch."""
    return [None if pd.isna(v) else v for v in series]


@pytest.fixture(scope="module")
def pipeline_sample() -> pd.DataFrame:
    return pd.read_parquet(PIPELINE_SAMPLE_PATH)


def test_add_timestamp_central_requires_utc():
    naive = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:00:00"])})
    with pytest.raises(ValueError, match="tz-aware UTC"):
        timestamps.add_timestamp_central(naive)


def test_calendar_and_holiday_match_known_sample(pipeline_sample):
    """
    Every row in pipeline_test_sample.parquet has hour/dayofweek/month/
    is_weekend/is_holiday/holiday_name already computed at fit time. Rebuild
    them from scratch via the shared module and assert exact equality.
    """
    raw = pipeline_sample[["timestamp"]].copy()
    out = timestamps.add_calendar_and_holiday_features(raw, timestamp_col="timestamp")

    for col in ("hour", "dayofweek", "month", "is_weekend", "is_holiday"):
        pd.testing.assert_series_equal(
            out[col].astype(pipeline_sample[col].dtype),
            pipeline_sample[col],
            check_names=False,
        )
    assert _normalize_missing(out["holiday_name"]) == _normalize_missing(pipeline_sample["holiday_name"])


def test_hour_dayofweek_diverge_from_utc_for_midnight_crossing_row(pipeline_sample):
    """
    2026-06-06 00:00:00 UTC is 2026-06-05 19:00:00 Central (CDT, UTC-5) —
    a different hour, day, and weekday than the UTC timestamp itself. If
    hour/dayofweek were (wrongly) derived from the UTC column, this row
    would come out as hour=0, dayofweek=5 (Saturday). The correct,
    Central-derived values are hour=19, dayofweek=4 (Friday) — exactly what
    the stored sample has.
    """
    row = pipeline_sample.loc[pipeline_sample["timestamp"] == pd.Timestamp("2026-06-06 00:00:00", tz="UTC")]
    assert len(row) == 1, "expected fixture row not found in pipeline_test_sample.parquet"
    row = row.iloc[0]

    assert row["hour"] == 19
    assert row["dayofweek"] == 4

    utc_hour_would_be = row["timestamp"].hour
    utc_dayofweek_would_be = row["timestamp"].dayofweek
    assert row["hour"] != utc_hour_would_be
    assert row["dayofweek"] != utc_dayofweek_would_be

    out = timestamps.add_calendar_and_holiday_features(pd.DataFrame({"timestamp": [row["timestamp"]]}))
    assert out.loc[0, "hour"] == 19
    assert out.loc[0, "dayofweek"] == 4


@pytest.mark.parametrize(
    "utc_ts, expected_hour",
    [
        # 2026-03-08 is the US spring-forward date (2am -> 3am, CST -> CDT).
        ("2026-03-08 07:59:00+00:00", 1),   # just before the jump: CST, UTC-6
        ("2026-03-08 09:01:00+00:00", 4),   # just after the jump: CDT, UTC-5
        # 2026-11-01 is the US fall-back date (2am CDT -> 1am CST).
        ("2026-11-01 06:59:00+00:00", 1),   # still CDT (UTC-5) before the fall-back
        ("2026-11-01 08:01:00+00:00", 2),   # after the fall-back: CST, UTC-6
    ],
)
def test_dst_transition_offsets(utc_ts, expected_hour):
    """Central-time conversion must follow DST, not a fixed UTC offset."""
    ts = pd.Timestamp(utc_ts)
    out = timestamps.add_calendar_and_holiday_features(pd.DataFrame({"timestamp": [ts]}))
    assert out.loc[0, "hour"] == expected_hour


def test_holiday_name_matches_features_fixture():
    """
    tests/fixtures/before_attach_holiday_train.parquet was captured from
    src/experiments/features.py::attach_holiday_names() BEFORE it was
    refactored to call this module — it contains real holiday names
    (Columbus Day, Juneteenth, MLK Day). Rebuild the same holiday_name
    column via the shared calendar-lookup primitives and confirm it matches.
    """
    fixture = pd.read_parquet(FEATURES_FIXTURE_PATH)
    start = fixture["timestamp_central"].min().tz_localize(None)
    end = fixture["timestamp_central"].max().tz_localize(None)
    holiday_names = timestamps.holiday_calendar_names(start, end)

    out = timestamps.apply_holiday_name(fixture, holiday_names)
    assert _normalize_missing(out["holiday_name"]) == _normalize_missing(fixture["holiday_name"])
    assert fixture["holiday_name"].notna().sum() >= 3  # sanity: fixture actually has holidays in it


def test_add_temp_derived_features():
    hours = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    temp_c = pd.Series(range(100), dtype=float) * 0.25  # deterministic, not constant
    df = pd.DataFrame({"timestamp": hours, "temp_c": temp_c, "hour": hours.hour})

    out = timestamps.add_temp_derived_features(df, temp_col="temp_c", hour_col="hour")

    expected_hour_x_temp = df["hour"] * df["temp_c"]
    pd.testing.assert_series_equal(out["hour_x_temp"], expected_hour_x_temp, check_names=False)

    expected_roll_std = df["temp_c"].rolling(window=72, min_periods=24).std()
    pd.testing.assert_series_equal(out["temp_c_roll_std_72"], expected_roll_std, check_names=False)

    expected_change = df["temp_c"] - df["temp_c"].shift(24)
    pd.testing.assert_series_equal(out["temp_change_vs_lag24"], expected_change, check_names=False)

    # warm-up rows (< 24) must be NaN for both rolling/lag features
    assert out["temp_c_roll_std_72"].iloc[:23].isna().all()
    assert out["temp_change_vs_lag24"].iloc[:24].isna().all()
    # by row 72 both should be populated
    assert out["temp_c_roll_std_72"].iloc[72:].notna().all()
    assert out["temp_change_vs_lag24"].iloc[24:].notna().all()
