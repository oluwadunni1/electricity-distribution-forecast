"""
timestamps.py — the ONE place that derives calendar/temp features from a
timestamp, for both training and live inference.

WHY this module exists: hour/dayofweek/month/is_weekend were once assumed
derivable from the UTC `timestamp` column during a test. They are not — they
come from `timestamp_central` (America/Chicago local time), because "is this
the evening peak" or "is this a weekend" are local-clock concepts, not UTC
ones (see Notebooks/Preprocessing.ipynb cells 4/6/8). That wrong assumption
didn't raise an error, it just silently fed the model the wrong local hour.
The root cause was that this derivation only lived inline in a notebook
cell, so nothing forced any other code path (a test, live ETL) to reproduce
it correctly. This module is the structural fix: src/experiments/features.py
(training) and src/etl/pipeline.py (live inference) both call these same
functions — there is exactly one implementation of "timestamp -> calendar
features," so a second, drifted copy can't exist.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CENTRAL_TZ = "America/Chicago"


def _require_utc(series: pd.Series, name: str) -> None:
    if series.dt.tz is None:
        raise ValueError(
            f"'{name}' must be tz-aware UTC, got tz-naive. A naive timestamp "
            f"is ambiguous about which local hour it represents — this is the "
            f"exact bug class this module exists to prevent."
        )


def add_timestamp_central(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds `timestamp_central`, always RECOMPUTED from the tz-aware UTC
    `timestamp_col` via tz_convert — never trusted from an upstream column.
    Mirrors Preprocessing.ipynb cell 4's
    `df["timestamp_central"] = df.index.tz_convert("America/Chicago")`,
    which itself deliberately recomputes rather than trusting an
    already-present timestamp_central, so the two columns can't drift apart.
    """
    _require_utc(df[timestamp_col], timestamp_col)
    out = df.copy()
    out["timestamp_central"] = out[timestamp_col].dt.tz_convert(CENTRAL_TZ)
    return out


def add_calendar_features(
    df: pd.DataFrame, timestamp_central_col: str = "timestamp_central"
) -> pd.DataFrame:
    """
    Adds hour/dayofweek/month/is_weekend, derived from `timestamp_central_col`.
    Faithful port of Preprocessing.ipynb cell 6 — do not derive these from a
    UTC timestamp; a UTC hour/day can be a different local hour/day/weekday
    around midnight, which is exactly the incident that motivated this module.
    """
    out = df.copy()
    central = out[timestamp_central_col]
    out["hour"] = central.dt.hour
    out["dayofweek"] = central.dt.dayofweek  # 0=Mon ... 6=Sun
    out["month"] = central.dt.month
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    return out


def holiday_calendar_names(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Single shared USFederalHolidayCalendar query: tz-naive normalized local
    calendar date -> holiday name, for every US federal holiday in
    [start, end]. Both training's fit-on-train-range calendar
    (src/experiments/features.py::attach_holiday_names) and live ETL's
    single-date calendar (src/etl/pipeline.py) go through this one function,
    so there's one holiday ruleset, not two.
    """
    cal = USFederalHolidayCalendar()
    return cal.holidays(start=start, end=end, return_name=True)


def apply_holiday_name(
    df: pd.DataFrame,
    holiday_names: pd.Series,
    timestamp_central_col: str = "timestamp_central",
) -> pd.DataFrame:
    """
    Maps `timestamp_central_col`'s normalized local date onto an
    already-built `holiday_names` map (from holiday_calendar_names()) and
    adds `holiday_name` (str or None). Does NOT touch `is_holiday` — callers
    that need a boolean flag derived from the SAME calendar range should use
    add_is_holiday_from_names() below; src/experiments/features.py's
    attach_holiday_names() deliberately keeps its pre-existing `is_holiday`
    column (built at Preprocessing time from the full dataset's date range)
    untouched, since re-deriving it from a train-only calendar range would
    silently flip it for holidays that fall in the test period.
    """
    out = df.copy()
    central_date = out[timestamp_central_col].dt.normalize().dt.tz_localize(None)
    out["holiday_name"] = central_date.map(holiday_names)
    return out


def add_is_holiday_from_names(df: pd.DataFrame, holiday_name_col: str = "holiday_name") -> pd.DataFrame:
    """is_holiday = holiday_name is not null. Kept as a separate step so callers
    can choose whether the boolean should come from the same calendar range as
    `holiday_name` (see apply_holiday_name's docstring)."""
    out = df.copy()
    out["is_holiday"] = out[holiday_name_col].notna().astype(int)
    return out


def add_holiday_features(
    df: pd.DataFrame,
    timestamp_central_col: str = "timestamp_central",
    calendar_pad_days: int = 3,
) -> pd.DataFrame:
    """
    Convenience wrapper for callers (live ETL) with no train/test split:
    builds the holiday calendar from this df's own
    [min(timestamp_central) - pad, max(timestamp_central) + pad] and adds
    both `holiday_name` and `is_holiday` from that single calendar query —
    they can't disagree because they come from the same lookup.

    calendar_pad_days matters for a single-row ETL slice (one target hour):
    without padding, start == end == that one date, which is a valid
    zero-width query, but padding keeps this robust for any future caller
    that passes a short, non-representative date range.
    """
    central_dates = df[timestamp_central_col].dt.normalize().dt.tz_localize(None)
    start = central_dates.min() - pd.Timedelta(days=calendar_pad_days)
    end = central_dates.max() + pd.Timedelta(days=calendar_pad_days)
    holiday_names = holiday_calendar_names(start, end)

    out = apply_holiday_name(df, holiday_names, timestamp_central_col=timestamp_central_col)
    out = add_is_holiday_from_names(out)
    return out


def add_calendar_and_holiday_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    End-to-end convenience: timestamp -> timestamp_central -> calendar ->
    holiday_name + is_holiday. This is the one function live ETL should call
    to go from a raw UTC timestamp to every timestamp_central-derived
    feature in a single step.
    """
    out = add_timestamp_central(df, timestamp_col=timestamp_col)
    out = add_calendar_features(out)
    out = add_holiday_features(out)
    return out


def add_temp_derived_features(
    df: pd.DataFrame, temp_col: str = "temp_c", hour_col: str = "hour"
) -> pd.DataFrame:
    """
    Adds hour_x_temp, temp_c_roll_std_72, temp_change_vs_lag24. Ported
    verbatim from src/experiments/features.py's add_base_features(). These
    live alongside the calendar logic (rather than in features.py) because
    hour_x_temp depends directly on the `hour` column derived above, and —
    like hour/dayofweek/month — getting it from a second, slightly different
    reimplementation is exactly the bug class this module exists to prevent.

    df must be sorted ascending by timestamp and hourly-contiguous (no gaps)
    for temp_c_roll_std_72 (72h trailing window, min_periods=24) and
    temp_change_vs_lag24 (shift(24)) to mean what they're supposed to mean.
    Produces NaN for the first ~24-72 rows of whatever df is passed in — for
    a full-history refit, drop those with
    src/experiments/features.py::drop_base_feature_warmup(). For live ETL
    predicting a single future hour, the earlier rows exist only to build
    this rolling window and should be dropped after this call, before the
    target row is sent onward — see src/etl/pipeline.py.
    """
    out = df.copy()
    out["hour_x_temp"] = out[hour_col] * out[temp_col]
    out["temp_c_roll_std_72"] = out[temp_col].rolling(window=72, min_periods=24).std()
    out["temp_change_vs_lag24"] = out[temp_col] - out[temp_col].shift(24)
    return out
