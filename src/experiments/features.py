"""
features.py — every derived feature, written once.

Previously this logic existed in three places (cell 10, cell 20, cell 49) with
cell 20 silently missing `is_high_precip_event`. The fix: always compute the
same four extreme-event flags regardless of feature version — FEATURES_V2
just doesn't select `is_high_precip_event` into X, so the unused column is
harmless. That collapses three copies into one.

Two kinds of function here, and the split matters:
  - "add_*" functions run ONCE on the whole dataframe (trend_idx, rolling
    stats, lag features) — nothing here is fit on train and applied to test,
    so there's no leakage risk.
  - "fit_*"/"apply_*" pairs are for anything derived from a THRESHOLD or
    FREQUENCY computed from data (extreme-event quantiles, holiday
    frequency). These must be fit on train (or the current CV fold's train
    slice) only, then applied to both splits — never refit on validation/test.
"""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# ---------------------------------------------------------------------------
# Whole-dataframe features (no fit/apply split needed — nothing to leak)
# ---------------------------------------------------------------------------

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """trend_idx, hour_x_temp, rolling temp std, and lag-24 temp change."""
    out = df.copy()
    out["trend_idx"] = (
        out["timestamp"] - out["timestamp"].min()
    ).dt.total_seconds() / (3600 * 24 * 365)
    out["hour_x_temp"] = out["hour"] * out["temp_c"]
    out["temp_c_roll_std_72"] = out["temp_c"].rolling(window=72, min_periods=24).std()
    out["temp_change_vs_lag24"] = out["temp_c"] - out["temp_c"].shift(24)
    return out


def attach_holiday_names(train: pd.DataFrame, test: pd.DataFrame):
    """
    Holiday calendar is built once from TRAIN's date range and applied to
    both splits — a fixed calendar lookup, not a fitted statistic, so this
    is safe to compute once rather than per-fold.
    """
    cal = USFederalHolidayCalendar()
    start = train["timestamp_central"].min().tz_localize(None)
    end = train["timestamp_central"].max().tz_localize(None)
    holiday_names = cal.holidays(start=start, end=end, return_name=True)

    for part in (train, test):
        central_date = part["timestamp_central"].dt.normalize().dt.tz_localize(None)
        part["holiday_name"] = central_date.map(holiday_names)
    return train, test


# ---------------------------------------------------------------------------
# Fit/apply pairs — must be fit on TRAIN (or the fold's train slice) only
# ---------------------------------------------------------------------------

def fit_extreme_event_thresholds(train_slice: pd.DataFrame) -> dict:
    return {
        "heat": float(train_slice["tmax"].quantile(0.95)),
        "cold": float(train_slice["tmin"].quantile(0.05)),
        "precip": float(train_slice["precip_mm"].quantile(0.95)),
    }


def apply_extreme_event_features(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    out["is_extreme_heat_event"] = (out["tmax"] > thresholds["heat"]).astype(int)
    out["is_extreme_cold_event"] = (out["tmin"] < thresholds["cold"]).astype(int)
    out["is_holiday_x_extreme"] = (
        out["is_holiday"] * (out["is_extreme_heat_event"] | out["is_extreme_cold_event"])
    ).astype(int)
    out["is_high_precip_event"] = (out["precip_mm"] > thresholds["precip"]).astype(int)
    return out


def fit_holiday_freq_map(train_slice: pd.DataFrame) -> dict:
    return train_slice["holiday_name"].value_counts(normalize=True).to_dict()


def apply_holiday_freq(df: pd.DataFrame, freq_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["holiday_freq"] = out["holiday_name"].map(freq_map).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Fold-level composition — what the CV loop actually calls
# ---------------------------------------------------------------------------

def extreme_event_fold_features(fold_train: pd.DataFrame, fold_val: pd.DataFrame):
    """Fits thresholds on fold_train only, applies to both splits."""
    thresholds = fit_extreme_event_thresholds(fold_train)
    return (
        apply_extreme_event_features(fold_train, thresholds),
        apply_extreme_event_features(fold_val, thresholds),
    )


def holiday_freq_fold_features(fold_train: pd.DataFrame, fold_val: pd.DataFrame):
    """Fits the holiday-frequency map on fold_train only, applies to both splits."""
    freq_map = fit_holiday_freq_map(fold_train)
    return (
        apply_holiday_freq(fold_train, freq_map),
        apply_holiday_freq(fold_val, freq_map),
    )


def compose(*fold_feature_fns):
    """Chains fold-feature functions, e.g. compose(extreme_event_fold_features, holiday_freq_fold_features)."""
    def _composed(fold_train, fold_val):
        for fn in fold_feature_fns:
            fold_train, fold_val = fn(fold_train, fold_val)
        return fold_train, fold_val
    return _composed
