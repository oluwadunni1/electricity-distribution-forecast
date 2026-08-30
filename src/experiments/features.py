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

hour_x_temp/temp_c_roll_std_72/temp_change_vs_lag24 and the holiday-name
calendar lookup are computed via src/common/timestamps.py, not here — that
module is the single shared implementation used by both training (this file)
and live ETL (src/etl/pipeline.py). See its docstring for why.
"""
import pandas as pd

from src.common import timestamps


# ---------------------------------------------------------------------------
# Whole-dataframe features (no fit/apply split needed — nothing to leak)
# ---------------------------------------------------------------------------

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    trend_idx, hour_x_temp, rolling temp std, and lag-24 temp change.

    NOTE: temp_c_roll_std_72 (window=72, min_periods=24) and temp_change_vs_lag24
    (shift(24)) both produce NaN for the first ~24-72 rows of whatever dataframe
    is passed in — this is a NEW warm-up period, separate from the load_lag_24/168
    warm-up already handled in Preprocessing.ipynb before the parquet was saved.
    Preprocessing's dropna only covered the columns that existed at that stage;
    it has no way to know about columns created later, here. Call
    drop_base_feature_warmup() right after this to clean it up, the same way
    Preprocessing.ipynb explicitly drops its own warm-up rows.
    """
    out = df.copy()
    out["trend_idx"] = (
        out["timestamp"] - out["timestamp"].min()
    ).dt.total_seconds() / (3600 * 24 * 365)
    out = timestamps.add_temp_derived_features(out, temp_col="temp_c", hour_col="hour")
    return out


def drop_base_feature_warmup(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drops rows with NaN in the columns add_base_features() just created.
    Call this immediately after add_base_features(), before the train/test
    split — mirrors Preprocessing.ipynb's own explicit warm-up drop, just for
    the features that get engineered downstream instead of at ingestion time.
    """
    warmup_cols = ["temp_c_roll_std_72", "temp_change_vs_lag24"]
    before = len(df)
    out = df.dropna(subset=warmup_cols).reset_index(drop=True)
    if verbose:
        print(f"Dropped {before - len(out)} warm-up row(s) for {warmup_cols}")
    return out


def attach_holiday_names(train: pd.DataFrame, test: pd.DataFrame):
    """
    Holiday calendar is built once from TRAIN's date range and applied to
    both splits — a fixed calendar lookup, not a fitted statistic, so this
    is safe to compute once rather than per-fold.

    Only adds `holiday_name` — deliberately does NOT touch `is_holiday`,
    which already exists on both frames from Preprocessing.ipynb, computed
    over the FULL dataset's date range (pre-split). Re-deriving it here from
    a train-only calendar range would silently flip it to 0 for any holiday
    that falls in the test period, purely because the lookup window didn't
    cover it.
    """
    start = train["timestamp_central"].min().tz_localize(None)
    end = train["timestamp_central"].max().tz_localize(None)
    holiday_names = timestamps.holiday_calendar_names(start, end)

    train = timestamps.apply_holiday_name(train, holiday_names)
    test = timestamps.apply_holiday_name(test, holiday_names)
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