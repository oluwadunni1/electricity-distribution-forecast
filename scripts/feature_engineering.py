"""
feature_engineering.py
-----------------------
Transforms the raw joined dataset into the Section 3 baseline feature table
for day-ahead (24-hour-ahead) load forecasting on ERCOT South Central.

Dataset columns (input)
-----------------------
    timestamp       : hourly UTC datetimes, 2016-01-01 → 2026-07-28
    actual_load_mw  : ERCOT South Central zone electricity demand (MW)
    temp_c          : population-weighted avg temperature for the zone (°C)
    humidity_pct    : relative humidity (%)
    precip_mm       : hourly precipitation (mm)

Section 3 baseline feature set (output)
----------------------------------------
    Calendar   : hour, day_of_week, month, is_weekend, is_holiday
    Weather    : tavg, tmin, tmax, prcp
    Lagged load: load_lag_24, load_lag_168
    Rolling    : load_roll_mean_24, load_roll_max_24, load_roll_std_24

Column-name mapping  (Open-Meteo raw names → experiment spec names)
--------------------------------------------------------------------
    temp_c          → tavg   (hourly avg temperature proxy, °C)
    rolling 24h max → tmax   (max temp over the prior 24 h)
    rolling 24h min → tmin   (min temp over the prior 24 h)
    precip_mm       → prcp   (hourly precipitation, mm)

Usage
-----
    uv run python scripts/feature_engineering.py
    uv run python scripts/feature_engineering.py \\
        --input  data/raw/ercot_south_central_raw.csv \\
        --output data/processed/ercot_south_central_features.csv

Design rules
------------
* Every public function is pure: DataFrame in → DataFrame out, no disk I/O.
  Call them identically from the notebook, CLI, or Azure inference pipeline.

* NO-LEAKAGE: every feature for row t uses only information available
  strictly before t.  Rolling and lag features use shift(1) or shift(n) to
  exclude the current row's own value.

* NaN handling: the first 168 rows carry NaNs in lag/rolling columns
  (168 h = the longest look-back, load_lag_168).  These rows are intentionally
  kept — the train/test split step decides how to handle them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Default I/O paths ─────────────────────────────────────────────────────────
DEFAULT_INPUT  = Path("data/raw/ercot_south_central_raw.csv")
DEFAULT_OUTPUT = Path("data/processed/ercot_south_central_features.csv")


# =============================================================================
# Building-block feature functions
# (each accepts a DataFrame and returns an enriched copy — no side effects)
# =============================================================================


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based calendar features derived purely from the timestamp.

    No leakage risk — all values are known at any future time t.

    New columns
    -----------
    hour        : int  0-23  (UTC hour, matches the hour used by lags/rolling)
    day_of_week : int  0-6   (Monday = 0, Sunday = 6)
    month       : int  1-12
    is_weekend  : int  0/1   (1 on Saturday = 5 or Sunday = 6)
    is_holiday  : int  0/1   (US Federal holiday calendar, date-level check)
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"]       = ts.dt.month
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)

    cal      = USFederalHolidayCalendar()
    holidays = cal.holidays(start=str(ts.dt.date.min()), end=str(ts.dt.date.max()))
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)

    return df


def add_load_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exact load lag features using pandas .shift().

    NO-LEAKAGE: shift(n) for positive n pulls the value from n rows before
    the current row.  The DataFrame is sorted hourly, so:
        shift(24)  → load exactly 24 h prior  (same hour, yesterday)
        shift(168) → load exactly 168 h prior (same hour, last week)

    New columns
    -----------
    load_lag_24  : actual_load_mw 24 hours prior
    load_lag_168 : actual_load_mw 168 hours (7 days) prior

    NaN rows
    --------
    First 24 rows  → NaN in load_lag_24.
    First 168 rows → NaN in load_lag_168.
    Do NOT backfill or interpolate — handle at the train/test split step.
    """
    df = df.copy()
    df["load_lag_24"]  = df["actual_load_mw"].shift(24)
    df["load_lag_168"] = df["actual_load_mw"].shift(168)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing 24-hour rolling statistics on actual load.

    LEAKAGE GUARD — why shift(1) before rolling:
    ---------------------------------------------
    rolling(24) without a shift includes the current hour's own load value
    in the window — which is the target being predicted.  Shifting the
    series by 1 first means the 24-h window for row t spans [t-25 … t-1],
    i.e. the 24 completed hours immediately before t.  That information
    is genuinely available at day-ahead inference time.

    New columns
    -----------
    load_roll_mean_24 : mean load over the 24 h immediately before t
    load_roll_max_24  : max  load over the 24 h immediately before t
    load_roll_std_24  : std  load over the 24 h immediately before t
                        (Bessel-corrected, ddof=1 — pandas default)

    NaN rows
    --------
    First 25 rows carry NaN (24-window + 1 shift), which is a subset of
    the 168-row NaN block from load_lag_168.
    """
    df = df.copy()
    load_shifted = df["actual_load_mw"].shift(1)   # leakage guard
    df["load_roll_mean_24"] = load_shifted.rolling(window=24).mean()
    df["load_roll_max_24"]  = load_shifted.rolling(window=24).max()
    df["load_roll_std_24"]  = load_shifted.rolling(window=24).std()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Open-Meteo column names to Section 3 experiment spec names and
    derive rolling temperature bounds (tmin, tmax).

    Column mapping
    --------------
    temp_c    → tavg  (direct alias — hourly avg temperature proxy, °C)
    precip_mm → prcp  (direct alias — hourly precipitation, mm)

    Derived with the same shift(1) leakage guard as rolling stats:
    tmax  : max temp_c over the 24 completed hours immediately before t
    tmin  : min temp_c over the 24 completed hours immediately before t

    NOTE: humidity_pct is retained as-is from the raw dataset; it has no
    alias in the spec but is available for modelling and EDA.

    NaN rows
    --------
    First 25 rows carry NaN in tmin/tmax (same window as rolling stats).
    """
    df = df.copy()

    # Direct aliases
    df["tavg"] = df["temp_c"]
    df["prcp"] = df["precip_mm"]

    # Rolling temperature bounds — leakage-guarded
    temp_shifted = df["temp_c"].shift(1)
    df["tmax"] = temp_shifted.rolling(window=24).max()
    df["tmin"] = temp_shifted.rolling(window=24).min()

    return df


# =============================================================================
# Primary public API — Section 3 baseline feature set
# =============================================================================


def add_baseline_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all Section 3 feature-engineering steps in dependency order and
    return the fully enriched DataFrame.

    Input columns required
    ----------------------
    timestamp, actual_load_mw, temp_c, humidity_pct, precip_mm

    Output adds
    -----------
    Calendar   : hour, day_of_week, month, is_weekend, is_holiday
    Weather    : tavg, tmin, tmax, prcp
    Lagged load: load_lag_24, load_lag_168
    Rolling    : load_roll_mean_24, load_roll_max_24, load_roll_std_24

    The original raw columns are preserved unchanged.

    Usage in experiments.ipynb
    --------------------------
    >>> import sys
    >>> sys.path.insert(0, "scripts")          # adjust if notebook is in a subfolder
    >>> from feature_engineering import add_baseline_feature_set
    >>>
    >>> df_feat = add_baseline_feature_set(df_raw)
    >>>
    >>> # Exact Section 3 column selection:
    >>> baseline_cols = [
    ...     "hour", "day_of_week", "month",
    ...     "tavg", "tmin", "tmax", "prcp",
    ...     "load_lag_24", "load_lag_168",
    ... ]
    >>> df_baseline = df_feat[["timestamp", "actual_load_mw"] + baseline_cols].copy()

    Parameters
    ----------
    df : pd.DataFrame
        Raw joined DataFrame.  ``timestamp`` may be a string or datetime;
        it is coerced to tz-aware UTC datetime internally.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all baseline columns appended.
    """
    df = df.copy()

    # Coerce timestamp to tz-aware datetime so all .dt accessors work
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = add_calendar_features(df)
    df = add_load_lags(df)
    df = add_rolling_stats(df)
    df = add_weather_features(df)

    log.info("add_baseline_feature_set: %d rows × %d columns.", *df.shape)
    return df


# =============================================================================
# I/O helpers (used only when running as a standalone CLI script)
# =============================================================================


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw joined CSV and parse the timestamp column to UTC datetime."""
    log.info("Reading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(
        "Raw data: %d rows  %s → %s",
        len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1],
    )
    return df


def save_features(df: pd.DataFrame, path: Path) -> None:
    """Write the feature DataFrame to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path.resolve())


def print_summary(df: pd.DataFrame) -> None:
    """Print a quality summary of the engineered feature table."""
    lag_rolling_cols = [
        "load_lag_24", "load_lag_168",
        "load_roll_mean_24", "load_roll_max_24", "load_roll_std_24",
        "tmax", "tmin",
    ]
    present = [c for c in lag_rolling_cols if c in df.columns]
    nan_mask = df[present].isna().any(axis=1)
    n_nan    = int(nan_mask.sum())

    print("\n" + "=" * 62)
    print("  FEATURE ENGINEERING SUMMARY")
    print("=" * 62)
    print(f"  Total rows                  : {len(df):,}")
    print(f"  Rows with NaN (lag/rolling) : {n_nan:,}  ← expected (first {n_nan} hours)")
    print(f"  NaN rows are NOT dropped here — handle at train/test split.")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype     = str(df[col].dtype)
        n_nulls   = int(df[col].isna().sum())
        null_note = f"   ← {n_nulls:,} NaN" if n_nulls else ""
        print(f"    {col:<30} {dtype:<12}{null_note}")
    print("=" * 62 + "\n")


# =============================================================================
# CLI entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the Section 3 baseline features for ERCOT load forecasting."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to raw joined CSV (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write feature CSV (default: {DEFAULT_OUTPUT})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        log.error("Run scripts/join_data.py first to produce the joined CSV.")
        sys.exit(1)

    # Temperature unit sanity check — Open-Meteo returns Celsius.
    # ERCOT South Central winter lows ~5-15 °C, summer peaks ~36-42 °C.
    # Values > 60 almost certainly mean Fahrenheit was returned instead.
    df_raw     = load_raw(args.input)
    temp_max   = df_raw["temp_c"].max()
    temp_med   = df_raw["temp_c"].median()
    log.info("temp_c sanity — median: %.1f °C  max: %.1f °C", temp_med, temp_max)
    if temp_max > 60:
        log.warning(
            "temp_c max = %.1f — this looks like Fahrenheit, not Celsius! "
            "Check the Open-Meteo fetch and join script.",
            temp_max,
        )

    log.info("Building baseline feature set …")
    df_features = add_baseline_feature_set(df_raw)

    print_summary(df_features)
    save_features(df_features, args.output)


if __name__ == "__main__":
    main()
