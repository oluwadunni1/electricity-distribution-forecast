"""
join_data.py
------------
Reads raw load + raw per-city weather, applies population-weighting and
Central-Time daily-extreme computation (via src/weather_transform.py), then
inner-joins the combined weather onto load on UTC timestamp. Prints a full
alignment report, including a Central-Time label for manual spot-checking.

Usage:
    uv run python scripts/join_data.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `from src...` import
from src.weather_transform import apply_population_weights, compute_daily_extremes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LOAD_PATH    = Path("data/raw/load_raw.csv")
WEATHER_PATH = Path("data/raw/weather_raw.csv")
OUTPUT_PATH  = Path("data/raw/ercot_south_central_raw.csv")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.error("%s not found: %s", label, path)
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("%-20s : %6d rows  |  %s → %s", label, len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return df


def check_quality(df: pd.DataFrame, label: str) -> None:
    if df["timestamp"].dt.tz is None:
        raise RuntimeError(f"[{label}] Timestamps are tz-naive — expected UTC.")
    dups = df["timestamp"].duplicated().sum()
    if dups:
        raise RuntimeError(f"[{label}] {dups} duplicate timestamp(s) found.")
    diffs = df["timestamp"].diff().dropna()
    gaps  = diffs[diffs != pd.Timedelta("1h")]
    if len(gaps):
        details = [f"  {df['timestamp'].iloc[i-1]}  →  {df['timestamp'].iloc[i]}  (Δ = {diffs.iloc[i]})" for i in gaps.index[:5]]
        warnings.warn(f"[{label}] {len(gaps)} gap(s) > 1 h:\n" + "\n".join(details), stacklevel=2)
    else:
        log.info("[%s] ✓ No gaps.", label)


def main() -> None:
    print(f"\n{'='*62}\n  JOIN: load_raw.csv + weather_raw.csv (weighted)\n{'='*62}\n")

    load_df       = load_csv(LOAD_PATH, "GridStatus / load")
    weather_raw   = load_csv(WEATHER_PATH, "Open-Meteo / weather (per-city raw)")

    check_quality(load_df, "GridStatus / load")
    check_quality(weather_raw, "Open-Meteo / weather (per-city raw)")

    # ── Apply weighting + Central-Time daily extremes (single source of truth) ──
    weighted, weights = apply_population_weights(weather_raw)
    weather_df = compute_daily_extremes(weighted)
    weather_df = weather_df[["timestamp", "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg"]]

    print("=== Population weights by year ===")
    print(weights.to_string(float_format="%.4f"))

    # ── Alignment report BEFORE join ─────────────────────────────────────────
    print("\n=== Alignment report (before join) ===")
    load_ts, weather_ts = set(load_df["timestamp"]), set(weather_df["timestamp"])
    only_in_load    = sorted(load_ts - weather_ts)
    only_in_weather = sorted(weather_ts - load_ts)
    print(f"  Load timestamps      : {len(load_ts):,}")
    print(f"  Weather timestamps   : {len(weather_ts):,}")
    print(f"  Common (inner join)  : {len(load_ts & weather_ts):,}")
    print(f"  Only in load (dropped)    : {len(only_in_load):,}")
    print(f"  Only in weather (dropped) : {len(only_in_weather):,}")

    # ── Inner join ────────────────────────────────────────────────────────────
    joined = load_df.set_index("timestamp").join(weather_df.set_index("timestamp"), how="inner")
    joined = joined[["actual_load_mw", "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg"]]
    joined.index.name = "timestamp"
    joined = joined.reset_index()

    # ── Manual-validation column ─────────────────────────────────────────────
    joined["timestamp_central"] = joined["timestamp"].dt.tz_convert("America/Chicago")
    joined = joined[["timestamp", "timestamp_central"] + [c for c in joined.columns if c not in ("timestamp", "timestamp_central")]]

    print(f"\n=== Join result ===\n  Rows: {len(joined):,}  |  {joined['timestamp'].iloc[0]} → {joined['timestamp'].iloc[-1]}")
    print("\n=== First 5 rows ===")
    print(joined.head(5).to_string(index=False))
    print("\n=== Last 5 rows ===")
    print(joined.tail(5).to_string(index=False))

    print("\n=== Random 5 rows for manual spot-check ===")
    print(joined.sample(5, random_state=None).sort_values("timestamp").to_string(index=False))

    # ── Null check ────────────────────────────────────────────────────────────
    value_cols = ["actual_load_mw", "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg"]
    null_report = joined[value_cols].isna().sum()
    if null_report.sum() > 0:
        print("\n=== Null-value check ===")
        for col, n in null_report.items():
            if n:
                print(f"  ⚠  {col:<15} : {n:,} null value(s)")
        print(joined[joined[value_cols].isna().any(axis=1)].head(10).to_string(index=False))
    else:
        print("\n✓  No null values in any feature column.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓  Saved {len(joined):,} rows → {OUTPUT_PATH.resolve()}\n")


if __name__ == "__main__":
    main()