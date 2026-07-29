"""
join_data.py
------------
Reads data/raw/load_raw.csv and data/raw/weather_raw.csv (produced by
fetch_gridstatus.py and fetch_weather.py respectively), inner-joins them on
UTC timestamp, and writes the result to data/raw/ercot_south_central_raw.csv.

Prints a detailed alignment report so you can trace any mismatches.

Usage:
    uv run python scripts/join_data.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LOAD_PATH    = Path("data/raw/load_raw.csv")
WEATHER_PATH = Path("data/raw/weather_raw.csv")
OUTPUT_PATH  = Path("data/raw/ercot_south_central_raw.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.error("%s not found: %s", label, path)
        log.error("Run %s first.", "fetch_gridstatus.py" if "load" in str(path) else "fetch_weather.py")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("%-20s : %6d rows  |  %s → %s",
             label, len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return df


def check_quality(df: pd.DataFrame, label: str) -> None:
    """Raise on duplicates / tz issues; warn on gaps."""
    if df["timestamp"].dt.tz is None:
        raise RuntimeError(f"[{label}] Timestamps are tz-naive — expected UTC.")

    dups = df["timestamp"].duplicated().sum()
    if dups:
        raise RuntimeError(f"[{label}] {dups} duplicate timestamp(s) found.")

    diffs = df["timestamp"].diff().dropna()
    gaps  = diffs[diffs != pd.Timedelta("1h")]
    if len(gaps):
        details = []
        for i in gaps.index[:5]:
            details.append(f"  {df['timestamp'].iloc[i-1]}  →  {df['timestamp'].iloc[i]}  (Δ = {diffs.iloc[i]})")
        warnings.warn(
            f"[{label}] {len(gaps)} gap(s) > 1 h:\n" + "\n".join(details),
            stacklevel=2,
        )
    else:
        log.info("[%s] ✓ No gaps — all rows exactly 1 hour apart.", label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*62}")
    print("  JOIN: load_raw.csv + weather_raw.csv")
    print(f"{'='*62}\n")

    # ── Read both sources ─────────────────────────────────────────────────────
    load_df    = load_csv(LOAD_PATH,    "GridStatus / load")
    weather_df = load_csv(WEATHER_PATH, "Open-Meteo / weather")

    # ── Quality checks ────────────────────────────────────────────────────────
    check_quality(load_df,    "GridStatus / load")
    check_quality(weather_df, "Open-Meteo / weather")

    # ── Alignment report BEFORE join ─────────────────────────────────────────
    print("\n=== Alignment report (before join) ===")

    load_ts    = set(load_df["timestamp"])
    weather_ts = set(weather_df["timestamp"])

    only_in_load    = sorted(load_ts    - weather_ts)
    only_in_weather = sorted(weather_ts - load_ts)

    print(f"  Load timestamps      : {len(load_ts):,}")
    print(f"  Weather timestamps   : {len(weather_ts):,}")
    print(f"  Common (inner join)  : {len(load_ts & weather_ts):,}")
    print(f"  Only in load (dropped by inner join)    : {len(only_in_load):,}")
    print(f"  Only in weather (dropped by inner join) : {len(only_in_weather):,}")

    if only_in_load:
        print(f"\n  First 5 load-only timestamps (no weather match):")
        for ts in only_in_load[:5]:
            row = load_df[load_df["timestamp"] == ts][["timestamp", "actual_load_mw"]].iloc[0]
            print(f"    {row['timestamp']}  actual_load_mw={row['actual_load_mw']}")

    if only_in_weather:
        print(f"\n  First 5 weather-only timestamps (no load match):")
        for ts in only_in_weather[:5]:
            row = weather_df[weather_df["timestamp"] == ts][["timestamp", "temp_c", "humidity_pct", "precip_mm"]].iloc[0]
            print(f"    {row['timestamp']}  temp_c={row['temp_c']}  humidity={row['humidity_pct']}  precip={row['precip_mm']}")

    # ── UTC vs Central time note ──────────────────────────────────────────────
    print("\n⚡ UTC ↔ Central Time reference:")
    print("   CST = UTC − 6 h  (Nov–Mar)")
    print("   CDT = UTC − 5 h  (Mar–Nov)")
    print("   e.g. CSV 23:00 UTC  =  18:00 CDT on the grid.io website")
    print("        CSV 06:00 UTC  =  00:00 CST on the grid.io website")

    # ── Inner join ────────────────────────────────────────────────────────────
    load_idx    = load_df.set_index("timestamp")
    weather_idx = weather_df.set_index("timestamp")
    joined      = load_idx.join(weather_idx, how="inner")

    # Enforce output column order
    joined = joined[["actual_load_mw", "temp_c", "humidity_pct", "precip_mm"]]
    joined.index.name = "timestamp"
    joined = joined.reset_index()

    print(f"\n=== Join result ===")
    print(f"  Rows in output   : {len(joined):,}")
    print(f"  First timestamp  : {joined['timestamp'].iloc[0]}")
    print(f"  Last  timestamp  : {joined['timestamp'].iloc[-1]}")

    print("\n=== First 5 rows of joined output ===")
    print(joined.head(5).to_string(index=False))
    print("\n=== Last 5 rows of joined output ===")
    print(joined.tail(5).to_string(index=False))

    # ── Sample mid-range rows for spot inspection ─────────────────────────────
    mid = len(joined) // 2
    print(f"\n=== 5 rows from the middle (row {mid}) ===")
    print(joined.iloc[mid-2 : mid+3].to_string(index=False))

    # ── Null-value check (a row can exist with a missing value — not just a
    #    missing/gapped timestamp, which the earlier gap check already covers) ─
    print("\n=== Null-value check ===")
    value_cols = ["actual_load_mw", "temp_c", "humidity_pct", "precip_mm"]
    null_report = joined[value_cols].isna().sum()
    any_nulls = null_report.sum() > 0
    if any_nulls:
        for col, n in null_report.items():
            if n:
                print(f"  ⚠  {col:<15} : {n:,} null value(s)")
        null_rows = joined[joined[value_cols].isna().any(axis=1)]
        print(f"\n  First 5 rows with a null value:")
        print(null_rows.head(5).to_string(index=False))
    else:
        print("  ✓ No null values in actual_load_mw, temp_c, humidity_pct, or precip_mm.")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓  Saved {len(joined):,} rows → {OUTPUT_PATH.resolve()}\n")
    print("=" * 62)


if __name__ == "__main__":
    main()