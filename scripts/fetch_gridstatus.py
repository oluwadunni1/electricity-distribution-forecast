"""
fetch_gridstatus.py
-------------------
Fetches raw ERCOT South Central load data from the GridStatus API and saves
it as-is to data/raw/load_raw.csv for inspection.

Usage:
    uv run python scripts/fetch_gridstatus.py --start_date 2016-01-01 --end_date 2016-03-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/raw/load_raw.csv")


def parse_args() -> argparse.Namespace:
    today = date.today().isoformat()
    p = argparse.ArgumentParser(description="Fetch ERCOT load from GridStatus API.")
    p.add_argument("--start_date", default="2016-01-01")
    p.add_argument("--end_date", default=today)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        log.error("GRIDSTATUS_API_KEY not set in .env")
        sys.exit(1)

    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end   = date.fromisoformat(args.end_date)

    log.info("=" * 62)
    log.info("GridStatus fetch: %s → %s", start, end)
    log.info("=" * 62)

    client = GridStatusClient(api_key=api_key)

    try:
        df = client.get_dataset(
            dataset="ercot_load_by_weather_zone",
            start=start.isoformat(),
            end=end.isoformat(),
        )
    except Exception as exc:
        log.error("GridStatus API call failed: %s", exc)
        raise

    if df is None or df.empty:
        log.error("No data returned.")
        sys.exit(1)

    log.info("Returned columns : %s", list(df.columns))
    log.info("Raw row count    : %d", len(df))

    # ── Identify and normalise the timestamp column ───────────────────────────
    ts_candidates = [
        c for c in df.columns
        if c.lower() in ("interval_start_utc", "interval_start", "time", "timestamp")
    ]
    if not ts_candidates:
        raise RuntimeError(
            f"Cannot find timestamp column. Columns present: {list(df.columns)}"
        )
    ts_col = ts_candidates[0]
    log.info("Timestamp column used: '%s'", ts_col)

    if "south_central" not in df.columns:
        raise RuntimeError(
            f"'south_central' column missing. Columns present: {list(df.columns)}"
        )

    # ── Keep only timestamp + south_central ───────────────────────────────────
    out = df[[ts_col, "south_central"]].copy()
    out = out.rename(columns={ts_col: "timestamp", "south_central": "actual_load_mw"})

    # ── Coerce timestamp to UTC-aware ─────────────────────────────────────────
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    # ── Diagnostic prints ─────────────────────────────────────────────────────
    log.info("First timestamp  : %s", out["timestamp"].iloc[0])
    log.info("Last  timestamp  : %s", out["timestamp"].iloc[-1])

    # Show first and last 5 rows for manual inspection
    print("\n=== First 5 rows ===")
    print(out.head(5).to_string(index=False))
    print("\n=== Last 5 rows ===")
    print(out.tail(5).to_string(index=False))

    # Check for gaps
    diffs = out["timestamp"].diff().dropna()
    gaps = diffs[diffs != pd.Timedelta("1h")]
    if len(gaps):
        print(f"\n⚠  {len(gaps)} gap(s) ≠ 1 hour found:")
        for i in gaps.index[:10]:
            print(f"   {out['timestamp'].iloc[i-1]}  →  {out['timestamp'].iloc[i]}  (Δ = {diffs.iloc[i]})")
    else:
        print("\n✓  All rows are exactly 1 hour apart.")

    # ── Note UTC vs Central time offset ──────────────────────────────────────
    print("\n⚡ NOTE: All timestamps are stored in UTC.")
    print("   The grid.io website displays in US Central time (CST = UTC-6 / CDT = UTC-5).")
    print("   Example: CSV 23:00 UTC = 18:00 CDT on the website — same moment, different label.\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} rows → {OUTPUT_PATH.resolve()}")
    print(f"Columns: {list(out.columns)}\n")


if __name__ == "__main__":
    main()
