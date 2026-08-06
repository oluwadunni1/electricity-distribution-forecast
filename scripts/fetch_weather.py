"""
fetch_weather.py
----------------
Fetches raw hourly weather (temperature, humidity, precipitation) for four
representative South Central Texas cities from the Open-Meteo archive API
and saves them side-by-side, UTC-indexed, with NO weighting or aggregation
applied. Combining/weighting/daily-extremes logic lives in
src/weather_transform.py so it can be reused identically at inference time.

Output columns:
    timestamp, timestamp_central,
    temp_c_<city>, humidity_pct_<city>, precip_mm_<city>  (x4 cities)

Usage:
    uv run python scripts/fetch_weather.py --start_date 2016-01-01 --end_date 2016-03-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import requests_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CITY_COORDS = {
    "san_antonio": {"lat": 29.4241, "lon": -98.4936},
    "austin":      {"lat": 30.2672, "lon": -97.7431},
    "round_rock":  {"lat": 30.5083, "lon": -97.6789},
    "san_marcos":  {"lat": 29.8833, "lon": -97.9414},
}

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_PATH   = Path("data/raw/weather_raw.csv")


def parse_args() -> argparse.Namespace:
    today = date.today().isoformat()
    p = argparse.ArgumentParser(description="Fetch raw per-city hourly weather from Open-Meteo.")
    p.add_argument("--start_date", default="2016-01-01")
    p.add_argument("--end_date", default=today)
    return p.parse_args()


def fetch_city_hourly(
    cache_session: requests_cache.CachedSession,
    name: str, lat: float, lon: float,
    start: date, end: date,
) -> pd.DataFrame:
    """Fetch hourly temp, humidity, precip for a single city (UTC, instantaneous values)."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "hourly":     "temperature_2m,relative_humidity_2m,precipitation",
        "timezone":   "UTC",
    }

    log.info("Fetching hourly %s (lat=%.4f, lon=%.4f) ...", name, lat, lon)
    for attempt in range(3):
        try:
            resp = cache_session.get(OPENMETEO_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout:
            if attempt == 2:
                log.error("Request timed out for %s (hourly) after 3 attempts.", name)
                sys.exit(1)
            log.warning("Timeout for %s (hourly), retrying (%d/3)...", name, attempt + 2)
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP %s for %s (hourly): %s", exc.response.status_code, name, exc.response.text[:300])
            sys.exit(1)

    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"No 'hourly' key in response for {name}. Keys: {list(payload.keys())}")

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp":            pd.to_datetime(hourly["time"], utc=True),
        f"temp_c_{name}":       hourly["temperature_2m"],
        f"humidity_pct_{name}": hourly["relative_humidity_2m"],
        f"precip_mm_{name}":    hourly["precipitation"],
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("  → %d hourly rows for %s", len(df), name)
    return df


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end   = date.fromisoformat(args.end_date)

    log.info("=" * 72)
    log.info("Open-Meteo raw fetch (per-city, unweighted): %s → %s", start, end)
    log.info("Cities: %s", ", ".join(CITY_COORDS.keys()))
    log.info("=" * 72)

    cache_session = requests_cache.CachedSession(".cache/openmeteo_cache", expire_after=86_400)

    names = list(CITY_COORDS.keys())
    hourly_frames: dict[str, pd.DataFrame] = {}
    for name, coords in CITY_COORDS.items():
        hourly_frames[name] = fetch_city_hourly(
            cache_session, name, coords["lat"], coords["lon"], start, end,
        )

    # ── Merge all cities side-by-side on timestamp ───────────────────────────
    merged = hourly_frames[names[0]]
    for name in names[1:]:
        merged = merged.merge(hourly_frames[name], on="timestamp", how="inner")

    row_counts = {name: len(df) for name, df in hourly_frames.items()}
    if len(set(row_counts.values())) > 1 or len(merged) != row_counts[names[0]]:
        log.warning(
            "Per-city row counts didn't all match (%s) — merged=%d. Check for gaps in one city's response.",
            row_counts, len(merged),
        )

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # ── Manual-validation column: same instant, Central-Time label ──────────
    merged["timestamp_central"] = merged["timestamp"].dt.tz_convert("America/Chicago")

    # Reorder for readability
    cols = ["timestamp", "timestamp_central"] + [c for c in merged.columns if c not in ("timestamp", "timestamp_central")]
    merged = merged[cols]

    # ── Gap / duplicate checks ────────────────────────────────────────────────
    diffs = merged["timestamp"].diff().dropna()
    gaps  = diffs[diffs != pd.Timedelta("1h")]
    dupes = merged["timestamp"].duplicated().sum()

    print("\n=== First 5 rows ===")
    print(merged.head(5).to_string(index=False))
    print("\n=== Last 5 rows ===")
    print(merged.tail(5).to_string(index=False))

    if dupes:
        print(f"\n⚠  {dupes} duplicate timestamp(s) found.")
    if len(gaps):
        print(f"\n⚠  {len(gaps)} gap(s) ≠ 1 hour found:")
        for i in gaps.index[:10]:
            print(f"   {merged['timestamp'].iloc[i-1]}  →  {merged['timestamp'].iloc[i]}  (Δ = {diffs.iloc[i]})")
    if not dupes and not len(gaps):
        print("\n✓  All rows exactly 1 hour apart, no duplicates.")

    print("\n⚡ NOTE: 'timestamp' is UTC (canonical, used for all joins/lags).")
    print("   'timestamp_central' is a derived label for manual cross-checking against")
    print("   grid.io or NOAA's Central-Time displays — do not use it for merges.\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(merged):,} rows → {OUTPUT_PATH.resolve()}")
    print(f"Columns: {list(merged.columns)}\n")


if __name__ == "__main__":
    main()