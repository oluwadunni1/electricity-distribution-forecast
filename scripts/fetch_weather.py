"""
fetch_weather.py
----------------
Fetches raw hourly weather data for four representative South Central Texas
cities (Austin, San Antonio, Round Rock, San Marcos) from the Open-Meteo
archive API, combines them into a single South-Central-zone weather series
using a population-weighted average, and saves the result to
data/raw/weather_raw.csv for inspection.

This directly mirrors the multi-city, population-weighted aggregation
approach used in the ERCOT SCENT SHAP paper this project is based on —
see the note at the bottom of this file for details and sourcing.

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

# ── Representative cities for the South Central weather zone ─────────────────
# Coordinates + 2024 Census/ACS population estimates, used to compute weights.
CITIES = {
    "san_antonio": {"lat": 29.4241, "lon": -98.4936, "population": 1_479_835},
    "austin":      {"lat": 30.2672, "lon": -97.7431, "population":   979_539},
    "round_rock":  {"lat": 30.5083, "lon": -97.6789, "population":   135_665},
    "san_marcos":  {"lat": 29.8833, "lon": -97.9414, "population":    74_319},
}

TOTAL_POPULATION = sum(c["population"] for c in CITIES.values())
for _name, _info in CITIES.items():
    _info["weight"] = _info["population"] / TOTAL_POPULATION

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_PATH   = Path("data/raw/weather_raw.csv")


def parse_args() -> argparse.Namespace:
    today = date.today().isoformat()
    p = argparse.ArgumentParser(
        description="Fetch and population-weight-average weather from Open-Meteo for South Central TX."
    )
    p.add_argument("--start_date", default="2016-01-01")
    p.add_argument("--end_date", default=today)
    return p.parse_args()


def fetch_city(cache_session: requests_cache.CachedSession, name: str, lat: float, lon: float,
               start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "hourly":     "temperature_2m,relative_humidity_2m,precipitation",
        "timezone":   "UTC",
    }

    log.info("Fetching %s (lat=%.4f, lon=%.4f) ...", name, lat, lon)
    try:
        resp = cache_session.get(OPENMETEO_URL, params=params, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.error("Request timed out for %s.", name)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP %s for %s: %s", exc.response.status_code, name, exc.response.text[:300])
        sys.exit(1)

    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"No 'hourly' key in response for {name}. Keys: {list(payload.keys())}")

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp":               pd.to_datetime(hourly["time"], utc=True),
        f"temp_c_{name}":          hourly["temperature_2m"],
        f"humidity_pct_{name}":    hourly["relative_humidity_2m"],
        f"precip_mm_{name}":       hourly["precipitation"],
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("  → %d rows for %s", len(df), name)
    return df


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end   = date.fromisoformat(args.end_date)

    log.info("=" * 62)
    log.info("Open-Meteo fetch (South Central proxy): %s → %s", start, end)
    log.info("Cities and population weights:")
    for name, info in CITIES.items():
        log.info("  %-12s pop=%9d  weight=%.4f", name, info["population"], info["weight"])
    log.info("=" * 62)

    cache_session = requests_cache.CachedSession(
        ".cache/openmeteo_cache", expire_after=86_400
    )

    # ── Fetch each city separately ────────────────────────────────────────────
    city_frames = {
        name: fetch_city(cache_session, name, coords["lat"], coords["lon"], start, end)
        for name, coords in CITIES.items()
    }

    # ── Merge all four on timestamp (inner join — should match exactly, all
    #    queried for the same range/timezone) ─────────────────────────────────
    names = list(CITIES.keys())
    merged = city_frames[names[0]]
    for name in names[1:]:
        merged = merged.merge(city_frames[name], on="timestamp", how="inner")

    row_counts = {name: len(df) for name, df in city_frames.items()}
    if len(set(row_counts.values())) > 1 or len(merged) != row_counts[names[0]]:
        log.warning(
            "City row counts didn't all match (%s) — merged=%d. Check for gaps in one city's response.",
            row_counts, len(merged),
        )

    # ── Population-weighted average across the four cities ───────────────────
    for field in ("temp_c", "humidity_pct", "precip_mm"):
        merged[field] = sum(
            merged[f"{field}_{name}"] * CITIES[name]["weight"] for name in names
        )

    out = merged[["timestamp", "temp_c", "humidity_pct", "precip_mm"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    log.info("Weighted-average row count : %d", len(out))
    log.info("First ts UTC               : %s", out["timestamp"].iloc[0])
    log.info("Last  ts UTC               : %s", out["timestamp"].iloc[-1])

    # ── Diagnostic prints ─────────────────────────────────────────────────────
    print("\n=== First 5 rows (population-weighted average) ===")
    print(out.head(5).to_string(index=False))
    print("\n=== Last 5 rows (population-weighted average) ===")
    print(out.tail(5).to_string(index=False))

    # Gap / duplicate checks
    diffs = out["timestamp"].diff().dropna()
    gaps = diffs[diffs != pd.Timedelta("1h")]
    dupes = out["timestamp"].duplicated().sum()

    if dupes:
        print(f"\n⚠  {dupes} duplicate timestamp(s) found.")
    if len(gaps):
        print(f"\n⚠  {len(gaps)} gap(s) ≠ 1 hour found (excluding duplicates):")
        for i in gaps.index[:10]:
            print(f"   {out['timestamp'].iloc[i-1]}  →  {out['timestamp'].iloc[i]}  (Δ = {diffs.iloc[i]})")
    if not dupes and not len(gaps):
        print("\n✓  All rows are exactly 1 hour apart, no duplicates.")

    print("\n⚡ NOTE: timezone='UTC' was sent to the API for all four cities — all timestamps are UTC.")
    print("   Values are a population-weighted average across Austin, San Antonio, Round Rock,")
    print("   and San Marcos, used as a proxy for the ERCOT South Central weather zone.\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} rows → {OUTPUT_PATH.resolve()}")
    print(f"Columns: {list(out.columns)}\n")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY NOTE — relationship to the source research paper
# ─────────────────────────────────────────────────────────────────────────────
# The ERCOT SCENT SHAP paper this project draws on notes that "since a unified
# weather dataset does not exist for the SCENT region as a whole," the authors
# extracted weather from four representative cities — Austin, San Antonio,
# Round Rock, and San Marcos — and combined them using a population-weighted
# average, so more populous cities have proportionally more influence on the
# region-level weather signal used for load forecasting.
#
# This script mirrors that approach directly: each city's population share
# (2024 Census/ACS estimates) determines its weight in the final average.
# San Antonio (the largest of the four) carries the most weight (~55%),
# followed by Austin (~37%), with Round Rock and San Marcos contributing
# smaller shares (~5% and ~3% respectively) reflecting their smaller
# populations relative to the two anchor cities.
#
# Population figures should be revisited periodically (e.g. against annual
# Census Bureau QuickFacts updates) if this project is extended or re-run
# well beyond the initial build, since city populations — especially
# Round Rock and San Marcos, both fast-growing — will shift over time.
# ─────────────────────────────────────────────────────────────────────────────