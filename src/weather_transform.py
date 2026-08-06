"""
weather_transform.py
---------------------
Reusable weather-combination logic: population weighting and Central-Time
daily-extreme computation. Imported by both the training-time notebook/pipeline
and (later) the live daily-predict script, so there is exactly one
implementation of "how we turn 4 cities' raw readings into one South-Central
weather signal."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITY_NAMES = ["san_antonio", "austin", "round_rock", "san_marcos"]

POPULATION_ANCHORS: dict[int, dict[str, int]] = {
    2016: {"san_antonio": 1_492_510, "austin": 938_200, "round_rock": 120_955, "san_marcos": 61_782},
    2018: {"san_antonio": 1_532_233, "austin": 962_800, "round_rock": 129_046, "san_marcos": 63_802},
    2020: {"san_antonio": 1_434_625, "austin": 962_163, "round_rock": 120_556, "san_marcos": 68_199},
    2022: {"san_antonio": 1_472_909, "austin": 978_358, "round_rock": 127_340, "san_marcos": 70_572},
    2024: {"san_antonio": 1_526_656, "austin": 993_588, "round_rock": 135_665, "san_marcos": 74_319},
}


def get_yearly_weights(start_year: int, end_year: int, cities: list[str] = CITY_NAMES) -> pd.DataFrame:
    """Linearly interpolate populations between anchor years; return per-year weights summing to 1."""
    anchor_years = sorted(POPULATION_ANCHORS.keys())
    years = list(range(start_year, end_year + 1))

    pops: dict[str, np.ndarray] = {}
    for city in cities:
        anchor_pops = [POPULATION_ANCHORS[y][city] for y in anchor_years]
        pops[city] = np.interp(years, anchor_years, anchor_pops)

    df = pd.DataFrame(pops, index=years)
    df.index.name = "year"
    return df.div(df.sum(axis=1), axis=0)


def apply_population_weights(
    raw_hourly: pd.DataFrame,
    cities: list[str] = CITY_NAMES,
) -> pd.DataFrame:
    """
    Take the wide per-city raw frame (timestamp, temp_c_<city>, humidity_pct_<city>,
    precip_mm_<city> for each city) and produce weighted temp_c/humidity_pct/precip_mm.
    """
    df = raw_hourly.copy()
    start_year = df["timestamp"].dt.year.min()
    end_year   = df["timestamp"].dt.year.max()
    weights = get_yearly_weights(start_year, end_year, cities)

    df["_year"] = df["timestamp"].dt.year
    for field in ("temp_c", "humidity_pct", "precip_mm"):
        df[field] = sum(
            df[f"{field}_{city}"] * df["_year"].map(weights[city])
            for city in cities
        )

    return df, weights


def compute_daily_extremes(weighted_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tmax/tmin/tavg by grouping the ALREADY-WEIGHTED hourly temp_c series
    on the Central-Time calendar day, then broadcast back onto every hourly row
    of that day. No second API call — avoids any UTC/Central daily-window bug.
    """
    df = weighted_hourly.copy()
    df["_date_central"] = df["timestamp"].dt.tz_convert("America/Chicago").dt.normalize()

    daily = df.groupby("_date_central")["temp_c"].agg(tmax="max", tmin="min", tavg="mean").reset_index()

    df = df.merge(daily, on="_date_central", how="left")
    df = df.drop(columns=["_date_central"])
    return df
    