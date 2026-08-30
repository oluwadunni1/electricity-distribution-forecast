"""
extract.py — live data extraction for the inference-time ETL pipeline.

Two independent sources:

  - Open-Meteo's FORECAST API (api.open-meteo.com/v1/forecast) — NOT the
    archive API scripts/fetch_weather.py uses for historical backfill, which
    has no data beyond "now". Fetches enough per-city hourly history before
    target_timestamp to build temp_c_roll_std_72 / temp_change_vs_lag24, and
    enough of target_timestamp's local calendar day to build tmax/tmin/tavg
    the same way src/weather_transform.compute_daily_extremes() does at
    training time (grouped by Central-Time calendar day). The combining
    logic itself (population weighting, daily extremes) is NOT reimplemented
    here — it's imported from src/weather_transform.py, which already exists
    specifically to be reused by "the live daily-predict script."

  - GridStatus (same `ercot_load_by_weather_zone` dataset
    scripts/fetch_gridstatus.py bulk-fetches historically) for the two
    specific hours load_lag_24/load_lag_168 need. Both lag hours are always
    already-occurred load by the time a target hour within a realistic
    forecasting horizon (day-ahead / hour-ahead) is being scored, so this is
    a live point lookup against the same dataset, not a forecast.
"""
from __future__ import annotations

import math
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src.etl import config
from src.weather_transform import CITY_NAMES, apply_population_weights, compute_daily_extremes

assert set(config.CITY_COORDS) == set(CITY_NAMES), (
    "src/etl/config.py's CITY_COORDS and src/weather_transform.py's CITY_NAMES "
    "have drifted apart — both must name the same set of cities."
)


def _require_utc(ts: pd.Timestamp, name: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be tz-aware UTC, got a naive timestamp")


def _openmeteo_day_windows(
    target_timestamp: pd.Timestamp,
    lookback_hours: int,
    now: pd.Timestamp | None = None,
) -> tuple[int, int]:
    """
    Translates [target_timestamp - lookback_hours, end of target_timestamp's
    Central calendar day] into Open-Meteo's day-granular past_days /
    forecast_days parameters. Both of those are counted from "now" (when the
    request is made), not from target_timestamp, so this has to convert
    between the two reference points. Over-fetching a day or two of padding
    is harmless — fetch_weather_features() filters down to the exact window
    it needs afterward — under-fetching would silently truncate the rolling
    window, so this rounds up (ceil) and adds a day of padding on each side.
    """
    _require_utc(target_timestamp, "target_timestamp")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    window_start = target_timestamp - pd.Timedelta(hours=lookback_hours)
    target_central_date = target_timestamp.tz_convert(config.CENTRAL_TZ).normalize()
    window_end = (target_central_date + pd.Timedelta(days=1)).tz_convert("UTC")

    past_days = max(0, math.ceil((now.normalize() - window_start.normalize()) / pd.Timedelta(days=1))) + 1
    forecast_days = max(0, math.ceil((window_end.normalize() - now.normalize()) / pd.Timedelta(days=1))) + 1

    return (
        min(past_days, config.OPENMETEO_MAX_PAST_DAYS),
        min(forecast_days, config.OPENMETEO_MAX_FORECAST_DAYS),
    )


def _fetch_city_hourly_forecast(name: str, lat: float, lon: float, past_days: int, forecast_days: int) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    resp = requests.get(config.OPENMETEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo forecast response for '{name}' missing 'hourly': keys={list(payload.keys())}")

    hourly = payload["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        f"temp_c_{name}": hourly["temperature_2m"],
        f"humidity_pct_{name}": hourly["relative_humidity_2m"],
        f"precip_mm_{name}": hourly["precipitation"],
    })


def fetch_weather_features(
    target_timestamp: pd.Timestamp,
    lookback_hours: int = config.WEATHER_LOOKBACK_HOURS,
) -> pd.DataFrame:
    """
    Returns hourly weighted weather — columns timestamp, temp_c,
    humidity_pct, precip_mm, tmax, tmin, tavg — spanning enough history
    before target_timestamp to build the rolling/lag temp features, through
    the end of target_timestamp's Central calendar day. Uses the SAME
    weighting + daily-extremes logic as training
    (src/weather_transform.py), fed by the forecast endpoint instead of the
    archive endpoint.

    Raises RuntimeError if the fetched range doesn't actually cover
    target_timestamp (e.g. it's outside Open-Meteo's forecast horizon).
    """
    _require_utc(target_timestamp, "target_timestamp")
    past_days, forecast_days = _openmeteo_day_windows(target_timestamp, lookback_hours)

    frames = {
        name: _fetch_city_hourly_forecast(name, coords["lat"], coords["lon"], past_days, forecast_days)
        for name, coords in config.CITY_COORDS.items()
    }
    merged = frames[CITY_NAMES[0]]
    for name in CITY_NAMES[1:]:
        merged = merged.merge(frames[name], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    weighted, _weights = apply_population_weights(merged)
    with_extremes = compute_daily_extremes(weighted)
    out = with_extremes[["timestamp", "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg"]]

    window_start = target_timestamp - pd.Timedelta(hours=lookback_hours)
    target_central_date = target_timestamp.tz_convert(config.CENTRAL_TZ).normalize()
    window_end = (target_central_date + pd.Timedelta(days=1)).tz_convert("UTC")
    out = out[(out["timestamp"] >= window_start) & (out["timestamp"] < window_end)].reset_index(drop=True)

    if target_timestamp not in set(out["timestamp"]):
        raise RuntimeError(
            f"Open-Meteo forecast response doesn't cover target_timestamp={target_timestamp}. "
            f"Fetched range: {out['timestamp'].min()} .. {out['timestamp'].max()}. "
            f"Is target_timestamp within Open-Meteo's forecast horizon "
            f"(~{config.OPENMETEO_MAX_FORECAST_DAYS} days ahead)?"
        )
    return out


def _fetch_gridstatus_hour(client, target_hour: pd.Timestamp) -> float:
    """Looks up ERCOT south_central actual load for exactly one UTC hour."""
    window_end = target_hour + pd.Timedelta(hours=1)
    df = client.get_dataset(
        dataset=config.GRIDSTATUS_LOAD_DATASET,
        start=target_hour.isoformat(),
        end=window_end.isoformat(),
    )
    if df is None or df.empty:
        raise RuntimeError(f"GridStatus returned no rows for {target_hour}")
    if config.GRIDSTATUS_LOAD_COLUMN not in df.columns:
        raise RuntimeError(f"GridStatus response missing '{config.GRIDSTATUS_LOAD_COLUMN}': columns={list(df.columns)}")

    ts_col = next(
        (c for c in df.columns if c.lower() in ("interval_start_utc", "interval_start", "time", "timestamp")),
        None,
    )
    if ts_col is None:
        raise RuntimeError(f"GridStatus response has no recognizable timestamp column: columns={list(df.columns)}")

    ts = pd.to_datetime(df[ts_col], utc=True)
    row = df.loc[ts == target_hour]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly 1 GridStatus row for {target_hour}, got {len(row)}. "
            f"Timestamps returned: {sorted(ts.tolist())}"
        )
    return float(row[config.GRIDSTATUS_LOAD_COLUMN].iloc[0])


def fetch_load_lags(target_timestamp: pd.Timestamp) -> dict:
    """
    Live lookup of actual load 24h and 168h before target_timestamp, from
    the same GridStatus dataset scripts/fetch_gridstatus.py bulk-fetches
    historically. Requires GRIDSTATUS_API_KEY in the environment/.env, same
    as that script.

    Both lag hours must already be published in GridStatus, which in
    practice means target_timestamp - 24h has to be far enough in the past
    to clear GridStatus's own publish latency — NOT just "in the past" by a
    few minutes. For a target_timestamp close to real-time "now" (e.g.
    scoring the next hour), load_lag_24 is only ~1 hour old and can
    legitimately not be published yet; _fetch_gridstatus_hour() raises
    RuntimeError rather than silently propagating a missing value in that
    case. This is unrelated to load_lag_168, which for any target_timestamp
    within a normal forecasting horizon is always safely historical.
    """
    _require_utc(target_timestamp, "target_timestamp")

    from gridstatusio import GridStatusClient

    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY not set in environment/.env")
    client = GridStatusClient(api_key=api_key)

    lag_24_hour = target_timestamp - pd.Timedelta(hours=24)
    lag_168_hour = target_timestamp - pd.Timedelta(hours=168)
    return {
        "load_lag_24": _fetch_gridstatus_hour(client, lag_24_hour),
        "load_lag_168": _fetch_gridstatus_hour(client, lag_168_hour),
    }
