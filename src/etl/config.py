"""
config.py — constants for the live inference-time ETL pipeline: API base
URL, weather source config, GridStatus dataset, extraction windows.

Kept separate from src/experiments/config.py, which holds training-only
constants (FEATURES lists, CV fold boundaries) plus init_mlflow(). ETL
reuses init_mlflow() directly rather than duplicating it — see pipeline.py.
"""
import os

CENTRAL_TZ = "America/Chicago"

# Where the FastAPI service (src/api/main.py) is running. Override via env
# var for staging/prod; defaults to the local dev server from its docstring
# (`uvicorn src.api.main:app --port 8000`).
API_BASE_URL = os.environ.get("FORECAST_API_BASE_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 15

# Forecast (not archive) endpoint — scripts/fetch_weather.py's archive-api.
# only has historical data, useless for a target hour that hasn't happened
# yet. Open-Meteo's limits: past_days <= 92, forecast_days <= 16.
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_MAX_PAST_DAYS = 92
OPENMETEO_MAX_FORECAST_DAYS = 16

# Same four cities/coordinates scripts/fetch_weather.py uses for historical
# backfill, and the same set src/weather_transform.py.CITY_NAMES expects —
# extract.py asserts the two stay in sync. Duplicated here (rather than
# imported from the script) because scripts/ isn't part of the importable
# src/ package; if the training city set ever changes, both places need
# updating together.
CITY_COORDS = {
    "san_antonio": {"lat": 29.4241, "lon": -98.4936},
    "austin":      {"lat": 30.2672, "lon": -97.7431},
    "round_rock":  {"lat": 30.5083, "lon": -97.6789},
    "san_marcos":  {"lat": 29.8833, "lon": -97.9414},
}

# Hours of trailing temp_c history to fetch before target_timestamp. Must
# cover both temp_c_roll_std_72's 72h window (min_periods=24) and
# temp_change_vs_lag24's 24h shift; padded well above 72 so day-boundary
# rounding in extract._openmeteo_day_windows() can't leave the window short.
WEATHER_LOOKBACK_HOURS = 96

GRIDSTATUS_LOAD_DATASET = "ercot_load_by_weather_zone"
GRIDSTATUS_LOAD_COLUMN = "south_central"
