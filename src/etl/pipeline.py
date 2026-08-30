"""
pipeline.py — orchestrates the live inference-time ETL for one target hour:

    extract.py (weather forecast + load lags)
      -> src/common/timestamps.py (calendar + holiday + temp-derived features)
      -> src/inference/feature_engineering.transform_for_inference()
      -> POST src/api/main.py's /predict

Each arrow above is a stage boundary; src/etl/validation.py runs immediately
after every stage that produces a DataFrame, so a bad row is rejected at the
stage that produced it — not three stages downstream, inside the model.

USAGE
-----
    uv run python -m src.etl.pipeline --target-timestamp 2026-09-01T18:00:00Z

Requires the `api` and `experiments` optional-dependency groups (this module
bridges both worlds: mlflow + pydantic from `api`, requests + gridstatusio
from `experiments`) and a running API (see src/api/main.py's docstring).
"""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from src.api.schema import build_request_model
from src.common import timestamps
from src.etl import config, extract, validation
from src.experiments import config as exp_config
from src.inference import feature_engineering as fe


def _require_utc(ts: pd.Timestamp, name: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be tz-aware UTC, got a naive timestamp")


def build_raw_feature_row(target_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Extracts + assembles the single raw feature row target_timestamp needs —
    every column feature_engineering.RAW_REQUIRED_COLUMNS expects, plus
    `timestamp`. Validates the schema at each stage boundary; raises
    immediately on any mismatch rather than letting a bad value reach
    transform_for_inference().
    """
    _require_utc(target_timestamp, "target_timestamp")

    # --- Stage 1: weather (forecast + recent history) ----------------------
    weather = extract.fetch_weather_features(target_timestamp)
    validation.validate_dataframe(weather, validation.WEATHER_SPEC, stage="extract.fetch_weather_features")

    # --- Stage 2: calendar/holiday/temp-derived features --------------------
    # The one place hour/dayofweek/month/is_weekend/holiday_name/is_holiday
    # are derived — from timestamp_central, not the UTC `timestamp` column.
    # See src/common/timestamps.py's module docstring for why that
    # distinction matters here specifically.
    enriched = timestamps.add_calendar_and_holiday_features(weather, timestamp_col="timestamp")
    enriched = timestamps.add_temp_derived_features(enriched, temp_col="temp_c", hour_col="hour")
    validation.validate_dataframe(enriched, validation.CALENDAR_SPEC, stage="common.timestamps")

    target_row = enriched.loc[enriched["timestamp"] == target_timestamp].reset_index(drop=True)
    if len(target_row) != 1:
        raise RuntimeError(
            f"Expected exactly 1 row for target_timestamp={target_timestamp} after "
            f"feature assembly, got {len(target_row)}."
        )
    # The fetched window's EARLIEST rows are expected to have NaN
    # roll_std/temp_change (they exist only to build the rolling window) —
    # but the TARGET row itself must not, or the lookback window was too
    # short and this must fail loudly, not send a NaN downstream.
    warmup_cols = ["temp_c_roll_std_72", "temp_change_vs_lag24"]
    if target_row[warmup_cols].isna().any(axis=None):
        raise RuntimeError(
            f"Rolling/lag temp feature(s) {warmup_cols} are NaN for "
            f"target_timestamp={target_timestamp} — the fetched weather window "
            f"didn't provide enough lookback history."
        )

    # --- Stage 3: load lags ---------------------------------------------------
    lags = extract.fetch_load_lags(target_timestamp)
    validation.validate_dataframe(pd.DataFrame([lags]), validation.LOAD_LAG_SPEC, stage="extract.fetch_load_lags")
    for col, value in lags.items():
        target_row[col] = value

    return target_row[fe.RAW_REQUIRED_COLUMNS + ["timestamp"]]


def predict(target_timestamp: pd.Timestamp, api_base_url: str = config.API_BASE_URL) -> dict:
    """Runs the full pipeline for one target hour and returns the API's prediction response."""
    raw = build_raw_feature_row(target_timestamp)

    exp_config.init_mlflow()
    artifacts = fe.load_champion_artifacts()
    transformed = fe.transform_for_inference(raw, artifacts=artifacts)

    # Validate against the EXACT contract the API enforces (built from the
    # same champion artifacts) rather than a hand-maintained copy of it —
    # see src/api/schema.py::build_request_model, imported here read-only.
    model_input_columns = artifacts["expected_features"] + ["trend_idx"]
    RequestModel = build_request_model(model_input_columns)
    payload = transformed.iloc[0].to_dict()
    payload["target_timestamp"] = target_timestamp.isoformat()
    validated = RequestModel(**payload)

    resp = requests.post(
        f"{api_base_url}/predict",
        json=validated.model_dump(mode="json"),
        timeout=config.API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_target_timestamp(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live inference-time ETL pipeline for one target hour.")
    parser.add_argument("--target-timestamp", required=True, help="ISO8601 timestamp of the hour to forecast (UTC if no offset given)")
    parser.add_argument("--api-base-url", default=config.API_BASE_URL)
    args = parser.parse_args()

    target_timestamp = _parse_target_timestamp(args.target_timestamp)
    result = predict(target_timestamp, api_base_url=args.api_base_url)
    print(result)


if __name__ == "__main__":
    main()
