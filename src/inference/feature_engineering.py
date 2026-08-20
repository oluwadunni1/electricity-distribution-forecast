"""
feature_engineering.py

Inference-side preprocessing: applies the same feature derivation used at
training time (src/experiments/features.py) to new raw rows before they go
to the model. The threshold-fitting side lives ONLY in src/experiments/features.py
now — this module just applies whatever thresholds/holiday_freq_map the
champion model's bundle was saved with, so training and inference can never
define the fit-time logic in two places and drift apart.
"""

from __future__ import annotations
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Feature definitions — must stay identical to what the model was trained on
# ---------------------------------------------------------------------------

FEATURES_V2 = [
    "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg",
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "load_lag_24", "load_lag_168", "hour_x_temp", "temp_c_roll_std_72",
    "is_extreme_heat_event", "is_extreme_cold_event", "is_holiday_x_extreme",
]

FEATURES_V3 = FEATURES_V2 + ["temp_change_vs_lag24", "is_high_precip_event"]

# Raw columns that must already exist BEFORE this module runs — i.e. produced
# by upstream ingestion/ETL (including any lag/rolling features), not derived here.
# trend_idx is NOT in this list — see compute_trend_idx() below, it's derived
# here from target_timestamp + the bundle's trend_idx_origin, not supplied raw.
RAW_REQUIRED_COLUMNS = [
    "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg",
    "hour", "dayofweek", "month", "is_weekend", "is_holiday", "holiday_name",
    "load_lag_24", "load_lag_168", "hour_x_temp", "temp_c_roll_std_72",
    "temp_change_vs_lag24",
]


def compute_trend_idx(timestamp: pd.Timestamp, trend_idx_origin: pd.Timestamp) -> float:
    """
    trend_idx = years elapsed since the ORIGINAL TRAINING SET's earliest
    timestamp — a fixed anchor, not something recomputed relative to whatever
    batch of data happens to be passed in. trend_idx_origin must come from the
    champion bundle (see load_champion_artifacts), never recomputed locally —
    recomputing it from a new batch's own min() would silently produce
    trend_idx=0 for every row, which is wrong but raises no error.
    """
    return (timestamp - trend_idx_origin).total_seconds() / (3600 * 24 * 365)


# ---------------------------------------------------------------------------
# TRANSFORM — shared by train, test, and live inference. Pure function:
# given raw rows + already-fit thresholds/freq_map, produce the model-ready row.
# `thresholds` keys (heat/cold/precip) must match src/experiments/features.py's
# fit_extreme_event_thresholds() — that's the only place these are fit.
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    thresholds: dict,
    trend_idx_origin: pd.Timestamp,
    holiday_freq_map: dict | None = None,
    use_holiday_feature: bool = True,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Apply the exact derivation logic used at training time.
    `thresholds` / `holiday_freq_map` / `trend_idx_origin` must come from an
    artifact fit on TRAIN — never recomputed here. That's what keeps inference
    in sync with whichever model version produced them.

    `df` must contain RAW_REQUIRED_COLUMNS plus a `timestamp_col` column
    (the target hour being forecasted) — trend_idx is computed from that,
    not supplied as a raw input.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if timestamp_col not in df.columns:
        missing.append(timestamp_col)
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    out = df.copy()

    out["trend_idx"] = out[timestamp_col].apply(lambda ts: compute_trend_idx(ts, trend_idx_origin))

    out["is_extreme_heat_event"] = (out["tmax"] > thresholds["heat"]).astype(int)
    out["is_extreme_cold_event"] = (out["tmin"] < thresholds["cold"]).astype(int)
    out["is_holiday_x_extreme"] = (
        out["is_holiday"] * (out["is_extreme_heat_event"] | out["is_extreme_cold_event"])
    ).astype(int)
    out["is_high_precip_event"] = (out["precip_mm"] > thresholds["precip"]).astype(int)

    if use_holiday_feature:
        if holiday_freq_map is None:
            raise ValueError("use_holiday_feature=True requires holiday_freq_map")
        out["holiday_freq"] = out["holiday_name"].map(holiday_freq_map).fillna(0.0)
        winning_features = FEATURES_V3 + ["holiday_freq"]
    else:
        winning_features = FEATURES_V3

    model_input_cols = winning_features + ["trend_idx"]
    return out[model_input_cols]


# ---------------------------------------------------------------------------
# INFERENCE-SIDE loader — pulls thresholds/freq_map from the CHAMPION bundle.
# Requires the training script to add "thresholds" and "holiday_freq_map"
# keys to `production_bundle` before logging (see note at bottom of file).
# ---------------------------------------------------------------------------

def load_champion_artifacts(
    model_uri: str = "models:/Electricity-Load-Forecaster@champion",
) -> dict:
    """
    Loads the joblib bundle backing the current @champion model version and
    returns the thresholds + holiday_freq_map it was fit with.
    """
    import mlflow

    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    bundle_path = f"{local_dir}/artifacts/electricity_load_forecaster.pkl"
    bundle = joblib.load(bundle_path)

    return {
        "thresholds": bundle["thresholds"],
        "holiday_freq_map": bundle["holiday_freq_map"],
        "trend_idx_origin": bundle["trend_idx_origin"],
        "use_holiday_feature": "holiday_freq" in bundle.get("features", []),
        "model_version_tag": bundle.get("feature_version", "unknown"),
        "expected_features": bundle["features"],
    }


# ---------------------------------------------------------------------------
# End-to-end inference transform
# ---------------------------------------------------------------------------

def transform_for_inference(raw_df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    """
    artifacts: pass explicitly to test locally against sliced train/test data
    (skips the MLflow round-trip). Leave None to pull live from @champion.
    """
    if artifacts is None:
        artifacts = load_champion_artifacts()

    return engineer_features(
        raw_df,
        thresholds=artifacts["thresholds"],
        trend_idx_origin=artifacts["trend_idx_origin"],
        holiday_freq_map=artifacts.get("holiday_freq_map"),
        use_holiday_feature=artifacts.get("use_holiday_feature", True),
    )


# ---------------------------------------------------------------------------
# NOTE — training script requirement:
# production_bundle must include these three keys before mlflow.pyfunc.log_model():
#   production_bundle["thresholds"] = thresholds        # from features.fit_extreme_event_thresholds()
#   production_bundle["holiday_freq_map"] = holiday_freq_map  # from features.fit_holiday_freq_map(), or None
#   production_bundle["trend_idx_origin"] = df["timestamp"].min()  # the FULL df's min, pre-split — see add_base_features()
# Without this, load_champion_artifacts() has nothing to load, and trend_idx
# can't be reproduced correctly for new data.
# ---------------------------------------------------------------------------