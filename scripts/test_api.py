"""
test_api.py

Sends each row of a preprocessed parquet slice to the running /predict API,
and separately calls the champion model directly (same data, no API in the
loop) to confirm the two match. If they diverge, the bug is in the API layer
(serialization, column ordering, Pydantic coercion) — not the model.

USAGE
-----
1) Start the API in one terminal:
    uvicorn src.api.main:app --port 8000

2) In another terminal:
    python scripts/test_api.py --input data/interim/pipeline_test_output.parquet
"""
import argparse
from datetime import datetime, timedelta, timezone

import mlflow
import pandas as pd
import requests

from src.experiments import config

API_URL = "http://localhost:8000/predict"
MODEL_NAME = "Electricity-Load-Forecaster"
MODEL_ALIAS = "champion"


def get_reference_predictions(df: pd.DataFrame) -> pd.Series:
    """Same model, called directly — no API involved."""
    config.init_mlflow()
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return pd.Series(model.predict(df), index=df.index)


def get_api_predictions(df: pd.DataFrame) -> pd.Series:
    # target_timestamp is required by the API but isn't part of this file's
    # columns (it's pure engineered-feature output — the timestamp never
    # survives engineer_features). It's metadata only, not a model input, so
    # a synthetic sequential value here doesn't affect what's being tested:
    # whether the API's prediction matches calling the model directly.
    base_ts = datetime.now(timezone.utc)

    preds = []
    for i, (_, row) in enumerate(df.iterrows()):
        payload = row.to_dict()
        payload["target_timestamp"] = (base_ts + timedelta(hours=i)).isoformat()

        resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"API returned {resp.status_code}: {resp.text}")
        preds.append(resp.json()["predicted_load_mw"])
    return pd.Series(preds, index=df.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Preprocessed parquet (e.g. pipeline_test_output.parquet)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max allowed absolute difference")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} preprocessed row(s) from {args.input}")

    print("\nCalling model directly (reference)...")
    reference = get_reference_predictions(df)

    print("Calling the running API for each row...")
    api_preds = get_api_predictions(df)

    comparison = pd.DataFrame({"reference": reference, "api": api_preds})
    comparison["abs_diff"] = (comparison["reference"] - comparison["api"]).abs()

    print("\n", comparison)

    max_diff = comparison["abs_diff"].max()
    if max_diff <= args.tolerance:
        print(f"\nMATCH — max difference {max_diff:.6f} MW, within tolerance ({args.tolerance})")
    else:
        print(f"\nMISMATCH — max difference {max_diff:.6f} MW exceeds tolerance ({args.tolerance})")
        print("Check column ordering, Pydantic type coercion, or missing fields in the API request path.")


if __name__ == "__main__":
    main()