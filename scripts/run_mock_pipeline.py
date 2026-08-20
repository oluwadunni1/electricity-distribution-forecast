"""
run_mock_pipeline.py

Simulates the real production flow end to end, minus live ingestion:
    raw feature slice -> feature_engineering.transform_for_inference() -> API /predict

Distinct from test_preprocessing.py (tests preprocessing alone) and
test_api.py (tests the API alone, given already-engineered input) — this one
chains both stages together, the way real traffic will actually flow.

USAGE
-----
1) Start the API:
    uvicorn src.api.main:app --port 8000

2) Run:
    uv run scripts/run_mock_pipeline.py --input data/interim/pipeline_test_sample.parquet
"""
import argparse

import pandas as pd
import requests

from src.experiments import config
from src.inference import feature_engineering as fe

API_URL = "http://localhost:8000/predict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw feature slice (must include `timestamp` column)")
    parser.add_argument("--output", default=None, help="Optional: save results to this CSV path")
    args = parser.parse_args()

    # Without this, mlflow defaults to a local registry store and can't find
    # anything registered on DagsHub — same fix as test_preprocessing.py.
    config.init_mlflow()

    raw = pd.read_parquet(args.input)
    print(f"Loaded {len(raw)} raw row(s) from {args.input}")

    print("\n--- Preprocessing (feature engineering) ---")
    transformed = fe.transform_for_inference(raw)
    print(f"Transformed shape: {transformed.shape}")
    print(f"Columns: {list(transformed.columns)}")

    print("\n--- Sending each row to the API ---")
    results = []
    for i, (_, row) in enumerate(transformed.iterrows()):
        payload = row.to_dict()
        # target_timestamp isn't part of the engineered feature row — it rides
        # alongside separately, same as raw["timestamp"] identifies which hour
        # this row is for.
        payload["target_timestamp"] = raw.iloc[i]["timestamp"].isoformat()

        resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"  Row {i}: FAILED ({resp.status_code}) — {resp.text}")
            continue

        body = resp.json()
        results.append(body)
        print(f"  Row {i}: {body['target_timestamp']} -> {body['predicted_load_mw']:.1f} MW "
              f"(model v{body['model_version']})")

    results_df = pd.DataFrame(results)
    print(f"\n{len(results_df)}/{len(transformed)} predictions succeeded")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()