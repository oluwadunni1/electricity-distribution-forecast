"""
test_preprocessing.py

Standalone test for the inference preprocessing pipeline (feature_engineering.py),
independent of the API and independent of live traffic. Run this whenever:
  - you change engineer_features() or the RAW_REQUIRED_COLUMNS contract
  - the champion model alias moves, to confirm the pipeline still reads its
    thresholds/holiday_freq_map correctly
  - you're about to demo the pipeline and want a known-good output on hand

USAGE
-----
1) From the notebook, after the champion fit, save a raw slice to test against:

    sample = test[feature_engineering.RAW_REQUIRED_COLUMNS].sample(20, random_state=42)
    sample.to_parquet("../data/interim/pipeline_test_sample.parquet", index=False)

2) Then run this script:

    python scripts/test_preprocessing.py \\
        --input data/interim/pipeline_test_sample.parquet \\
        --output data/interim/pipeline_test_output.parquet

    # or, to test offline without hitting MLflow (e.g. no network / CI):
    python scripts/test_preprocessing.py \\
        --input data/interim/pipeline_test_sample.parquet \\
        --output data/interim/pipeline_test_output.parquet \\
        --local-bundle artifacts/electricity_load_forecaster.pkl
"""
import argparse
import sys

import joblib
import pandas as pd

from src.experiments import config
from src.inference import feature_engineering as fe


def load_artifacts_locally(bundle_path: str) -> dict:
    """Bypasses MLflow — reads thresholds/holiday_freq_map straight from a local joblib bundle."""
    bundle = joblib.load(bundle_path)
    return {
        "thresholds": bundle["thresholds"],
        "holiday_freq_map": bundle.get("holiday_freq_map"),
        "use_holiday_feature": "holiday_freq" in bundle.get("features", []),
        "model_version_tag": bundle.get("feature_version", "unknown"),
        "expected_features": bundle["features"],
    }


def validate_output(transformed: pd.DataFrame, expected_features: list[str] | None) -> list[str]:
    """Returns a list of problems found (empty list = clean)."""
    problems = []

    n_nan = transformed.isna().sum().sum()
    if n_nan:
        problems.append(f"{n_nan} NaN value(s) in transformed output")

    if expected_features is not None:
        model_cols = [c for c in transformed.columns if c != "trend_idx"]
        missing = set(expected_features) - set(model_cols)
        extra = set(model_cols) - set(expected_features)
        if missing:
            problems.append(f"missing columns the model expects: {sorted(missing)}")
        if extra:
            problems.append(f"unexpected extra columns: {sorted(extra)}")

    non_numeric = transformed.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        problems.append(f"non-numeric columns (model can't consume these): {non_numeric}")

    return problems


def main():
    parser = argparse.ArgumentParser(description="Test the inference preprocessing pipeline on a raw data slice.")
    parser.add_argument("--input", required=True, help="Path to a raw-columns parquet slice (see module docstring).")
    parser.add_argument("--output", required=True, help="Where to save the transformed, model-ready output.")
    parser.add_argument(
        "--local-bundle",
        default=None,
        help="Optional: path to a local joblib model bundle, to test without hitting MLflow.",
    )
    args = parser.parse_args()

    raw = pd.read_parquet(args.input)
    print(f"Loaded {len(raw)} raw row(s) from {args.input}")

    if args.local_bundle:
        artifacts = load_artifacts_locally(args.local_bundle)
        print(f"Using LOCAL bundle: {args.local_bundle}  (feature_version={artifacts['model_version_tag']})")
    else:
        # MLflow has no state until this is called — without it, mlflow defaults
        # to a local registry store, which is what produced the "Registered
        # Model ... not found" error (it was never looking at DagsHub at all).
        config.init_mlflow()
        artifacts = fe.load_champion_artifacts()
        print("Using LIVE @champion artifacts from MLflow")

    print(f"use_holiday_feature: {artifacts['use_holiday_feature']}")

    transformed = fe.transform_for_inference(raw, artifacts=artifacts)

    problems = validate_output(transformed, artifacts.get("expected_features"))
    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print(f"\nValidation passed. Output shape: {transformed.shape}")
    print(transformed.head())

    transformed.to_parquet(args.output, index=False)
    print(f"\nSaved transformed output to {args.output}")


if __name__ == "__main__":
    main()