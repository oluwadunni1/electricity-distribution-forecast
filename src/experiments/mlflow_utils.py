"""
mlflow_utils.py — one logging call instead of a repeated ~15-line block per experiment.
"""
import mlflow


def log_cv_run(run_name: str, params: dict, metrics: dict) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        print(f"Logged '{run_name}' — " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))
