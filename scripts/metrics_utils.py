"""
Evaluation metrics matching the source paper's Section 3 definitions
(MAE, RMSE, MAPE, Peak-MAPE).

Peak-MAPE isolates accuracy on the top-5% highest-demand hours in the
evaluation window, matching the paper's operational focus on peak periods.
"""
import numpy as np

__all__ = ["mape", "rmse", "mae", "peak_mape", "compute_all_metrics"]


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def peak_mape(y_true, y_pred, percentile: float = 95, eps: float = 1e-6) -> float:
    """MAPE restricted to hours where y_true is at/above `percentile`
    (paper uses the top 5% of observed hourly demand, i.e. percentile=95)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    threshold = np.percentile(y_true, percentile)
    peak_idx = y_true >= threshold
    if peak_idx.sum() == 0:
        return float("nan")
    return mape(y_true[peak_idx], y_pred[peak_idx], eps=eps)


def compute_all_metrics(y_true, y_pred, peak_percentile: float = 95) -> dict:
    """Returns {mape, rmse, mae, peak_mape} — the exact set logged to
    MLflow for every run in the experiments notebook."""
    return {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "peak_mape": peak_mape(y_true, y_pred, percentile=peak_percentile),
    }
