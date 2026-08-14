"""
config.py — all constants in one place. Previously TARGET/FEATURES/fold_boundaries
were scattered across cells 6, 8, and 10, and FEATURES_V2 was redefined identically
in two different cells (10 and 20).
"""
import os
import mlflow
from dotenv import load_dotenv

TARGET = "actual_load_mw"

FEATURES = [
    "temp_c", "humidity_pct", "precip_mm", "tmax", "tmin", "tavg",
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "load_lag_24", "load_lag_168",
]

FEATURES_V2 = FEATURES + [
    "hour_x_temp", "temp_c_roll_std_72",
    "is_extreme_heat_event", "is_extreme_cold_event", "is_holiday_x_extreme",
]

FEATURES_V3 = FEATURES_V2 + ["temp_change_vs_lag24", "is_high_precip_event"]

PURGE_DAYS = 7

# (train_start, train_end, val_start, val_end) per expanding-window CV fold
FOLD_BOUNDARIES = [
    ("2016-01-08 00:00:00+00:00", "2020-07-30 05:00:00+00:00",
     "2020-08-06 06:00:00+00:00", "2021-08-06 05:00:00+00:00"),
    ("2016-01-08 00:00:00+00:00", "2021-07-30 05:00:00+00:00",
     "2021-08-06 06:00:00+00:00", "2022-08-06 05:00:00+00:00"),
    ("2016-01-08 00:00:00+00:00", "2022-07-30 05:00:00+00:00",
     "2022-08-06 06:00:00+00:00", "2023-08-06 05:00:00+00:00"),
    ("2016-01-08 00:00:00+00:00", "2023-07-30 05:00:00+00:00",
     "2023-08-06 06:00:00+00:00", "2024-08-06 05:00:00+00:00"),
    ("2016-01-08 00:00:00+00:00", "2024-07-30 05:00:00+00:00",
     "2024-08-06 06:00:00+00:00", "2025-08-06 05:00:00+00:00"),
]


def init_mlflow(experiment_name: str = "electricity-load-forecast-production") -> None:
    """Loads .env credentials and points MLflow at the DagsHub tracking server."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # MLFLOW_TRACKING_USERNAME / PASSWORD are read directly from the environment
    # by the mlflow client itself — no need to hold them as local variables.
    mlflow.set_experiment(experiment_name)
