"""
SHAP-guided feature engineering, mirroring the source paper's per-model
feature sets (Sections 6.1-6.3 / Table 1).

Each model gets its OWN builder because the paper's engineered features
are not shared verbatim across models — the window length and
normalization of `load_spike_vs_mean` / `temp_spike_vs_mean` differ
between XGBoost and LightGBM even though the feature names match.

All builders take `df_feat` — the output of
`feature_engineering.add_baseline_feature_set` (which must still carry the
raw `timestamp`, `actual_load_mw`, `tavg`, `tmin`, `tmax`, `prcp` columns
alongside the baseline calendar/lag features) — and return a new dataframe:
`timestamp`, `actual_load_mw`, plus that model's feature columns, with
rows containing NaNs from rolling/lag windows dropped.

Caveat carried over from the paper: `tavg`/`tmax`/`tmin` are used as
same-day observed values, not forecasts. That's fine for evaluating
explainability-driven feature engineering (the paper's stated goal), but
if you point this at a live forecasting pipeline you'd need to swap
these for weather forecasts, or the "future" weather at inference time
won't exist yet.
"""
import numpy as np
import pandas as pd

LOAD_COL = "actual_load_mw"


def _rolling_load_stats(df: pd.DataFrame, windows=(24, 168, 336)) -> pd.DataFrame:
    """Rolling mean/std/max of load, shifted 1h so the current row never
    sees its own value (matches the paper's stated leakage-avoidance)."""
    out = {}
    shifted = df[LOAD_COL].shift(1)
    for w in windows:
        out[f"load_roll_mean_{w}"] = shifted.rolling(w).mean()
        out[f"load_roll_std_{w}"] = shifted.rolling(w).std()
        out[f"load_roll_max_{w}"] = shifted.rolling(w).max()
    return pd.DataFrame(out, index=df.index)


def _cdd_hdd(tavg: pd.Series, baseline: float = 65.0):
    cdd = (tavg - baseline).clip(lower=0)
    hdd = (baseline - tavg).clip(lower=0)
    return cdd, hdd


def _cyclical(values: pd.Series, period: float):
    radians = 2 * np.pi * values.astype(float) / period
    return np.sin(radians), np.cos(radians)


def build_lr_engineered_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Section 6.1: CDD/HDD lags, temp-spike ratio, extreme heat/cold
    flags, cyclical hour/dayofweek encodings, two interaction terms."""
    df = df_feat.copy()
    df = pd.concat([df, _rolling_load_stats(df, windows=(24, 168))], axis=1)

    cdd, hdd = _cdd_hdd(df["tavg"])
    df["CDD"], df["HDD"] = cdd, hdd
    df["CDD_lag_24"] = df["CDD"].shift(24)
    df["HDD_lag_24"] = df["HDD"].shift(24)

    df["temp_spike_vs_mean"] = (df["tmax"] - df["tavg"]) / (df["tavg"] + 1)

    heat_thresh = df["tmax"].quantile(0.95)
    cold_thresh = df["tmin"].quantile(0.05)
    df["is_extreme_heat_event"] = (df["tmax"] > heat_thresh).astype(int)
    df["is_extreme_cold_event"] = (df["tmin"] < cold_thresh).astype(int)

    df["hour_sin"], _ = _cyclical(df["hour"], 24)
    _, df["dayofweek_cos"] = _cyclical(df["day_of_week"], 7)

    df["lag_24_x_hour"] = df["load_lag_24"] * df["hour"]
    df["CDD_x_hour"] = df["CDD"] * df["hour"]

    feature_cols = [
        "hour", "day_of_week", "month", "tavg", "tmin", "tmax", "prcp",
        "load_lag_24", "load_lag_168",
        "CDD_lag_24", "HDD_lag_24", "temp_spike_vs_mean",
        "is_extreme_cold_event", "is_extreme_heat_event",
        "hour_sin", "dayofweek_cos", "lag_24_x_hour", "CDD_x_hour",
    ]
    return (
        df[["timestamp", LOAD_COL] + feature_cols]
        .dropna()
        .reset_index(drop=True)
    )


def build_xgb_engineered_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Section 6.2: 24h-window load-spike ratio, 72h rolling temp max,
    Monday/heat-event flags, two interaction terms."""
    df = df_feat.copy()
    df = pd.concat([df, _rolling_load_stats(df, windows=(24,))], axis=1)

    df["load_spike_vs_mean"] = (
        (df[LOAD_COL] - df["load_roll_mean_24"]) / (df["load_roll_mean_24"] + 1)
    )
    df["temp_spike_vs_mean"] = (df["tmax"] - df["tavg"]) / (df["tavg"] + 1)
    df["tmax_roll_max_72"] = df["tmax"].shift(1).rolling(72).max()

    cdd, _ = _cdd_hdd(df["tavg"])
    df["CDD"] = cdd
    df["CDD_x_hour"] = df["CDD"] * df["hour"]
    df["lag_24_x_hour"] = df["load_lag_24"] * df["hour"]

    heat_thresh = df["tmax"].quantile(0.95)
    df["is_extreme_heat_event"] = (df["tmax"] > heat_thresh).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)

    feature_cols = [
        "hour", "day_of_week", "month", "tavg", "tmin", "tmax", "prcp",
        "load_lag_24", "load_lag_168",
        "load_spike_vs_mean", "temp_spike_vs_mean", "tmax_roll_max_72",
        "CDD_x_hour", "lag_24_x_hour", "is_extreme_heat_event", "is_monday",
    ]
    return (
        df[["timestamp", LOAD_COL] + feature_cols]
        .dropna()
        .reset_index(drop=True)
    )


def build_lgbm_engineered_features(df_feat: pd.DataFrame, eps: float = 1.0) -> pd.DataFrame:
    """LightGBM section — NOTE the formulas differ from XGBoost's
    same-named features: 168h window (not 24h) for the spike ratio,
    normalized by rolling std (not mean), and an unnormalized temp
    spike (tmax - tavg, no division)."""
    df = df_feat.copy()
    df = pd.concat([df, _rolling_load_stats(df, windows=(168,))], axis=1)

    df["load_spike_vs_mean"] = (
        (df[LOAD_COL] - df["load_roll_mean_168"]) / (df["load_roll_std_168"] + eps)
    )
    df["temp_spike_vs_mean"] = df["tmax"] - df["tavg"]
    df["lag_24_x_hour"] = df["load_lag_24"] * df["hour"]

    heat_thresh = df["tmax"].quantile(0.95)
    df["is_extreme_heat_event"] = (df["tmax"] > heat_thresh).astype(int)

    feature_cols = [
        "hour", "day_of_week", "month", "tavg", "tmin", "tmax", "prcp",
        "load_lag_24", "load_lag_168",
        "load_spike_vs_mean", "load_roll_mean_168", "temp_spike_vs_mean",
        "lag_24_x_hour", "is_extreme_heat_event",
    ]
    return (
        df[["timestamp", LOAD_COL] + feature_cols]
        .dropna()
        .reset_index(drop=True)
    )
