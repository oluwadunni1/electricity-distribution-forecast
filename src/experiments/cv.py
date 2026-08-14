"""
cv.py — one generic CV runner used by every experiment in the notebook.

Previously cells 18, 20, 22, 24, 26, 28 (v1/v2/v3/LightGBM/CatBoost/KNN) each
hand-wrote the same ~40-line expanding-window loop, and cell 36's HPO harness
was a near-exact copy of the same loop again. `run_trend_plus_cv` below is
that loop, written once, parametrized by feature set / model / optional
fold-feature engineering / optional scaling.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def _fold_masks(train: pd.DataFrame, tr_start, tr_end, val_start, val_end):
    train_mask = (train["timestamp"] >= tr_start) & (train["timestamp"] <= tr_end)
    val_mask = (train["timestamp"] >= val_start) & (train["timestamp"] <= val_end)
    return train_mask, val_mask


def run_naive_baselines(train: pd.DataFrame, fold_boundaries, target: str, verbose: bool = True):
    """Constant-mean and persistence (load_lag_24) baselines — cell 12."""
    oof_mean = np.full(len(train), np.nan)
    oof_persistence = np.full(len(train), np.nan)

    for fold_id, (tr_start, tr_end, val_start, val_end) in enumerate(fold_boundaries, start=1):
        train_mask, val_mask = _fold_masks(train, tr_start, tr_end, val_start, val_end)
        y_train, y_val = train.loc[train_mask, target], train.loc[val_mask, target]

        mean_pred = np.full(val_mask.sum(), y_train.mean())
        oof_mean[val_mask.values] = mean_pred
        oof_persistence[val_mask.values] = train.loc[val_mask, "load_lag_24"].values

        if verbose:
            rmse_mean = mean_squared_error(y_val, mean_pred) ** 0.5
            rmse_pers = mean_squared_error(y_val, oof_persistence[val_mask.values]) ** 0.5
            print(f"Fold {fold_id}  |  Constant RMSE: {rmse_mean:.2f}  |  Persistence RMSE: {rmse_pers:.2f}")

    return {"constant": oof_mean, "persistence": oof_persistence}


def run_raw_cv(train, fold_boundaries, target, features, model_builder, scale_X=False, verbose=True):
    """No trend decomposition — used for the raw-feature LR and XGBoost baselines (cells 14, 16)."""
    oof = np.full(len(train), np.nan)
    fold_rmses = []

    for fold_id, (tr_start, tr_end, val_start, val_end) in enumerate(fold_boundaries, start=1):
        train_mask, val_mask = _fold_masks(train, tr_start, tr_end, val_start, val_end)
        X_train, y_train = train.loc[train_mask, features], train.loc[train_mask, target]
        X_val, y_val = train.loc[val_mask, features], train.loc[val_mask, target]

        if scale_X:
            scaler = StandardScaler()
            X_train, X_val = scaler.fit_transform(X_train), scaler.transform(X_val)

        model = model_builder()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof[val_mask.values] = preds

        if verbose:
            print(f"Fold {fold_id} RMSE: {mean_squared_error(y_val, preds) ** 0.5:.2f}")

    return {"oof": oof, "fold_rmses": fold_rmses}


def run_trend_plus_cv(
    train,
    fold_boundaries,
    target,
    features,
    model_builder,
    fold_feature_fn=None,
    scale_X=False,
    store_fold_details=False,
    verbose=True,
):
    """
    The trend-decomposition CV loop shared by v1/v2/v3, LightGBM, CatBoost,
    KNN, and the HPO objective functions.

    fold_feature_fn: optional callable(fold_train, fold_val) -> (fold_train, fold_val),
        e.g. features.extreme_event_fold_features or a features.compose(...) chain.
        Pass None for v1 (raw features, no derived flags).
    """
    oof = np.full(len(train), np.nan)
    fold_rmses = []
    fold_details = {} if store_fold_details else None

    for fold_id, (tr_start, tr_end, val_start, val_end) in enumerate(fold_boundaries, start=1):
        train_mask, val_mask = _fold_masks(train, tr_start, tr_end, val_start, val_end)
        fold_train = train.loc[train_mask].copy()
        fold_val = train.loc[val_mask].copy()

        if fold_feature_fn is not None:
            fold_train, fold_val = fold_feature_fn(fold_train, fold_val)

        X_train, y_train = fold_train[features], fold_train[target]
        X_val, y_val = fold_val[features], fold_val[target]
        trend_train, trend_val = fold_train[["trend_idx"]], fold_val[["trend_idx"]]

        trend_model = LinearRegression().fit(trend_train, y_train)
        resid_train = y_train - trend_model.predict(trend_train)
        trend_pred_val = trend_model.predict(trend_val)

        if scale_X:
            scaler = StandardScaler()
            X_train, X_val = scaler.fit_transform(X_train), scaler.transform(X_val)

        model = model_builder()
        model.fit(X_train, resid_train)
        resid_pred_val = model.predict(X_val)

        final_pred = trend_pred_val + resid_pred_val
        oof[val_mask.values] = final_pred
        rmse = mean_squared_error(y_val, final_pred) ** 0.5
        fold_rmses.append(rmse)

        if verbose:
            print(f"Fold {fold_id} RMSE: {rmse:.2f}")
        if store_fold_details:
            fold_details[fold_id] = {
                "model": model,
                "y_val": y_val,
                "final_pred": final_pred,
                "residual": y_val.values - final_pred,
                "timestamps": fold_val["timestamp_central"].values,
            }

    return {"oof": oof, "fold_rmses": fold_rmses, "fold_details": fold_details}


def compute_oof_metrics(train: pd.DataFrame, oof: np.ndarray, target: str) -> dict:
    """RMSE + MAPE over the valid (non-NaN) OOF rows — used after every CV run."""
    valid = ~np.isnan(oof)
    y_true, y_pred = train.loc[valid, target], oof[valid]
    return {
        "valid_mask": valid,
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def compute_metrics(y_true, y_pred, peak_pct: float = 0.05) -> dict:
    """RMSE / MAPE / peak-hours MAPE for the model comparison table (cell 30)."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    peak_cutoff = np.quantile(y_true, 1 - peak_pct)
    peak_mask = y_true >= peak_cutoff
    peak_mape = mean_absolute_percentage_error(
        np.asarray(y_true)[peak_mask], np.asarray(y_pred)[peak_mask]
    ) * 100
    return {"rmse": rmse, "mape": mape, "peak_mape": peak_mape}


def hill_climb_ensemble(oof_dict, y_true_full, valid_mask, n_iterations=50, tol=1e-6):
    """Unchanged from cell 34 — already a clean, generic function."""
    preds = {name: arr[valid_mask] for name, arr in oof_dict.items()}
    y = y_true_full[valid_mask]

    solo_rmse = {name: mean_squared_error(y, p) ** 0.5 for name, p in preds.items()}
    best_start = min(solo_rmse, key=solo_rmse.get)

    selected = [best_start]
    ensemble_preds = preds[best_start].copy()
    history = [solo_rmse[best_start]]

    for _ in range(n_iterations):
        best_rmse, best_name, best_preds = np.inf, None, None
        for name, p in preds.items():
            candidate = (ensemble_preds * len(selected) + p) / (len(selected) + 1)
            rmse = mean_squared_error(y, candidate) ** 0.5
            if rmse < best_rmse:
                best_rmse, best_name, best_preds = rmse, name, candidate
        if best_rmse < history[-1] - tol:
            selected.append(best_name)
            ensemble_preds = best_preds
            history.append(best_rmse)
        else:
            break

    weights = pd.Series(selected).value_counts(normalize=True).sort_values(ascending=False)
    return ensemble_preds, selected, weights, history, solo_rmse
