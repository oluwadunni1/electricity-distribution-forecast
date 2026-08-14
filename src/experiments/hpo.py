"""
hpo.py — model registry + Optuna tuning, built on top of cv.run_trend_plus_cv
instead of a separately hand-written CV loop (previously cell 36 duplicated
cells 22/24/26's loop almost line for line).
"""
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

from . import cv as cv_mod

MODEL_BUILDERS = {
    "xgb": lambda params: xgb.XGBRegressor(**params, random_state=42, n_jobs=-1),
    "lgbm": lambda params: lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1),
    "catboost": lambda params: cb.CatBoostRegressor(**params, random_state=42, thread_count=-1, verbose=0),
}

PARAM_SPACES = {
    "xgb": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    },
    "lgbm": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    },
    "catboost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
    },
}


def oof_rmse_for_model(train, fold_boundaries, target, features, model_fn, fold_feature_fn):
    """Same signature/behavior as the old cell-36 harness, now just a thin wrapper."""
    result = cv_mod.run_trend_plus_cv(
        train, fold_boundaries, target, features, model_fn,
        fold_feature_fn=fold_feature_fn, verbose=False,
    )
    return cv_mod.compute_oof_metrics(train, result["oof"], target)["rmse"]


def make_objective(model_name, train, fold_boundaries, target, features, fold_feature_fn):
    """Builds an Optuna objective for the given model name using the shared param space + CV runner."""
    param_space_fn = PARAM_SPACES[model_name]
    build_model = MODEL_BUILDERS[model_name]

    def objective(trial):
        params = param_space_fn(trial)
        model_fn = lambda: build_model(params)
        return oof_rmse_for_model(train, fold_boundaries, target, features, model_fn, fold_feature_fn)

    return objective


def run_study(model_name, train, fold_boundaries, target, features, fold_feature_fn, n_trials=40):
    study = optuna.create_study(direction="minimize", study_name=f"{model_name}_v3_tuning")
    objective = make_objective(model_name, train, fold_boundaries, target, features, fold_feature_fn)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
