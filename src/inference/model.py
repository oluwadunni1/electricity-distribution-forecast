"""
model.py — the MLflow pyfunc wrapper logged as the production model.

Previously this class only existed inline in Notebooks/Experiments.ipynb
(the training notebook), which meant nothing outside that notebook could
import it — blocking any future automated retraining or promotion script
from reusing the exact same prediction logic. Moved here so the notebook
and any future tooling both `from src.inference.model import
ElectricityForecaster` instead of maintaining two copies.

Implements the "trend + residual" architecture used throughout this repo:
a LinearRegression fit on trend_idx captures the long-run trend, and a
separate boosted model (xgb/lgbm/catboost, chosen at training time) is fit
on the RESIDUAL of that trend — predict() sums both halves back together.
See src/experiments/cv.py::run_trend_plus_cv for the training-side
counterpart of this decomposition.
"""
from __future__ import annotations

import joblib
import mlflow.pyfunc
import pandas as pd


class ElectricityForecaster(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["model_bundle"])
        self.trend_model = self.bundle["trend_model"]
        self.residual_model = self.bundle["residual_model"]
        self.features = self.bundle["features"]

    def predict(self, context, model_input):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        trend_preds = self.trend_model.predict(df[["trend_idx"]])
        residual_preds = self.residual_model.predict(df[self.features])
        return trend_preds + residual_preds
