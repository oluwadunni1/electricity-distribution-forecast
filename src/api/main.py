"""
main.py

The API's whole job: validate an already-engineered feature row (Pydantic),
run it through the champion pyfunc model, return the prediction. Feature
engineering does NOT happen here — see src/inference/feature_engineering.py
for the upstream preprocessing pipeline that produces the API's input shape.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
import asyncio

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from src.experiments import config
from src.inference import feature_engineering as fe
from src.api import db
from src.api.schema import build_request_model, PredictionResponse, HealthResponse

MODEL_NAME = "Electricity-Load-Forecaster"
MODEL_ALIAS = "champion"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the champion model + builds the request schema once, at startup.

    Note: since the request schema depends on which model is currently champion,
    it can't be a fixed Pydantic type on the /predict decorator (that would only
    ever reflect whatever champion was live at import time). Instead /predict
    takes the raw request body and validates it against app.state.RequestModel,
    which this lifespan block (re)builds from the champion bundle actually
    loaded. Redeploying the API after a model swap picks up any new feature
    set automatically.
    """
    config.init_mlflow()

    artifacts = fe.load_champion_artifacts(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version

    # bundle["features"] (WINNING_FEATURES) only covers the residual model's
    # inputs — trend_idx feeds the separate trend LinearRegression half of the
    # pyfunc wrapper and is never included in that list. engineer_features()
    # always outputs WINNING_FEATURES + ["trend_idx"] together (see
    # src/inference/feature_engineering.py), so the API's schema and the
    # DataFrame it builds for model.predict() both need that same combined list.
    model_input_columns = artifacts["expected_features"] + ["trend_idx"]

    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.model_version = str(version)
    app.state.expected_features = artifacts["expected_features"]
    app.state.model_input_columns = model_input_columns
    app.state.feature_version = artifacts["model_version_tag"]
    app.state.use_holiday_feature = artifacts["use_holiday_feature"]
    app.state.RequestModel = build_request_model(model_input_columns)

    app.state.db_pool = db.create_pool()

    print(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} v{version}  |  input columns: {model_input_columns}")
    yield
    app.state.db_pool.close()


app = FastAPI(title="Electricity Load Forecaster API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=app.state.model_name,
        model_version=app.state.model_version,
        feature_version=app.state.feature_version,
        use_holiday_feature=app.state.use_holiday_feature,
        expected_features=app.state.expected_features,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    payload = await request.json()

    try:
        validated = app.state.RequestModel(**payload)
    except ValidationError as exc:
        # This is the bad-data-injection guardrail: a malformed, out-of-range,
        # wrong-typed, missing, or unexpected field is rejected here — before
        # engineered data ever reaches the model.
        raise HTTPException(status_code=422, detail=exc.errors())

    try:
        row = pd.DataFrame([validated.model_dump()])[app.state.model_input_columns]
        prediction = app.state.model.predict(row)[0]
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller deliberately
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    try:
        # Runs off the event loop deliberately — db.log_prediction is a
        # blocking psycopg call. Without to_thread, a slow or hanging DB
        # connection would stall every other request the server is handling,
        # not just this one (this was the cause of the full-server freeze).
        await asyncio.to_thread(
            db.log_prediction,
            app.state.db_pool,
            target_timestamp=validated.target_timestamp,
            features=row.iloc[0].to_dict(),
            predicted_load_mw=float(prediction),
            model_name=app.state.model_name,
            model_version=app.state.model_version,
        )
    except Exception as exc:  # noqa: BLE001
        # A logging failure shouldn't turn a successful prediction into a 500 —
        # the caller still gets their forecast. Surfacing this as a server-side
        # log line (not silently swallowed) is deliberate: it needs to be
        # noticeable in ops, just not block the response.
        print(f"WARNING: failed to log prediction to DB: {exc}")

    return PredictionResponse(
        target_timestamp=validated.target_timestamp,
        predicted_load_mw=float(prediction),
        model_name=app.state.model_name,
        model_version=app.state.model_version,
    )