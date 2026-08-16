"""
schema.py

The API only validates + predicts (feature engineering already happened
upstream, in the preprocessing pipeline — see src/inference/feature_engineering.py).
So the request schema must match the CURRENT champion's expected feature list —
which varies depending on `use_holiday_feature`.

Rather than hand-maintaining two schemas (with/without holiday_freq) and risking
them drifting from the model, build_request_model() constructs the Pydantic
model at API startup directly from the champion bundle's `features` list. When
the champion swaps to a model with a different feature set, restarting the API
(which happens automatically on redeploy) rebuilds the schema to match — no
manual schema edit required.
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field, create_model

# One entry per feature this model could ever see across FEATURES_V3 + holiday_freq.
# (type, Field(...)) pairs, consumed by pydantic.create_model. Bounds here are the
# bad-data guardrail — an out-of-range or wrong-type value gets rejected at the
# door instead of silently reaching the model.
FEATURE_FIELD_SPECS: dict = {
    "temp_c": (float, Field(..., description="Current temperature, Celsius")),
    "humidity_pct": (float, Field(..., ge=0, le=100)),
    "precip_mm": (float, Field(..., ge=0)),
    "tmax": (float, Field(...)),
    "tmin": (float, Field(...)),
    "tavg": (float, Field(...)),
    "hour": (int, Field(..., ge=0, le=23)),
    "dayofweek": (int, Field(..., ge=0, le=6)),
    "month": (int, Field(..., ge=1, le=12)),
    "is_weekend": (int, Field(..., ge=0, le=1)),
    "is_holiday": (int, Field(..., ge=0, le=1)),
    "load_lag_24": (float, Field(..., ge=0)),
    "load_lag_168": (float, Field(..., ge=0)),
    "hour_x_temp": (float, Field(...)),
    "temp_c_roll_std_72": (float, Field(..., ge=0)),
    "is_extreme_heat_event": (int, Field(..., ge=0, le=1)),
    "is_extreme_cold_event": (int, Field(..., ge=0, le=1)),
    "is_holiday_x_extreme": (int, Field(..., ge=0, le=1)),
    "temp_change_vs_lag24": (float, Field(...)),
    "is_high_precip_event": (int, Field(..., ge=0, le=1)),
    "holiday_freq": (float, Field(..., ge=0, le=1)),
    "trend_idx": (float, Field(...)),
}


def build_request_model(expected_features: list[str]) -> type[BaseModel]:
    """
    Builds a PredictionRequest model containing exactly the champion's expected
    features — no more, no less — plus `target_timestamp`, which every request
    must supply regardless of which features the current champion needs.

    target_timestamp identifies WHICH HOUR is being forecasted (not when the
    API call happens — that's predicted_at in the response). This is the join
    key the DB logging and later ground-truth backfill job rely on, and it
    can't be reliably reconstructed from the engineered features alone
    (hour/dayofweek/month are decomposed, trend_idx doesn't cleanly invert),
    so the caller — which already knows what hour it built features for —
    supplies it explicitly. Pydantic's `extra="forbid"` means a payload with
    any other unexpected field (e.g. holiday_freq sent to a model that
    doesn't use it) is rejected, not silently ignored.
    """
    unknown = set(expected_features) - set(FEATURE_FIELD_SPECS)
    if unknown:
        raise ValueError(
            f"expected_features contains fields with no known schema: {unknown}. "
            f"Add them to FEATURE_FIELD_SPECS in schema.py."
        )

    fields = {name: FEATURE_FIELD_SPECS[name] for name in expected_features}
    fields["target_timestamp"] = (
        datetime,
        Field(..., description="UTC timestamp of the hour being forecasted"),
    )
    return create_model(
        "PredictionRequest",
        __config__={"extra": "forbid"},
        **fields,
    )


class PredictionResponse(BaseModel):
    target_timestamp: datetime
    predicted_load_mw: float
    model_name: str
    model_version: str
    model_alias: str = "champion"
    predicted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    feature_version: str
    use_holiday_feature: bool
    expected_features: list[str]