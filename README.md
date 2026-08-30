# Electricity Load Forecaster

Hourly electricity load forecasting for ERCOT South Central, from raw
weather/load ingestion through a served, MLflow-backed model with a live
inference-time ETL pipeline.

## Architecture

```
training (offline, notebook-driven)          live inference (this repo's code path)
─────────────────────────────────────        ─────────────────────────────────────
Notebooks/Preprocessing.ipynb                 src/etl/extract.py
  raw CSV -> timestamp_central,                 Open-Meteo (forecast) + live
  hour/dayofweek/month/is_weekend/               GridStatus lookup for
  is_holiday, load lags                          load_lag_24 / load_lag_168
        |                                              |
src/experiments/features.py                   src/common/timestamps.py
  trend_idx, hour_x_temp,                        THE shared implementation of
  temp_c_roll_std_72,                             every timestamp_central-derived
  temp_change_vs_lag24,                           feature (hour/dayofweek/month/
  holiday_name, extreme-event                     is_weekend/holiday_name/
  flags, holiday_freq            <──shared──>      is_holiday) plus hour_x_temp/
        |                                          temp_c_roll_std_72/
Notebooks/Experiments.ipynb                        temp_change_vs_lag24
  HPO (src/experiments/hpo.py) + CV                    |
  (src/experiments/cv.py), logs the             src/inference/feature_engineering.py
  champion bundle to MLflow                       transform_for_inference(): applies
        |                                          whichever thresholds/holiday_freq_map/
        v                                          trend_idx_origin the CURRENT @champion
  MLflow Model Registry                            bundle was fit with
  (models:/Electricity-Load-Forecaster@champion)         |
        |                                              v
        └──────────────────loaded by──────────> src/api/main.py (FastAPI)
                                                    POST /predict -> model.predict()
                                                    -> logs to Supabase (src/api/db.py)
```

`src/common/timestamps.py` exists because of a real incident: hour/day/month
were once assumed derivable from the UTC `timestamp` column in a test, when
they actually come from `timestamp_central` (America/Chicago local time).
That produced no error — just a silently wrong local hour fed to the model.
The fix is structural: this is the *only* place that logic lives, and both
training (`src/experiments/features.py`) and live inference
(`src/etl/pipeline.py`) call through it.

### Package layout

| Package | Role |
|---|---|
| `src/api/` | FastAPI serving layer. Loads whichever model version has the `@champion` alias at startup, validates requests against a schema built from that model's own feature list, predicts, logs to Supabase. |
| `src/etl/` | Live inference-time extraction: fetches weather + load-lag data for a target UTC hour, assembles the raw feature row, POSTs it to the API. |
| `src/common/` | Shared timestamp/calendar-derivation logic — the single source of truth used by both training and live ETL. |
| `src/experiments/` | Training-time feature engineering, HPO (Optuna), cross-validation, MLflow config. Not installed in the API's Docker image (see `pyproject.toml`'s `experiments` extra). |
| `src/inference/` | The bridge between training and serving: applies a champion bundle's fitted thresholds/holiday_freq_map to new rows identically at train and inference time. |

### `scripts/` — two different kinds of thing

- **One-time historical data ingestion** (bulk/archive APIs, run manually
  when backfilling or refreshing training data):
  `fetch_gridstatus.py`, `fetch_weather.py`, `join_data.py`,
  `validate_raw_data.py` (a Great Expectations gate on the joined raw CSV).
- **Manual live/dev debugging tools** (hit a real running API and/or live
  MLflow — intentionally *not* part of the automated test suite, which
  mocks all network calls): `run_mock_pipeline.py`, `test_preprocessing.py`,
  `test_api.py`.

`scripts/validate_raw_data.py` and `src/etl/validation.py` are not
duplicates: the former gates the raw *historical* training CSV at ingestion
time; the latter validates each *live ETL stage's* output at inference time.

## Running things

Install dependencies with the extras you need (`api` = serving deps,
`experiments` = training/notebook deps, `dev` = pytest):

```bash
uv sync --extra dev --extra api --extra experiments
```

Run the test suite (no network calls — external services are mocked):

```bash
uv run --extra dev --extra api --extra experiments pytest -q
```

Run the API locally:

```bash
uv run --extra api uvicorn src.api.main:app --reload --port 8000
```

Run the live ETL pipeline for one target hour (requires the API running,
plus `MLFLOW_TRACKING_URI`/`GRIDSTATUS_API_KEY`/`DATABASE_URL` in `.env`):

```bash
uv run --extra api --extra experiments python -m src.etl.pipeline \
    --target-timestamp 2026-09-01T18:00:00Z
```

## Data

Raw and interim data are versioned with DVC (`data/raw.dvc`,
`data/interim/df_core_features.parquet.dvc`) — the actual files aren't
committed to git. A few files under `data/interim/` (test/scratch outputs
like `pipeline_test_sample.parquet`) are gitignored and untracked by DVC on
purpose: they're regeneratable test fixtures, not versioned data.

## Roadmap

Beyond the pipeline above, the planned next phases are:

1. **CI** — run the test suite on every push/PR via GitHub Actions.
2. **Scheduled prediction demo** — a GitHub Actions cron job that runs the
   ETL pipeline against a deployed instance of the API on a schedule,
   logging predictions to Supabase automatically.
3. **Automated model promotion** — a `workflow_dispatch`-triggered check
   that compares a newly registered MLflow model version's test metrics
   against the current `@champion` and promotes it (moves the alias) if it
   wins by a defined margin.
4. **Drift monitoring** — a NannyML-based job comparing served predictions
   (already logged to Supabase) against realized actual load, once enough
   prediction history has accumulated.
