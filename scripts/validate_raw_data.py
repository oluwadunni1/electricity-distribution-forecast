"""
validate_raw_data.py
--------------------
Great Expectations (V1 API) silent validation gate.

Sits immediately before the feature-engineering pipeline and enforces:
  1. Column count  – exactly 5 columns.
  2. Column names  – {timestamp, actual_load_mw, temp_c, humidity_pct, precip_mm}.
  3. Row count     – tight window around the expected 92,688 hourly rows
                     (2016-01-01 00:00 UTC → 2026-07-28 23:00 UTC).
  4. Null checks   – zero nulls in every column.
  5. Value ranges  – realistic physical bounds per numeric column.
  6. Timestamp fmt – strict ISO-8601 UTC format (%Y-%m-%d %H:%M:%S+00:00).

Public API
----------
validate_dataframe(df) → bool
    Accept a pandas DataFrame already in memory (e.g. from a notebook).
    Renames columns to canonical names, runs the full GX suite, logs results.
    Use this in experiments.ipynb.

run_validation_gate(data_path) → bool
    Load a CSV from disk and delegate to validate_dataframe.
    Used by the CLI / Azure pipeline step.

Both functions return True (pass) or False (fail) and never raise.

CLI usage:
    python scripts/validate_raw_data.py --data_path data/raw/ercot_south_central_raw.csv

Exit codes:
    0  – all expectations passed; pipeline may continue.
    1  – one or more expectations failed or an unexpected error occurred;
         pipeline is halted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical column names matching the joined raw dataset produced by
# join_data.py:  timestamp | actual_load_mw | temp_c | humidity_pct | precip_mm
COLUMNS: list[str] = [
    "timestamp",
    "actual_load_mw",
    "temp_c",
    "humidity_pct",
    "precip_mm",
]

# Row-count boundaries.
# Exact row count from ercot_south_central_raw.csv:
#   92,689 total lines − 1 header = 92,688 data rows
#   (2016-01-01 00:00 UTC → 2026-07-28 23:00 UTC, hourly cadence).
# ±3% window catches partial API pulls while tolerating minor date-range drift
# (e.g. a pipeline re-run that extends the end date by a few hours/days).
EXPECTED_ROWS: int = 92_688
ROW_COUNT_MIN: int = int(EXPECTED_ROWS * 0.97)   # ≈ 89,907
ROW_COUNT_MAX: int = int(EXPECTED_ROWS * 1.03)   # ≈ 95,469

# Physical / operational bounds for each numeric column.
#
# actual_load_mw:
#   ERCOT South-Central zone electricity demand.
#   Observed winter baseline ≈ 5,000 MW; summer peak on record ≈ 25,000 MW.
#   Wide safety margins applied to tolerate multi-year historical extremes.
#   min=500  → catches near-zero values that indicate a data outage/gap.
#   max=30_000 → comfortably above any historical ERCOT SC peak.
LOAD_MW_MIN: float = 500.0
LOAD_MW_MAX: float = 30_000.0

# temp_c:
#   South-Central Texas (Austin / San Antonio area).
#   Historic low ≈ -18 °C (Feb 2021 winter storm);
#   historic high ≈ 44 °C (summer 2023).
#   Extra margin applied on both ends.
TEMP_MIN: float = -20.0
TEMP_MAX: float = 50.0

# humidity_pct: relative humidity, physically bounded 0–100 %.
HUMIDITY_MIN: float = 0.0
HUMIDITY_MAX: float = 100.0

# precip_mm: precipitation in mm, always non-negative.
PRECIP_MIN: float = 0.0

# Strict UTC timestamp regex: "YYYY-MM-DD HH:MM:SS+00:00"
TIMESTAMP_REGEX: str = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+00:00$"

# GX object identifiers (used throughout the ephemeral context)
SUITE_NAME: str = "raw_data_validation_suite"
DATASOURCE_NAME: str = "raw_csv_datasource"
DATA_ASSET_NAME: str = "raw_csv_asset"
BATCH_DEFINITION_NAME: str = "raw_csv_batch"
VALIDATION_DEFINITION_NAME: str = "raw_data_validation_def"
CHECKPOINT_NAME: str = "raw_data_checkpoint"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Great Expectations silent validation gate for raw ML data."
    )
    p.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/raw/ercot_south_central_raw.csv"),
        help="Path to the raw joined CSV file (default: data/raw/ercot_south_central_raw.csv).",
    )
    return p.parse_args()


# ── Suite builder ─────────────────────────────────────────────────────────────

def build_expectation_suite(context: gx.DataContext) -> gx.ExpectationSuite:
    """
    Construct the ExpectationSuite using the GX V1 gx.expectations API.

    All six validation requirements are encoded here as discrete Expectation
    objects so that every failure is independently identifiable in the log.
    """
    suite = gx.ExpectationSuite(name=SUITE_NAME)

    # ── 1. Column count ───────────────────────────────────────────────────────
    suite.add_expectation(
        gxe.ExpectTableColumnCountToEqual(value=len(COLUMNS))
    )

    # ── 2. Column names (exact set, order-insensitive) ────────────────────────
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(
            column_set=COLUMNS,
            exact_match=True,
        )
    )

    # ── 3. Row count range ────────────────────────────────────────────────────
    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=ROW_COUNT_MIN,
            max_value=ROW_COUNT_MAX,
        )
    )

    # ── 4. Null / missing value checks (all five columns) ────────────────────
    for col in COLUMNS:
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(column=col)
        )

    # ── 5. Value ranges for the four numeric columns ───────────────────────────

    # ERCOT South-Central electricity demand (MW)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="actual_load_mw",
            min_value=LOAD_MW_MIN,
            max_value=LOAD_MW_MAX,
        )
    )

    # Population-weighted average temperature for the zone (°C)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="temp_c",
            min_value=TEMP_MIN,
            max_value=TEMP_MAX,
        )
    )

    # Relative humidity (%)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="humidity_pct",
            min_value=HUMIDITY_MIN,
            max_value=HUMIDITY_MAX,
        )
    )

    # Precipitation (mm) — always non-negative, no physical upper bound
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="precip_mm",
            min_value=PRECIP_MIN,
            max_value=None,   # open upper bound
        )
    )

    # ── 6. Timestamp string format ────────────────────────────────────────────
    # The timestamp column is kept as a raw string (dtype=str during CSV load)
    # so this regex runs character-for-character on the literal cell content,
    # catching timezone drift (e.g. +05:30), truncated seconds, or any other
    # malformed entries that slipped through the upstream fetch scripts.
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(
            column="timestamp",
            regex=TIMESTAMP_REGEX,
        )
    )

    context.suites.add_or_update(suite)
    log.info(
        "ExpectationSuite '%s' registered (%d expectations).",
        SUITE_NAME,
        len(suite.expectations),
    )
    return suite


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical names if and only if the count
    matches exactly.  If the count differs, return the DataFrame unchanged
    so Great Expectations can surface a structured column-count failure
    rather than a raw Python ValueError.
    """
    n_actual = df.shape[1]
    n_expected = len(COLUMNS)

    if n_actual == n_expected:
        df = df.copy()
        df.columns = COLUMNS
        log.info("Columns renamed to canonical names: %s", COLUMNS)
    else:
        log.warning(
            "Column count mismatch (got %d, expected %d) — skipping rename. "
            "GX will report this as a structured failure.",
            n_actual,
            n_expected,
        )
    return df


def _count_expectations(result: gx.checkpoint.CheckpointResult) -> int:
    """Return total number of evaluated expectations across all validation results."""
    return sum(len(vr.results) for vr in result.run_results.values())


def _log_failures(result: gx.checkpoint.CheckpointResult) -> None:
    """
    Parse CheckpointResult and emit one structured log line per failed
    expectation.  Azure pipeline logs are self-contained — no DataDocs
    server is required to diagnose a failure.
    """
    for val_result in result.run_results.values():
        for er in val_result.results:
            if not er.success:
                exp_type = type(er.expectation_config).__name__
                kwargs = er.expectation_config.kwargs
                observed = er.result.get("observed_value", "N/A")
                unexpected_pct = er.result.get("unexpected_percent")

                msg = (
                    f"  FAILED  {exp_type:<48} "
                    f"kwargs={kwargs}  |  observed={observed}"
                )
                if unexpected_pct is not None:
                    msg += f"  |  unexpected_pct={unexpected_pct:.2f}%"
                log.error(msg)


# ── Core GX runner (shared by both public entry points) ──────────────────────

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Run the full GX validation suite against a pandas DataFrame
    **already in memory**.

    This is the primary entry point for notebooks and interactive scripts
    where data has already been loaded.  ``run_validation_gate`` (CLI / Azure)
    delegates to this function after loading the CSV from disk.

    The timestamp column must still be a raw string (``dtype=object / str``)
    for the regex expectation to evaluate the literal characters.  If you
    loaded the CSV with ``parse_dates=True``, cast it back first:

        df["timestamp"] = df["timestamp"].astype(str)

    Parameters
    ----------
    df : pd.DataFrame
        The raw joined DataFrame.  Column names are normalised internally
        to the canonical set before validation begins.

    Returns
    -------
    bool
        True  – every expectation passed.
        False – any check failed or an unexpected error occurred.
    """
    log.info("=" * 64)
    log.info("GX Validation Gate  |  in-memory DataFrame  (%d rows × %d cols)", *df.shape)
    log.info("=" * 64)

    try:
        log.info("Source column names: %s", list(df.columns))

        # ── Rename columns to canonical names (graceful on mismatch) ──────────
        df = _safe_rename_columns(df)

        # ── Ephemeral in-memory GX context ────────────────────────────────────
        context: gx.DataContext = gx.get_context(mode="ephemeral")

        # ── Wire pandas in-memory datasource ──────────────────────────────────
        datasource = context.data_sources.add_pandas(name=DATASOURCE_NAME)
        data_asset = datasource.add_dataframe_asset(name=DATA_ASSET_NAME)
        batch_def = data_asset.add_batch_definition_whole_dataframe(
            BATCH_DEFINITION_NAME
        )

        # ── Build and register expectation suite ──────────────────────────────
        suite = build_expectation_suite(context)

        # ── Validation definition: one batch ↔ one suite ──────────────────────
        val_def = gx.ValidationDefinition(
            name=VALIDATION_DEFINITION_NAME,
            data=batch_def,
            suite=suite,
        )
        context.validation_definitions.add(val_def)

        # ── Checkpoint: silent local gate, zero notification actions ──────────
        checkpoint = gx.Checkpoint(
            name=CHECKPOINT_NAME,
            validation_definitions=[val_def],
            actions=[],
        )
        context.checkpoints.add(checkpoint)

        # ── Execute ───────────────────────────────────────────────────────────
        log.info("Running checkpoint …")
        result: gx.checkpoint.CheckpointResult = checkpoint.run(
            batch_parameters={"dataframe": df},
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        passed: bool = result.success
        total = _count_expectations(result)

        log.info("=" * 64)
        if passed:
            log.info("  VALIDATION PASSED — all %d expectations met.", total)
        else:
            log.error(
                "  VALIDATION FAILED — %d expectation(s) not met:", total
            )
            _log_failures(result)
        log.info("=" * 64)
        return passed

    except Exception:
        log.error("=" * 64)
        log.error("  UNEXPECTED ERROR in validation gate.")
        log.error("Full traceback:\n%s", traceback.format_exc())
        log.error("=" * 64)
        return False


# ── Gate function (public API consumed by the Azure pipeline / CLI) ───────────

def run_validation_gate(data_path: Path) -> bool:
    """
    Load a CSV from *data_path* and delegate to :func:`validate_dataframe`.

    Every failure mode (missing file, unreadable CSV, failed expectations, or
    any unexpected exception) is caught and returned as ``False``.

    Parameters
    ----------
    data_path : Path
        Absolute or relative path to the raw joined CSV file.

    Returns
    -------
    bool
        True  – every expectation passed  → proceed to feature engineering.
        False – any check failed or an error occurred → halt the pipeline.
    """
    log.info("=" * 64)
    log.info("GX Validation Gate  |  file: %s", data_path.resolve())
    log.info("=" * 64)

    try:
        if not data_path.exists():
            log.error("Data file not found: %s", data_path.resolve())
            return False

        # Keep column 0 (timestamp) as a raw string so the regex expectation
        # evaluates the literal character sequence, not a parsed datetime.
        log.info("Loading CSV …")
        df = pd.read_csv(data_path, header=0, dtype={0: str})
        log.info("Loaded %d rows × %d columns.", *df.shape)

        return validate_dataframe(df)

    except Exception:
        log.error("=" * 64)
        log.error("  UNEXPECTED ERROR loading data — pipeline is HALTED.")
        log.error("Full traceback:\n%s", traceback.format_exc())
        log.error("=" * 64)
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    passed = run_validation_gate(data_path=args.data_path)
    # Exit 0 = success  → Azure pipeline step proceeds.
    # Exit 1 = failure  → Azure pipeline step fails and halts the run.
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
