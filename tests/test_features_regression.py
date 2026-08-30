"""
test_features_regression.py

src/experiments/features.py's add_base_features() and attach_holiday_names()
were refactored to call src/common/timestamps.py instead of duplicating its
logic. This test proves that refactor didn't change any output: the fixtures
under tests/fixtures/ were captured by running the PRE-refactor
implementation against a deterministic slice of the real
data/interim/df_core_features.parquet; this test reruns the CURRENT
(post-refactor) implementation on the identical slice and asserts the output
is byte-for-byte identical.

If this ever fails, either the refactor changed behavior (bug) or the
fixtures are stale because add_base_features/attach_holiday_names changed on
purpose (regenerate the fixtures — see the capture snippet in the PR/commit
that added this file).
"""
from __future__ import annotations

import pandas as pd

from src.experiments import data, features

DF_CORE_FEATURES_PATH = "data/interim/df_core_features.parquet"


def test_add_base_features_matches_pre_refactor_fixture():
    df = data.load_raw_data(DF_CORE_FEATURES_PATH)
    df = features.add_base_features(df)
    df = features.drop_base_feature_warmup(df, verbose=False)
    sample = df.iloc[::733].reset_index(drop=True)

    before = pd.read_parquet("tests/fixtures/before_add_base_features.parquet")
    pd.testing.assert_frame_equal(sample, before, check_like=False)


def test_attach_holiday_names_matches_pre_refactor_fixture():
    df = data.load_raw_data(DF_CORE_FEATURES_PATH)
    df = features.add_base_features(df)
    df = features.drop_base_feature_warmup(df, verbose=False)

    train, test, _train_end, _test_start = data.chronological_split(df, purge_days=7)
    train_h, test_h = features.attach_holiday_names(train, test)

    train_sample = train_h.iloc[::1009].reset_index(drop=True)
    test_sample = test_h.iloc[::131].reset_index(drop=True) if len(test_h) else test_h

    before_train = pd.read_parquet("tests/fixtures/before_attach_holiday_train.parquet")
    before_test = pd.read_parquet("tests/fixtures/before_attach_holiday_test.parquet")

    for actual, expected in ((train_sample, before_train), (test_sample, before_test)):
        non_holiday_cols = [c for c in actual.columns if c != "holiday_name"]
        pd.testing.assert_frame_equal(actual[non_holiday_cols], expected[non_holiday_cols])
        # NaN (freshly computed) and None (parquet-loaded null) both mean
        # "missing" here but compare unequal via ==  — normalize before comparing.
        actual_names = [None if pd.isna(v) else v for v in actual["holiday_name"]]
        expected_names = [None if pd.isna(v) else v for v in expected["holiday_name"]]
        assert actual_names == expected_names
