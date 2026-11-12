from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_check import (
    ENVELOPE_PERCENTILES,
    SENSOR_COLUMNS,
    WEEK_HOURS,
    ReferenceProfile,
    build_feature_baselines,
    check_batch_for_drift,
)


def _raw_train_df(n_rows: int = 3000) -> pd.DataFrame:
    """Hourly readings with a slow seasonal cycle plus noise, so the rolling
    weekly means span a realistic band instead of collapsing to one value."""
    rng = np.random.default_rng(42)
    t = np.arange(n_rows)
    season = 5.0 * np.sin(2 * np.pi * t / 1000)
    data = {col: 10.0 + season + rng.normal(scale=2.0, size=n_rows) for col in SENSOR_COLUMNS}
    return pd.DataFrame(data)


def test_build_feature_baselines_covers_every_sensor_column() -> None:
    profile = build_feature_baselines(_raw_train_df())

    assert set(profile.features.keys()) == set(SENSOR_COLUMNS)
    assert profile.n_rows == 3000
    assert profile.baseline_test_mse is None
    assert profile.baseline_test_mae is None


def test_build_feature_baselines_matches_numpy_mean_and_std() -> None:
    df = _raw_train_df()
    profile = build_feature_baselines(df)

    baseline = profile.features["OT"]
    assert baseline.mean == pytest.approx(float(df["OT"].mean()), abs=1e-9)
    assert baseline.std == pytest.approx(float(df["OT"].std(ddof=0)), abs=1e-9)


def test_envelope_is_the_p1_p99_band_of_rolling_weekly_means() -> None:
    df = _raw_train_df()
    profile = build_feature_baselines(df)

    weekly_means = df["OT"].rolling(WEEK_HOURS).mean().dropna()
    low, high = np.percentile(weekly_means, ENVELOPE_PERCENTILES)
    assert profile.features["OT"].weekly_mean_low == pytest.approx(low)
    assert profile.features["OT"].weekly_mean_high == pytest.approx(high)
    assert low < high


def test_series_shorter_than_a_week_still_gets_an_envelope() -> None:
    profile = build_feature_baselines(_raw_train_df(n_rows=50))

    baseline = profile.features["OT"]
    # Window = the whole series: the band collapses to its mean, not to NaN.
    assert baseline.weekly_mean_low == pytest.approx(baseline.mean)
    assert baseline.weekly_mean_high == pytest.approx(baseline.mean)


def test_reference_profile_round_trips_through_json(tmp_path) -> None:
    profile = build_feature_baselines(_raw_train_df())
    profile.baseline_test_mse = 1.23
    profile.baseline_test_mae = 0.98

    path = profile.write(tmp_path / "reference_profile.json")
    restored = ReferenceProfile.read(path)

    assert restored.n_rows == profile.n_rows
    assert restored.baseline_test_mse == pytest.approx(1.23)
    assert restored.baseline_test_mae == pytest.approx(0.98)
    for column in SENSOR_COLUMNS:
        original, loaded = profile.features[column], restored.features[column]
        assert loaded.mean == pytest.approx(original.mean)
        assert loaded.weekly_mean_low == pytest.approx(original.weekly_mean_low)
        assert loaded.weekly_mean_high == pytest.approx(original.weekly_mean_high)


def test_a_week_from_the_training_period_is_not_drift() -> None:
    df = _raw_train_df()
    reference = build_feature_baselines(df)
    one_week = df.iloc[1000 : 1000 + WEEK_HOURS]

    report = check_batch_for_drift(one_week, reference)

    assert report.status == "ok"
    assert report.drifted_features == []


def test_a_level_shift_outside_the_training_envelope_is_drift() -> None:
    df = _raw_train_df()
    reference = build_feature_baselines(df)
    shifted_week = df.iloc[1000 : 1000 + WEEK_HOURS].copy()
    shifted_week["OT"] = shifted_week["OT"] - 20.0  # colder than any training week

    report = check_batch_for_drift(shifted_week, reference)

    assert report.status == "drift"
    assert report.drifted_features == ["OT"]
    result = report.per_feature["OT"]
    assert result.drifted is True
    assert result.batch_mean < result.reference_low


def test_check_batch_for_drift_ignores_columns_missing_from_the_batch() -> None:
    df = _raw_train_df()
    reference = build_feature_baselines(df)
    partial_batch = df.iloc[:WEEK_HOURS].drop(columns=["OT"])

    report = check_batch_for_drift(partial_batch, reference)

    assert "OT" not in report.per_feature
