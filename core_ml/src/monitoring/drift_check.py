"""Simple data drift check for batch inference.

It loads the reference profile logged with the model and answers one
question per sensor: is the average level of this batch inside the range of
weekly averages the model saw during training? A batch outside that range
means the model is being asked to extrapolate to conditions it has never
seen, which is the failure mode that matters most for a forecaster.

Why weekly averages and not the raw training distribution: the readings are
strongly seasonal and autocorrelated. A batch covers days, the training set
covers seasons, so any distribution-level test against the whole training
set (PSI, KS, a z-score against the per-reading spread) is either always
alerting or never alerting. Rolling weekly means from the training period
give the reference the same time scale as a batch, so the p1-p99 envelope
is a calibrated "normal" band.

It intentionally does NOT retrain, alert an external system, or store any
state between runs; it only answers "does this batch still look like
the data the model was trained on?" and reports the answer. Deciding what
to do about a drift report (investigate, retrain, ignore) is left to a
human. For a project at this scale, that is enough: no message queue, no
extra database table, no cooldown/ceiling bookkeeping to maintain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone  # not datetime.UTC: 3.11+ only, runtime images are 3.10
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

SENSOR_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

# One week of hourly readings: the time scale of a batch (the daily CronJob
# scores roughly a week of history) and a full weekly load cycle.
WEEK_HOURS = 168

# The "normal" band is the 1st-99th percentile of the rolling weekly means,
# not their min-max: a single anomalous week in the training data should not
# widen the band for good.
ENVELOPE_PERCENTILES = (1.0, 99.0)


@dataclass
class FeatureBaseline:
    mean: float
    std: float
    # Envelope of the rolling weekly means over the training period.
    weekly_mean_low: float
    weekly_mean_high: float


@dataclass
class ReferenceProfile:
    """The statistical baseline logged alongside the model.

    Built in two steps because they happen in two different pipeline
    stages: data_processing.py computes the per-sensor statistics from the
    raw training split (`build_feature_baselines`) before any model exists;
    train.py later fills in the held-out test MSE/MAE once it has
    evaluated one, and logs the completed profile as an MLflow artifact of
    that run.
    """

    created_at: str
    n_rows: int
    features: dict[str, FeatureBaseline] = field(default_factory=dict)
    baseline_test_mse: float | None = None
    baseline_test_mae: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceProfile:
        return cls(
            created_at=payload["created_at"],
            n_rows=int(payload["n_rows"]),
            features={k: FeatureBaseline(**v) for k, v in payload.get("features", {}).items()},
            baseline_test_mse=payload.get("baseline_test_mse"),
            baseline_test_mae=payload.get("baseline_test_mae"),
        )

    @classmethod
    def read(cls, path: str | Path) -> ReferenceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_feature_baselines(raw_train_df: pd.DataFrame) -> ReferenceProfile:
    """Per-sensor statistics of the raw (pre-scaling), chronologically
    ordered training split.

    Called from data_processing.py, which has the raw training dataframe
    but has not trained anything yet; the performance fields are left
    None here and filled in later by train.py.
    """
    features: dict[str, FeatureBaseline] = {}
    for column in SENSOR_COLUMNS:
        series = raw_train_df[column].dropna()
        # Shorter-than-a-week inputs (smoke-test data) fall back to a window
        # of the whole series instead of producing an empty envelope.
        window = min(WEEK_HOURS, len(series))
        weekly_means = series.rolling(window).mean().dropna().to_numpy(dtype=float)
        low, high = np.percentile(weekly_means, ENVELOPE_PERCENTILES)
        features[column] = FeatureBaseline(
            mean=float(series.mean()),
            std=float(series.std(ddof=0)),
            weekly_mean_low=float(low),
            weekly_mean_high=float(high),
        )

    return ReferenceProfile(
        created_at=datetime.now(timezone.utc).isoformat(),
        n_rows=int(len(raw_train_df)),
        features=features,
    )


@dataclass
class FeatureDriftResult:
    feature: str
    batch_mean: float
    reference_low: float
    reference_high: float
    drifted: bool


@dataclass
class DriftReport:
    status: str  # "ok" | "drift"
    checked_at: str
    n_rows: int
    drifted_features: list[str] = field(default_factory=list)
    per_feature: dict[str, FeatureDriftResult] = field(default_factory=dict)


def check_batch_for_drift(batch_df: pd.DataFrame, reference: ReferenceProfile) -> DriftReport:
    """Compares one batch's per-feature mean against the training envelope.

    A feature "drifts" when the batch mean falls outside the p1-p99 band of
    the training period's rolling weekly means. One number per feature, no
    state carried between batches, no action taken automatically.
    """
    per_feature: dict[str, FeatureDriftResult] = {}
    drifted: list[str] = []

    for name, baseline in reference.features.items():
        if name not in batch_df.columns:
            continue
        batch_mean = float(batch_df[name].dropna().mean())
        is_drifted = not baseline.weekly_mean_low <= batch_mean <= baseline.weekly_mean_high
        per_feature[name] = FeatureDriftResult(
            feature=name,
            batch_mean=batch_mean,
            reference_low=baseline.weekly_mean_low,
            reference_high=baseline.weekly_mean_high,
            drifted=is_drifted,
        )
        if is_drifted:
            drifted.append(name)

    return DriftReport(
        status="drift" if drifted else "ok",
        checked_at=datetime.now(timezone.utc).isoformat(),
        n_rows=len(batch_df),
        drifted_features=drifted,
        per_feature=per_feature,
    )


def log_drift_report(report: DriftReport) -> None:
    """Logs the report. No auto-mitigation, no retraining: visibility only."""
    if report.status == "drift":
        logger.warning(
            "drift_check_alert",
            drifted_features=report.drifted_features,
            checked_at=report.checked_at,
            n_rows=report.n_rows,
        )
        for name in report.drifted_features:
            result = report.per_feature[name]
            logger.warning(
                "drift_check_feature",
                feature=result.feature,
                batch_mean=result.batch_mean,
                reference_low=result.reference_low,
                reference_high=result.reference_high,
            )
    else:
        logger.info("drift_check_ok", checked_at=report.checked_at, n_rows=report.n_rows)
