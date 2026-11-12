"""Quality gate for the MLflow Model Registry.

Promotes the version registered by a given training run (its run_id) to the
`production` alias ONLY if:

1. it beats the naive baseline (persistence: repeat the last value) on the same
   test split, the very first version included. A model that does not improve
   on "doing nothing" does not justify running it; and
2. its test metric is at least as good as the current `production` version's,
   and the two are comparable (same dataset, seq_len and pred_len, see
   COMPARABILITY_PARAMS).

Otherwise the script exits non-zero, which in CI blocks the deploy, so a worse
model never replaces the one serving real traffic.

"production" is a Registry alias (not the `stages` API, deprecated since MLflow
2.9): the same reference api/main.py and batch_inference.py resolve at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from src.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

PRODUCTION_ALIAS = "production"
# final_test_mse, not final_val_mse: val already chose the learning rate and the
# checkpoint, so a candidate can look good on val precisely because selection
# pushed it there. Test influenced none of that.
DEFAULT_METRIC_KEY = "final_test_mse"  # lower is better (test MSE, in °C^2)
# For every test metric train.py also logs the persistence baseline's value
# under this suffix (final_test_mse_persistence).
BASELINE_METRIC_SUFFIX = "_persistence"


# train.py writes the run_id of the training it just finished here, and the gate
# judges exactly that version. "The highest version number" would race when two
# trainings register at once: the gate could promote the other one's model.
RUN_ID_FILE = "run_id.txt"


def _version_for_run(client: MlflowClient, model_name: str, run_id: str) -> ModelVersion:
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    if not versions:
        raise SystemExit(f"No registered version of '{model_name}' for run {run_id}.")
    return max(versions, key=lambda v: int(v.version))


# train.py params that define WHAT the test metric measures: two versions are
# comparable only if all of them match. An MSE on the toy dataset's test split
# (1,000 rows) says nothing next to one on the full ETTh1, nor does a 1-step
# MSE next to a 48-step one.
COMPARABILITY_PARAMS = ("dataset", "seq_len", "pred_len")
SMOKE_TEST_DATASET = "toy"


def _metric_for_version(
    client: MlflowClient, version: ModelVersion, metric_key: str
) -> tuple[float | None, dict[str, str]]:
    """(metric, run params) of the given version."""
    if version.run_id is None:
        raise SystemExit(f"Version {version.version} has no associated run_id.")
    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric_key)
    return (float(value) if value is not None else None), dict(run.data.params)


def _current_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def run_quality_gate(model_name: str, run_id: str, metric_key: str = DEFAULT_METRIC_KEY) -> bool:
    """True if the version registered by `run_id` was promoted to production."""
    client = MlflowClient()

    candidate = _version_for_run(client, model_name, run_id)
    candidate_metric, candidate_params = _metric_for_version(client, candidate, metric_key)
    if candidate_metric is None:
        raise SystemExit(
            f"Candidate version v{candidate.version} of '{model_name}' has no "
            f"'{metric_key}' metric logged; the quality gate cannot be evaluated."
        )

    # The minimum bar, before looking at production: a candidate trained on real
    # data must beat persistence on its own test split. The toy dataset is
    # exempt: its ~100 test windows exercise the plumbing, not skill, and a toy
    # model can never replace a real one (see below).
    if candidate_params.get("dataset") != SMOKE_TEST_DATASET:
        baseline_key = metric_key + BASELINE_METRIC_SUFFIX
        baseline_metric, _ = _metric_for_version(client, candidate, baseline_key)
        if baseline_metric is None:
            raise SystemExit(
                f"Candidate version v{candidate.version} of '{model_name}' has no "
                f"'{baseline_key}' metric logged; it cannot be compared against the "
                "baseline."
            )
        if candidate_metric >= baseline_metric:
            logger.error(
                "quality_gate_rejected_not_better_than_persistence",
                model_name=model_name,
                candidate_version=candidate.version,
                candidate_metric=candidate_metric,
                persistence_metric=baseline_metric,
                metric_key=metric_key,
            )
            return False

    production = _current_production_version(client, model_name)
    if production is None:
        logger.info(
            "quality_gate_baseline_promoted",
            model_name=model_name,
            version=candidate.version,
            metric_key=metric_key,
            metric_value=candidate_metric,
        )
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate.version)
        return True

    if candidate.version == production.version:
        logger.info("quality_gate_noop_already_production", version=candidate.version)
        return True

    production_metric, production_params = _metric_for_version(client, production, metric_key)
    logger.info(
        "quality_gate_evaluating",
        candidate_version=candidate.version,
        candidate_metric=candidate_metric,
        production_version=production.version,
        production_metric=production_metric,
        metric_key=metric_key,
    )

    candidate_dataset = candidate_params.get("dataset")
    production_dataset = production_params.get("dataset")
    if candidate_dataset == SMOKE_TEST_DATASET and production_dataset != SMOKE_TEST_DATASET:
        # A smoke-test model never replaces one trained on real data, however
        # good its metric looks: it was measured on a different test split.
        logger.error(
            "quality_gate_rejected_smoke_test_candidate",
            candidate_version=candidate.version,
            production_version=production.version,
            production_dataset=production_dataset,
        )
        return False
    if production_dataset == SMOKE_TEST_DATASET and candidate_dataset != SMOKE_TEST_DATASET:
        # The reverse: production only holds a smoke-test model, so any model
        # trained on real data becomes the new baseline.
        return _promote(
            client,
            model_name,
            candidate,
            production,
            metric_key,
            candidate_metric,
            production_metric,
        )

    mismatched = {
        key: (candidate_params.get(key), production_params.get(key))
        for key in COMPARABILITY_PARAMS
        if candidate_params.get(key) != production_params.get(key)
    }
    if mismatched:
        logger.error(
            "quality_gate_rejected_not_comparable",
            candidate_version=candidate.version,
            production_version=production.version,
            mismatched_params=mismatched,
            hint="Promote by hand if the configuration change is intentional.",
        )
        return False

    if production_metric is not None and candidate_metric > production_metric:
        logger.error(
            "quality_gate_rejected",
            model_name=model_name,
            candidate_version=candidate.version,
            candidate_metric=candidate_metric,
            production_version=production.version,
            production_metric=production_metric,
            metric_key=metric_key,
        )
        return False

    return _promote(
        client, model_name, candidate, production, metric_key, candidate_metric, production_metric
    )


def _promote(
    client: MlflowClient,
    model_name: str,
    candidate: ModelVersion,
    production: ModelVersion,
    metric_key: str,
    candidate_metric: float,
    production_metric: float | None,
) -> bool:
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate.version)
    logger.info(
        "quality_gate_promoted",
        model_name=model_name,
        previous_version=production.version,
        new_version=candidate.version,
        metric_key=metric_key,
        previous_metric=production_metric,
        new_metric=candidate_metric,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality gate for promotion to production (MLflow Registry)"
    )
    parser.add_argument("--model_name", type=str, default="dlinear-ett-forecaster")
    parser.add_argument("--metric_key", type=str, default=DEFAULT_METRIC_KEY)
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=f"Training run to evaluate (defaults to the one in {RUN_ID_FILE}).",
    )
    args = parser.parse_args()

    run_id = args.run_id
    if run_id is None:
        if not os.path.exists(RUN_ID_FILE):
            raise SystemExit(
                f"{RUN_ID_FILE} is missing: run train.py first, or pass --run_id explicitly."
            )
        with open(RUN_ID_FILE, encoding="utf-8") as handle:
            run_id = handle.read().strip()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    promoted = run_quality_gate(args.model_name, run_id, args.metric_key)
    sys.exit(0 if promoted else 1)
