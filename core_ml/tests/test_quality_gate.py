import mlflow
import mlflow.pyfunc
import pytest
from mlflow.exceptions import MlflowException

from src.quality_gate import run_quality_gate


class _DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture()
def mlflow_local_registry(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("quality-gate-test")
    yield


def _log_and_register(
    model_name: str, metric_value: float, persistence_value: float | None = None, **params
) -> str:
    """Log and register a version; return the run_id that registered it."""
    # By default persistence is worse than the candidate, so the tests that are
    # not about the baseline only exercise the version comparison.
    if persistence_value is None:
        persistence_value = metric_value + 1.0
    with mlflow.start_run():
        mlflow.log_metric("final_test_mse", metric_value)
        mlflow.log_metric("final_test_mse_persistence", persistence_value)
        if params:
            mlflow.log_params(params)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_DummyModel(),
            registered_model_name=model_name,
        )
        return mlflow.active_run().info.run_id


def test_first_version_is_promoted_as_baseline(mlflow_local_registry):
    run_id = _log_and_register("baseline-model", metric_value=1.0)

    promoted = run_quality_gate("baseline-model", run_id)

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("baseline-model", "production")
    assert str(production.version) == "1"


def test_better_candidate_is_promoted(mlflow_local_registry):
    run_id = _log_and_register("improving-model", metric_value=1.0)
    run_quality_gate("improving-model", run_id)

    run_id = _log_and_register("improving-model", metric_value=0.5)  # lower MSE = better
    promoted = run_quality_gate("improving-model", run_id)

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("improving-model", "production")
    assert str(production.version) == "2"


def test_worse_candidate_is_not_promoted(mlflow_local_registry):
    run_id = _log_and_register("regressing-model", metric_value=0.5)
    run_quality_gate("regressing-model", run_id)

    run_id = _log_and_register("regressing-model", metric_value=1.0)  # worse MSE
    promoted = run_quality_gate("regressing-model", run_id)

    assert promoted is False
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("regressing-model", "production")
    assert str(production.version) == "1"  # still v1, not downgraded


def test_equal_metric_still_promotes(mlflow_local_registry):
    run_id = _log_and_register("tied-model", metric_value=1.0)
    run_quality_gate("tied-model", run_id)

    run_id = _log_and_register("tied-model", metric_value=1.0)  # exact tie
    promoted = run_quality_gate("tied-model", run_id)

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("tied-model", "production")
    assert str(production.version) == "2"


def test_no_registered_versions_raises(mlflow_local_registry):
    with pytest.raises(SystemExit, match="No registered version"):
        run_quality_gate("nonexistent-model", "no-such-run")


def test_gate_evaluates_its_own_run_not_the_latest_version(mlflow_local_registry):
    """Two trainings register back to back: each gate judges its own version."""
    run_quality_gate("racing-model", _log_and_register("racing-model", metric_value=1.0))
    good_run = _log_and_register("racing-model", metric_value=0.5)
    _log_and_register("racing-model", metric_value=9.0)  # a later, worse training

    promoted = run_quality_gate("racing-model", good_run)

    assert promoted is True
    assert _production_version("racing-model") == "2"


def _production_version(model_name: str) -> str:
    client = mlflow.MlflowClient()
    return str(client.get_model_version_by_alias(model_name, "production").version)


REAL = {"dataset": "raw", "seq_len": 48, "pred_len": 48}
TOY = {"dataset": "toy", "seq_len": 48, "pred_len": 48}


def test_toy_candidate_never_replaces_a_real_production_model(mlflow_local_registry):
    run_id = _log_and_register("guarded-model", metric_value=5.0, **REAL)
    run_quality_gate("guarded-model", run_id)

    # A better metric, but measured on the toy dataset's test split.
    run_id = _log_and_register("guarded-model", metric_value=0.1, **TOY)
    promoted = run_quality_gate("guarded-model", run_id)

    assert promoted is False
    assert _production_version("guarded-model") == "1"


def test_real_candidate_replaces_a_toy_production_model(mlflow_local_registry):
    run_id = _log_and_register("smoke-model", metric_value=0.1, **TOY)
    run_quality_gate("smoke-model", run_id)

    run_id = _log_and_register("smoke-model", metric_value=5.0, **REAL)
    promoted = run_quality_gate("smoke-model", run_id)

    assert promoted is True
    assert _production_version("smoke-model") == "2"


def test_candidate_with_a_different_horizon_is_not_comparable(mlflow_local_registry):
    run_id = _log_and_register("horizon-model", metric_value=5.0, **REAL)
    run_quality_gate("horizon-model", run_id)

    run_id = _log_and_register("horizon-model", metric_value=0.1, **{**REAL, "pred_len": 1})
    promoted = run_quality_gate("horizon-model", run_id)

    assert promoted is False
    assert _production_version("horizon-model") == "1"


def test_first_real_version_worse_than_persistence_is_not_promoted(mlflow_local_registry):
    # Not even as the first baseline: with no production yet, it still has to
    # beat repeating the last reading.
    run_id = _log_and_register("naive-model", metric_value=6.0, persistence_value=5.9, **REAL)

    promoted = run_quality_gate("naive-model", run_id)

    assert promoted is False
    client = mlflow.MlflowClient()
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("naive-model", "production")


def test_candidate_tied_with_persistence_is_not_promoted(mlflow_local_registry):
    run_id = _log_and_register("tied-baseline-model", metric_value=5.0, **REAL)
    run_quality_gate("tied-baseline-model", run_id)

    # Better than production (5.0), but exactly equal to persistence.
    run_id = _log_and_register(
        "tied-baseline-model", metric_value=4.0, persistence_value=4.0, **REAL
    )
    promoted = run_quality_gate("tied-baseline-model", run_id)

    assert promoted is False
    assert _production_version("tied-baseline-model") == "1"


def test_toy_candidate_is_exempt_from_the_persistence_rule(mlflow_local_registry):
    run_id = _log_and_register("toy-smoke-model", metric_value=9.0, persistence_value=8.0, **TOY)

    promoted = run_quality_gate("toy-smoke-model", run_id)

    assert promoted is True
    assert _production_version("toy-smoke-model") == "1"


def test_candidate_without_persistence_metric_raises(mlflow_local_registry):
    with mlflow.start_run() as run:
        mlflow.log_metric("final_test_mse", 1.0)
        mlflow.log_params(REAL)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_DummyModel(),
            registered_model_name="no-baseline-model",
        )

    with pytest.raises(SystemExit, match="final_test_mse_persistence"):
        run_quality_gate("no-baseline-model", run.info.run_id)
