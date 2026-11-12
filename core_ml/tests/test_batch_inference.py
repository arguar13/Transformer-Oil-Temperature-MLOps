import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.batch_inference import BatchInferenceService
from src.data_contracts import FEATURE_COLUMNS, DataContractError
from src.monitoring.drift_check import FeatureBaseline, ReferenceProfile


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload


class _FakePyfuncModel:
    def predict(self, model_input):
        n_windows = model_input.shape[0] if model_input.ndim == 3 else 1
        return np.full((n_windows, 1), 42.5)


class _FakeMultiStepPyfuncModel:
    """A fake model with a multi-step horizon (pred_len > 1)."""

    def __init__(self, pred_len: int):
        self.pred_len = pred_len

    def predict(self, model_input):
        n_windows = model_input.shape[0] if model_input.ndim == 3 else 1
        return np.full((n_windows, self.pred_len), 42.5)


def _valid_batch_df(n_rows: int) -> pd.DataFrame:
    row = {
        "HUFL": 5.8,
        "HULL": 2.0,
        "MUFL": 1.6,
        "MULL": 0.5,
        "LUFL": 4.2,
        "LULL": 1.3,
        "OT": 30.5,
        "month": 7,
        "day": 1,
        "hour": 0,
    }
    return pd.DataFrame([row] * n_rows, columns=list(FEATURE_COLUMNS))


def _reference_profile_matching(row: dict) -> ReferenceProfile:
    """A reference profile centred exactly on `row`, so the test batch never
    triggers drift by accident."""
    return ReferenceProfile(
        created_at="2024-01-01T00:00:00+00:00",
        n_rows=1000,
        features={
            col: FeatureBaseline(
                mean=row[col],
                std=1.0,
                weekly_mean_low=row[col] - 1.0,
                weekly_mean_high=row[col] + 1.0,
            )
            for col in ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
        },
    )


@pytest.fixture()
def service():
    svc = BatchInferenceService()
    svc.model = _FakePyfuncModel()
    svc.reference_profile = _reference_profile_matching(
        {"HUFL": 5.8, "HULL": 2.0, "MUFL": 1.6, "MULL": 0.5, "LUFL": 4.2, "LULL": 1.3, "OT": 30.5}
    )
    return svc


def test_process_batch_rejects_input_shorter_than_seq_len(monkeypatch, service):
    short_df = _valid_batch_df(service.seq_len - 1)
    payload = short_df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    with pytest.raises(DataContractError, match="at least"):
        service.process_batch()


def test_process_batch_rejects_out_of_range_values(monkeypatch, service):
    df = _valid_batch_df(service.seq_len + 5)
    df.loc[0, "OT"] = 999.0
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    with pytest.raises(DataContractError):
        service.process_batch()


def test_process_batch_scores_and_uploads_predictions(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    uploaded = {}

    def _fake_put_object(Bucket, Key, Body):
        uploaded["bucket"] = Bucket
        uploaded["key"] = Key
        uploaded["body"] = Body

    monkeypatch.setattr(service.s3_client, "put_object", _fake_put_object)

    service.process_batch()

    assert uploaded["bucket"] == service.bucket
    assert uploaded["key"] == service.output_key
    result_df = pd.read_csv(io.StringIO(uploaded["body"]))
    assert list(result_df.columns) == ["Prediction"]
    assert len(result_df) == n_rows - service.seq_len + 1
    assert (result_df["Prediction"] == 42.5).all()


def test_process_batch_logs_an_ok_drift_report_when_batch_matches_reference(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client, "get_object", lambda Bucket, Key: {"Body": _FakeBody(payload)}
    )
    monkeypatch.setattr(service.s3_client, "put_object", lambda **_: None)

    reports = []
    monkeypatch.setattr(
        "src.batch_inference.log_drift_report", lambda report: reports.append(report)
    )

    service.process_batch()

    assert len(reports) == 1
    assert reports[0].status == "ok"


def test_process_batch_logs_a_drift_report_when_batch_diverges_from_reference(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    # Inside the contract's valid range ([-15, 60]) but far from the profile's
    # OT=30.5: drift must be flagged, not a contract violation.
    df["OT"] = 55.0
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client, "get_object", lambda Bucket, Key: {"Body": _FakeBody(payload)}
    )
    monkeypatch.setattr(service.s3_client, "put_object", lambda **_: None)

    reports = []
    monkeypatch.setattr(
        "src.batch_inference.log_drift_report", lambda report: reports.append(report)
    )

    service.process_batch()

    assert len(reports) == 1
    assert reports[0].status == "drift"
    assert "OT" in reports[0].drifted_features


def test_process_batch_skips_drift_check_when_no_reference_profile_loaded(monkeypatch, service):
    service.reference_profile = None
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client, "get_object", lambda Bucket, Key: {"Body": _FakeBody(payload)}
    )
    monkeypatch.setattr(service.s3_client, "put_object", lambda **_: None)

    reports = []
    monkeypatch.setattr(
        "src.batch_inference.log_drift_report", lambda report: reports.append(report)
    )

    service.process_batch()

    assert reports == []


def test_process_batch_names_columns_per_horizon_step_for_multi_step_model(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()
    service.model = _FakeMultiStepPyfuncModel(pred_len=3)

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    uploaded = {}
    monkeypatch.setattr(
        service.s3_client,
        "put_object",
        lambda Bucket, Key, Body: uploaded.update(bucket=Bucket, key=Key, body=Body),
    )

    service.process_batch()

    result_df = pd.read_csv(io.StringIO(uploaded["body"]))
    assert list(result_df.columns) == ["Prediction_h1", "Prediction_h2", "Prediction_h3"]
    assert len(result_df) == n_rows - service.seq_len + 1
    assert (result_df == 42.5).all().all()


class _FakeRegistryClient:
    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version="7", run_id="run-123")


def _model_with_config(**config):
    model = _FakePyfuncModel()
    model.metadata = SimpleNamespace(flavors={"python_function": {"model_config": config}})
    return model


def test_load_model_pins_the_resolved_version_and_reads_seq_len_from_it(monkeypatch, tmp_path):
    profile = _reference_profile_matching(
        {"HUFL": 5.8, "HULL": 2.0, "MUFL": 1.6, "MULL": 0.5, "LUFL": 4.2, "LULL": 1.3, "OT": 30.5}
    )
    profile_path = profile.write(tmp_path / "reference_profile.json")
    loaded_uris = []

    monkeypatch.setattr("src.batch_inference.MlflowClient", _FakeRegistryClient)
    monkeypatch.setattr(
        "src.batch_inference.load_model",
        lambda uri: loaded_uris.append(uri) or _model_with_config(seq_len=24, n_features=10),
    )
    monkeypatch.setattr(
        "src.batch_inference.mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path: str(profile_path),
    )

    svc = BatchInferenceService()
    svc.load_model()

    # A fixed version, not the alias: model and profile come from the SAME version.
    assert loaded_uris == [f"models:/{svc.model_name}/7"]
    assert svc.seq_len == 24
    assert svc.reference_profile is not None


def test_load_model_tolerates_a_missing_reference_profile(monkeypatch):
    def _missing(run_id, artifact_path):
        raise MlflowException("artifact not found")

    monkeypatch.setattr("src.batch_inference.MlflowClient", _FakeRegistryClient)
    monkeypatch.setattr("src.batch_inference.load_model", lambda uri: _model_with_config())
    monkeypatch.setattr("src.batch_inference.mlflow.artifacts.download_artifacts", _missing)

    svc = BatchInferenceService()
    svc.load_model()

    assert svc.model is not None
    assert svc.reference_profile is None
    assert svc.seq_len == 48  # no model_config: default fallback


def test_load_model_tolerates_a_reference_profile_in_an_unknown_format(monkeypatch, tmp_path):
    # A profile in another FeatureBaseline format (mean/std only) raises
    # TypeError on read, and scoring must go on without the drift check.
    old_format = tmp_path / "reference_profile.json"
    old_format.write_text(
        '{"created_at": "2024-01-01T00:00:00+00:00", "n_rows": 10,'
        ' "features": {"OT": {"mean": 16.3, "std": 8.4}}}'
    )
    monkeypatch.setattr("src.batch_inference.MlflowClient", _FakeRegistryClient)
    monkeypatch.setattr("src.batch_inference.load_model", lambda uri: _model_with_config())
    monkeypatch.setattr(
        "src.batch_inference.mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path: str(old_format),
    )

    svc = BatchInferenceService()
    svc.load_model()

    assert svc.model is not None
    assert svc.reference_profile is None
