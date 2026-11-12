import hashlib
from types import SimpleNamespace

import numpy as np

from src.mlflow_utils import (
    DLinearForecaster,
    build_reproducibility_tags,
    data_matches_dvc_pointer,
    get_container_image_tag,
    get_dvc_data_hash,
    get_git_commit_hash,
    get_model_config,
)
from src.model_architecture import DLinear


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


def test_get_git_commit_hash_prefers_ci_env_var(monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "abc123")

    assert get_git_commit_hash() == "abc123"


def test_get_git_commit_hash_uses_injected_sha_outside_ci(monkeypatch):
    # The local trainer container has no .git: the Makefile injects GIT_COMMIT_SHA.
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.setenv("GIT_COMMIT_SHA", "sha-from-host")

    assert get_git_commit_hash() == "sha-from-host"


def test_get_git_commit_hash_falls_back_to_git(monkeypatch):
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)

    result = get_git_commit_hash()

    # Inside a git repo it resolves a real 40-hex SHA, or "unknown" when there
    # are no commits yet; it never raises.
    assert result == "unknown" or (
        len(result) == 40 and all(c in "0123456789abcdef" for c in result)
    )


def test_get_container_image_tag_prefers_ci_commit_sha(monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "sha-from-ci")
    monkeypatch.delenv("CONTAINER_IMAGE_TAG", raising=False)

    assert get_container_image_tag() == "sha-from-ci"


def test_get_container_image_tag_defaults_to_local_dev(monkeypatch):
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.delenv("CONTAINER_IMAGE_TAG", raising=False)

    assert get_container_image_tag() == "local-dev"


def test_get_dvc_data_hash_reads_md5_from_dvc_file(tmp_path):
    dvc_file = tmp_path / "data.csv.dvc"
    fake_md5 = "not-a-real-hash-just-for-testing"
    dvc_file.write_text(f"outs:\n- md5: {fake_md5}\n  size: 10\n  path: data.csv\n")

    assert get_dvc_data_hash(dvc_file) == fake_md5


def test_get_dvc_data_hash_missing_file_returns_unknown(tmp_path):
    assert get_dvc_data_hash(tmp_path / "nope.csv.dvc") == "unknown"


def test_build_reproducibility_tags_has_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "commit-sha")
    dvc_file = tmp_path / "data.csv.dvc"
    dvc_file.write_text("outs:\n- md5: hash123\n  size: 10\n  path: data.csv\n")

    tags = build_reproducibility_tags(dvc_file)

    assert tags["git_commit_hash"] == "commit-sha"
    assert tags["dvc_data_hash"] == "hash123"
    assert tags["container_image_tag"] == "commit-sha"


def test_dlinear_forecaster_predict_handles_single_sequence():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    single_sequence = np.random.rand(48, 10)

    result = forecaster.predict(context=None, model_input=single_sequence)

    assert result.shape == (1, 1)


def test_dlinear_forecaster_predict_handles_batch_of_sequences():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    batch = np.random.rand(5, 48, 10)

    result = forecaster.predict(context=None, model_input=batch)

    assert result.shape == (5, 1)


def test_dlinear_forecaster_predict_handles_multi_step_horizon():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10, pred_len=48)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    batch = np.random.rand(5, 48, 10)

    result = forecaster.predict(context=None, model_input=batch)

    assert result.shape == (5, 48)


def _write_data_and_pointer(tmp_path, content: bytes, pointer_md5: str):
    (tmp_path / "data.csv").write_bytes(content)
    dvc_file = tmp_path / "data.csv.dvc"
    dvc_file.write_text(f"outs:\n- md5: {pointer_md5}\n  size: 10\n  path: data.csv\n")
    return dvc_file


def test_data_matches_dvc_pointer_when_hashes_agree(tmp_path):
    content = b"date,OT\n2016-07-01 00:00:00,30.5\n"
    dvc_file = _write_data_and_pointer(tmp_path, content, hashlib.md5(content).hexdigest())

    assert data_matches_dvc_pointer(dvc_file) is True


def test_data_does_not_match_dvc_pointer_when_file_differs(tmp_path):
    dvc_file = _write_data_and_pointer(tmp_path, b"other content", "0" * 32)

    assert data_matches_dvc_pointer(dvc_file) is False


def test_data_does_not_match_dvc_pointer_when_data_file_is_missing(tmp_path):
    dvc_file = tmp_path / "data.csv.dvc"
    dvc_file.write_text("outs:\n- md5: abc\n  size: 10\n  path: data.csv\n")

    assert data_matches_dvc_pointer(dvc_file) is False


def test_get_model_config_reads_pyfunc_flavor_config():
    model = SimpleNamespace(
        metadata=SimpleNamespace(
            flavors={"python_function": {"model_config": {"seq_len": 24, "pred_len": 12}}}
        )
    )

    assert get_model_config(model) == {"seq_len": 24, "pred_len": 12}


def test_get_model_config_is_empty_for_objects_without_metadata():
    assert get_model_config(object()) == {}
