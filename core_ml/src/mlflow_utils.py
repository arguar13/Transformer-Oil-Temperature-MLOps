"""Lineage and model registration through MLflow.

Every training run must be reconstructable from the tuple

    Git Commit Hash + DVC Data Hash + Hyperparameters + MLflow Run ID + Container Image Tag

This module captures that tuple (as MLflow tags/params) and packages the trained
model: NEVER as a loose `.pth`/`.pkl`, but as one versioned `mlflow.pyfunc`
artifact that bundles the DLinear network with the scalers needed to serve it.
"""

from __future__ import annotations

import hashlib
import os

# Only for `git rev-parse HEAD` below, with no shell and no external input.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.pyfunc import log_model as mlflow_log_model

from src.logging_config import get_logger

logger = get_logger(__name__)

UNKNOWN = "unknown"


def get_git_commit_hash() -> str:
    """Full SHA of the current commit.

    `CI_COMMIT_SHA` (GitLab CI) first, so the recorded hash is the commit that
    triggered the pipeline even on a detached or shallow checkout. Then
    `GIT_COMMIT_SHA`: docker-compose's `trainer` container mounts only core_ml/
    (no .git) and has no git, so the Makefile passes the host's SHA in; without
    it every locally trained model would be tagged "unknown".
    """
    injected_sha = os.getenv("CI_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA")
    if injected_sha:
        return injected_sha
    try:
        # Fixed argv, no shell=True, no user input; "git" is resolved through
        # PATH on purpose so it works the same in the devcontainer, CI and bare metal.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.warning("git_commit_hash_unavailable", fallback=UNKNOWN)
        return UNKNOWN


def get_dvc_data_hash(dvc_file_path: str | Path) -> str:
    """Content MD5 of the dataset, read from its `.dvc` pointer.

    It identifies the exact bytes of the dataset (DVC recomputes it whenever the
    content changes) without the data file itself being tracked by git.
    """
    path = Path(dvc_file_path)
    if not path.exists():
        logger.warning("dvc_file_missing", path=str(path), fallback=UNKNOWN)
        return UNKNOWN
    with path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    try:
        return str(spec["outs"][0]["md5"])
    except (KeyError, IndexError, TypeError):
        logger.warning("dvc_file_unexpected_format", path=str(path), fallback=UNKNOWN)
        return UNKNOWN


def data_matches_dvc_pointer(dvc_file_path: str | Path) -> bool:
    """True if the local data file has exactly the MD5 its `.dvc` pointer declares.

    `dvc_data_hash` comes from the pointer, not from the bytes used: without
    this check, training on a modified (or differently sourced) CSV would
    record a hash that does not match the run's real data.
    """
    dvc_path = Path(dvc_file_path)
    data_path = dvc_path.with_suffix("")  # "ETTh1.csv.dvc" -> "ETTh1.csv"
    expected = get_dvc_data_hash(dvc_path)
    if expected == UNKNOWN or not data_path.exists():
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with data_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    matches = digest.hexdigest() == expected
    if not matches:
        logger.warning(
            "data_does_not_match_dvc_pointer",
            data_path=str(data_path),
            expected_md5=expected,
            actual_md5=digest.hexdigest(),
        )
    return matches


def get_container_image_tag() -> str:
    """Tag of the container image running this training.

    In CI it is exactly the tag `docker:build-push` publishes to ECR
    (`$CI_COMMIT_SHA`), linking the code that trained the model to the image
    that serves it.
    """
    return os.getenv("CI_COMMIT_SHA") or os.getenv("CONTAINER_IMAGE_TAG") or "local-dev"


def build_reproducibility_tags(dvc_file_path: str | Path) -> dict[str, str]:
    """The reproducibility tuple as MLflow-ready tags."""
    return {
        "git_commit_hash": get_git_commit_hash(),
        "dvc_data_hash": get_dvc_data_hash(dvc_file_path),
        "dvc_data_file": str(dvc_file_path),
        "data_matches_dvc_pointer": str(data_matches_dvc_pointer(dvc_file_path)).lower(),
        "container_image_tag": get_container_image_tag(),
    }


class DLinearForecaster(PythonModel):
    """Bundle the DLinear network and its pre/post-processing scalers as one
    MLflow artifact, servable with `mlflow.pyfunc.load_model(...)`.

    An immutable, versioned reference (`models:/<name>/<version or alias>`) that
    contains everything needed to produce a correct prediction.
    """

    def load_context(self, context: PythonModelContext) -> None:
        # Local import: when the model is unpickled in another process (the API),
        # only `src.model_architecture` must be importable, not the whole `src`
        # with train.py's Optuna/DVC dependencies.
        from src.model_architecture import DLinear

        seq_len = int(context.model_config["seq_len"])
        n_features = int(context.model_config["n_features"])
        pred_len = int(context.model_config.get("pred_len", 1))

        self.model = DLinear(seq_len=seq_len, n_features=n_features, pred_len=pred_len)
        state_dict = torch.load(
            context.artifacts["model_state_dict"],
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.scaler_X = joblib.load(context.artifacts["scaler_X"])
        self.scaler_y = joblib.load(context.artifacts["scaler_y"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: Any,
        params: dict | None = None,
    ):
        """Accept one sequence ([seq_len, n_features], e.g. a DataFrame or 2-D
        array, the online API case) or a batch ([n_windows, seq_len,
        n_features], batch scoring). Always ONE vectorised forward pass,
        however many sequences the batch holds.
        """
        array = np.asarray(model_input, dtype=np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # [seq_len, n_features] -> [1, seq_len, n_features]

        n_windows, seq_len, n_features = array.shape
        scaled_flat = self.scaler_X.transform(array.reshape(-1, n_features))
        scaled = scaled_flat.reshape(n_windows, seq_len, n_features)

        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            output_scaled = self.model(tensor)  # [n_windows, pred_len]

        # scaler_y was fitted on one column (the target) and expects (N, 1):
        # flatten (n_windows, pred_len) to (n_windows * pred_len, 1), unscale,
        # and restore the shape (a no-op reshape when pred_len=1).
        n_windows, pred_len = output_scaled.shape
        flat = output_scaled.numpy().reshape(-1, 1)
        return self.scaler_y.inverse_transform(flat).reshape(n_windows, pred_len)


def get_model_config(model: Any) -> dict[str, Any]:
    """The `model_config` (seq_len, n_features, pred_len) a loaded pyfunc model
    was registered with.

    Lets consumers (API, batch) validate the input shape against the model
    actually served instead of assuming a fixed seq_len. {} when the model has
    no metadata (e.g. test doubles).
    """
    metadata = getattr(model, "metadata", None)
    flavors = getattr(metadata, "flavors", None) or {}
    config = flavors.get("python_function", {}).get("model_config") or {}
    return dict(config)


def log_and_register_model(
    *,
    model_state_dict_path: str,
    scaler_x_path: str,
    scaler_y_path: str,
    seq_len: int,
    n_features: int,
    registered_model_name: str,
    pred_len: int = 1,
) -> ModelVersion:
    """Log the bundled model in the active run and register a new immutable
    version in the MLflow Model Registry.
    """
    # .as_posix(): on Windows os.path.join yields "artifacts_toy\dlinear_model.pth",
    # and MLflow copies that separator into the portable MLmodel manifest; in the
    # Linux API container "\\" separates nothing and loading fails with
    # FileNotFoundError.
    model_info = mlflow_log_model(
        artifact_path="model",
        python_model=DLinearForecaster(),
        artifacts={
            "model_state_dict": Path(model_state_dict_path).as_posix(),
            "scaler_X": Path(scaler_x_path).as_posix(),
            "scaler_y": Path(scaler_y_path).as_posix(),
        },
        model_config={"seq_len": seq_len, "n_features": n_features, "pred_len": pred_len},
        registered_model_name=registered_model_name,
    )
    logger.info(
        "model_registered",
        model_name=registered_model_name,
        model_version=model_info.registered_model_version,
        mlflow_run_id=model_info.run_id,
    )
    client = MlflowClient()
    return client.get_model_version(registered_model_name, model_info.registered_model_version)
