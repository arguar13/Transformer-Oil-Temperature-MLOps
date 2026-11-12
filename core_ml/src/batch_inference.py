import io
import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel, load_model
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_contracts import (
    ETT_CONTRACT,
    FEATURE_COLUMNS,
    DataContractError,
    validate_feature_dataframe,
)
from src.logging_config import configure_logging, get_logger
from src.mlflow_utils import get_model_config
from src.monitoring.drift_check import ReferenceProfile, check_batch_for_drift, log_drift_report

configure_logging()
logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "dlinear-ett-forecaster"
DEFAULT_MODEL_ALIAS = "production"
# Fallback only, for a loaded model that does not declare its seq_len.
DEFAULT_SEQ_LEN = 48

# Bounded retry with backoff (3 attempts at most) for transient S3/MLflow blips.
# Kubernetes' `backoffLimit` (kubernetes/base/cronjob.yaml) bounds retries of the
# whole CronJob run.
_RESILIENT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


class BatchInferenceService:
    def __init__(self):
        aws_endpoint = os.getenv("AWS_ENDPOINT_URL")
        self.s3_client = boto3.client("s3", endpoint_url=aws_endpoint)
        self.bucket = os.getenv("MODEL_BUCKET_NAME", "mlops-portafolio-proj3-models")
        self.input_key = os.getenv("BATCH_INPUT_KEY", "batch/input_data.csv")
        self.output_key = os.getenv("BATCH_OUTPUT_KEY", "batch/predictions_output.csv")
        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_alias = os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        self.seq_len = DEFAULT_SEQ_LEN
        self.batch_size = int(os.getenv("BATCH_SIZE", "512"))
        self.model: PyFuncModel | None = None
        self.reference_profile: ReferenceProfile | None = None

    @_RESILIENT_RETRY
    def load_model(self):
        """Load the immutable version behind `{model_name}@{model_alias}` from the
        Model Registry. Model and scalers travel together as one artifact;
        nothing is read from a loose file.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        # The alias is resolved ONCE to a version number, and both the model and
        # its reference profile come from that version: if the gate moved the
        # alias in between, drift would be judged against another model's profile.
        version = MlflowClient().get_model_version_by_alias(self.model_name, self.model_alias)
        model_uri = f"models:/{self.model_name}/{version.version}"
        logger.info("model_load_started", model_uri=model_uri, alias=self.model_alias)
        self.model = load_model(model_uri)
        # seq_len comes from the served model, not a constant: a retrain with
        # another window needs no code change here.
        self.seq_len = int(get_model_config(self.model).get("seq_len", DEFAULT_SEQ_LEN))

        try:
            profile_path = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path="monitoring/reference_profile.json"
            )
            self.reference_profile = ReferenceProfile.read(profile_path)
        except (MlflowException, OSError, KeyError, TypeError, ValueError):
            # The drift check is informational: a version without a profile, or
            # with one this code cannot read (corrupt JSON, another
            # ReferenceProfile format), must not block scoring.
            logger.warning("reference_profile_unavailable", run_id=version.run_id, exc_info=True)
            self.reference_profile = None
        logger.info(
            "model_load_succeeded",
            model_uri=model_uri,
            version=version.version,
            seq_len=self.seq_len,
        )

    @_RESILIENT_RETRY
    def _download_input(self) -> pd.DataFrame:
        logger.info("batch_input_download_started", bucket=self.bucket, key=self.input_key)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=self.input_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    @_RESILIENT_RETRY
    def _upload_output(self, body: str) -> None:
        logger.info("batch_output_upload_started", bucket=self.bucket, key=self.output_key)
        self.s3_client.put_object(Bucket=self.bucket, Key=self.output_key, Body=body)

    def process_batch(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        df = self._download_input()

        # Validate the contract before spending compute on inference.
        if len(df) < self.seq_len:
            raise DataContractError(f"The dataset needs at least {self.seq_len} rows.")
        validate_feature_dataframe(df, contract=ETT_CONTRACT)

        # Overlapping windows: rows i..i+seq_len-1 form one sequence.
        feature_values = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        n_windows = len(feature_values) - self.seq_len + 1
        windows = np.stack([feature_values[i : i + self.seq_len] for i in range(n_windows)])

        logger.info("batch_scoring_started", n_windows=n_windows, batch_size=self.batch_size)
        predictions: list[np.ndarray] = []
        for start in range(0, n_windows, self.batch_size):
            batch = windows[start : start + self.batch_size]
            preds = self.model.predict(batch)  # one vectorised pass per batch
            predictions.append(np.asarray(preds))

        stacked = np.concatenate(predictions, axis=0)
        # One step gives a single "Prediction" column, several give
        # "Prediction_h1".."Prediction_hN": the names follow the model's actual
        # output shape instead of assuming a horizon.
        pred_len = stacked.shape[1] if stacked.ndim > 1 else 1
        if pred_len == 1:
            columns = ["Prediction"]
        else:
            columns = [f"Prediction_h{h}" for h in range(1, pred_len + 1)]
        output_df = pd.DataFrame(stacked.reshape(len(stacked), pred_len), columns=columns)
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)

        self._upload_output(csv_buffer.getvalue())
        logger.info("batch_inference_completed", n_predictions=len(output_df))

        # Drift check on the batch just scored: it only logs, no automatic action
        # (see core_ml/src/monitoring/drift_check.py).
        if self.reference_profile is not None:
            report = check_batch_for_drift(df, self.reference_profile)
            log_drift_report(report)


if __name__ == "__main__":
    service = BatchInferenceService()
    service.load_model()
    service.process_batch()
