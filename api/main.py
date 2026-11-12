import os
import time
from contextlib import asynccontextmanager
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import pybreaker
from fastapi import FastAPI, HTTPException
from mlflow.pyfunc import PyFuncModel, load_model
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from api.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# The model is NEVER read from a loose .pth/.pkl: it is resolved by name + alias
# in the MLflow Model Registry, an immutable versioned reference that bundles
# the DLinear network and its scalers (core_ml/src/mlflow_utils.py::DLinearForecaster).
MODEL_NAME = os.getenv("MODEL_NAME", "dlinear-ett-forecaster")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
# Input shape fixed by the HTTP contract (PredictionRequest). On load it is
# checked against the registered model's own model_config
# (_check_model_matches_contract).
SEQ_LEN = 48
N_FEATURES = 10

model: PyFuncModel | None = None

# Circuit breaker: after 5 consecutive Registry failures, stop trying for 60 s
# instead of hammering a dependency that is down. Each attempt while the circuit
# is closed has its own bounded exponential-backoff retry (3 attempts at most).
MODEL_REGISTRY_BREAKER = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="mlflow-model-registry",
)


@MODEL_REGISTRY_BREAKER
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _load_model_resilient(model_uri: str) -> PyFuncModel:
    return cast(PyFuncModel, load_model(model_uri))


def _check_model_matches_contract(loaded_model: PyFuncModel) -> None:
    """Raise if the registered model expects another input shape.

    Otherwise a model retrained with another seq_len would pass /ready and every
    /predict would fail, or worse, return garbage.
    """
    config = getattr(getattr(loaded_model, "metadata", None), "flavors", {}) or {}
    model_config = config.get("python_function", {}).get("model_config") or {}
    expected = {"seq_len": SEQ_LEN, "n_features": N_FEATURES}
    for key, value in expected.items():
        declared = model_config.get(key)
        if declared is not None and int(declared) != value:
            raise ValueError(
                f"The model declares {key}={declared}, but the API contract expects {value}."
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info("model_load_started", model_uri=model_uri)
    try:
        loaded_model = _load_model_resilient(model_uri)
        _check_model_matches_contract(loaded_model)
        model = loaded_model
        logger.info("model_load_succeeded", model_uri=model_uri)
    except Exception:
        # The process stays up: a pod deployed before the first quality_gate.py
        # run, or while the Registry is unreachable, keeps /health at 200
        # (liveness) and /predict at 503. There is no background retry: loading
        # again needs a pod restart or rollout.
        logger.error("model_load_failed", model_uri=model_uri, exc_info=True)
        model = None

    yield

    model = None


app = FastAPI(
    title="ETTh1 DLinear Time Series Prediction API",
    description="Production inference API on EKS for multivariate forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)


class FeatureReading(BaseModel):
    """One hourly ETT sensor reading plus its calendar features.

    Ranges mirror core_ml/src/data_contracts.py::ETTDatasetContract, the source
    of truth; repeated here so bad input fails with a 422 at the API edge,
    before the model runs.
    """

    HUFL: float = Field(ge=-40.0, le=40.0)
    HULL: float = Field(ge=-15.0, le=20.0)
    MUFL: float = Field(ge=-40.0, le=30.0)
    MULL: float = Field(ge=-15.0, le=15.0)
    LUFL: float = Field(ge=-10.0, le=15.0)
    LULL: float = Field(ge=-5.0, le=8.0)
    OT: float = Field(ge=-15.0, le=60.0)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hour: int = Field(ge=0, le=23)


class PredictionRequest(BaseModel):
    features: list[FeatureReading] = Field(
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"Exactly {SEQ_LEN} consecutive hourly readings (the model's seq_len).",
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness, unlike /health (liveness): a pod can be alive with no model
    loaded yet (see lifespan). A readinessProbe here keeps the Service from
    routing traffic to a pod that could only answer 503.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded yet.")
    return {"status": "ready"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        logger.warning("predict_rejected_model_not_ready")
        raise HTTPException(status_code=503, detail="The model is not loaded yet.")

    start = time.monotonic()
    try:
        # Column order must match data_processing.py's feature order
        # (HUFL,HULL,MUFL,MULL,LUFL,LULL,OT,month,day,hour); `model_dump()` keeps
        # FeatureReading's declaration order, which is that one.
        df_input = pd.DataFrame([reading.model_dump() for reading in request.features])
        # PyFuncModel.predict() is typed as PyFuncOutput (a union with
        # non-indexable members such as str/dict) because it covers every model
        # flavor; this DLinear always returns an np.ndarray (core_ml/src/mlflow_utils.py).
        prediction = cast(np.ndarray, model.predict(df_input))
        # Shape (1, pred_len): one input window, pred_len future values.
        predictions = [float(v) for v in prediction[0]]

        logger.info(
            "predict_succeeded",
            model_version=f"{MODEL_NAME}@{MODEL_ALIAS}",
            horizon=len(predictions),
            duration_ms=round((time.monotonic() - start) * 1000, 2),
        )
        return {
            "model_version": f"{MODEL_NAME}@{MODEL_ALIAS}",
            "predictions": predictions,
        }
    except Exception as e:
        # The payload already passed Pydantic validation (422 otherwise), so an
        # error here is the server's: 500, without leaking the internal message.
        logger.error(
            "predict_failed",
            error=str(e),
            duration_ms=round((time.monotonic() - start) * 1000, 2),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Error interno durante la inferencia.") from e
