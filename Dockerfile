# DLinear serving image: the FastAPI app and the batch-scoring CronJob share it.
FROM python:3.10-slim

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" "poetry-plugin-export==1.8.0"

# poetry.lock is copied without a wildcard: the build fails if the lockfile is
# not committed, instead of silently falling back to a non-deterministic resolve.
COPY api/pyproject.toml api/poetry.lock ./
# "poetry export" + pip, not "poetry install --sync": the latter died silently
# and intermittently on this host (exit 1, no traceback) with packages that
# build C extensions, the same issue core_ml/train.Dockerfile and the Makefile's
# install-core-ml-ci work around.
# --timeout 120 (pip's default is 15 s): large wheels such as torch timed out
# reading from files.pythonhosted.org on a busy shared host; pip already retries
# five times, what was missing was time per attempt.
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt --only main \
    && pip install --no-cache-dir --timeout 120 -r requirements.txt \
    && rm requirements.txt

COPY api/ ./api/

# The mlflow.pyfunc wrapper registered by core_ml/src/train.py references these
# modules by path (`src.model_architecture`, `src.mlflow_utils`), so they must
# exist here under the same package name to unpickle the model.
# batch_inference.py is the entrypoint of the batch-scoring CronJob
# (kubernetes/base/cronjob.yaml).
COPY core_ml/src/__init__.py ./src/__init__.py
COPY core_ml/src/model_architecture.py ./src/model_architecture.py
COPY core_ml/src/mlflow_utils.py ./src/mlflow_utils.py
COPY core_ml/src/data_contracts.py ./src/data_contracts.py
COPY core_ml/src/logging_config.py ./src/logging_config.py
COPY core_ml/src/batch_inference.py ./src/batch_inference.py
COPY core_ml/src/monitoring/ ./src/monitoring/

# Unprivileged user with a numeric UID, so Kubernetes can verify
# `runAsNonRoot: true` (kubernetes/base/deployment.yaml). MLflow downloads the
# model to a temp dir / HOME, both writable.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
