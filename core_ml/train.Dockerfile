# Local training image (see docker-compose.yml::trainer).
#
# mlflow.pyfunc.log_model() stores each artifact's relative path
# ("model_state_dict", "scaler_X", ...) with the separator of the OS that saves
# it (mlflow/pyfunc/model.py, _save_model_with_class_artifacts_params:
# os.path.join(...)), and that literal stays in the model manifest. Training
# directly on a Windows host records "artifacts\dlinear_model.pth", which the
# Linux API container cannot find. CI already trains inside `image: python:3.10`
# (.gitlab-ci.yml::train); this image gives local runs that same environment
# whatever the host OS.
FROM python:3.10-slim

WORKDIR /workspace

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencies are baked into the image instead of installed on every
# `docker compose run`, which cost several minutes per `make train`. This layer
# only rebuilds when the lockfile changes (`docker compose build trainer`;
# `make up` does not, trainer lives in the "tools" profile).
#
# "poetry export" + pip, not "poetry install": on Docker Desktop/WSL2 with
# python:3.10-slim, `poetry install` with core_ml's lockfile dies silently
# (exit 1, no trace) while setting up the virtualenv, and "poetry export" never
# touches that code path. --timeout 120 (pip's default is 15 s) gives torch time
# to download on a shared network.
COPY core_ml/pyproject.toml core_ml/poetry.lock /tmp/deps/
RUN pip install --no-cache-dir "poetry==1.8.3" "poetry-plugin-export==1.8.0" \
    && poetry export -C /tmp/deps -f requirements.txt --without-hashes --only main -o /tmp/deps/requirements.txt \
    && pip install --no-cache-dir --timeout 120 -r /tmp/deps/requirements.txt \
    && pip uninstall -y poetry poetry-plugin-export \
    && rm -rf /tmp/deps

# The code (core_ml/) arrives through a bind mount (docker-compose.yml), so
# editing src/ needs no rebuild.
