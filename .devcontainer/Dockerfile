FROM mcr.microsoft.com/devcontainers/python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

# Installed in its own layer so Docker caches it.
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 -
ENV PATH="/root/.local/bin:$PATH"

# make: the single command interface (local == CI). postgresql-client: RDS/MLflow debugging.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends make postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# pre-commit at system level, so the git hooks do not depend on any
# sub-project's venv.
RUN pip install --no-cache-dir pre-commit
