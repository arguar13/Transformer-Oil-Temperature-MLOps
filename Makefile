# ==============================================================================
# Makefile: the single command interface (local == CI)
#
# Every quality, dependency and test command goes through here, and the GitLab
# CI pipeline (.gitlab-ci.yml) calls these same targets.
# ==============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

API_DIR    := api
CORE_DIR   := core_ml
PROJECTS   := $(API_DIR) $(CORE_DIR)

# Used by docker-build/docker-push/deploy; override them, e.g.
# `make docker-push AWS_ACCOUNT_ID=123456789012`.
AWS_REGION       ?= us-east-1
AWS_ACCOUNT_ID   ?=
ECR_REPOSITORY   ?= dlinear-forecast-api
IMAGE_TAG        ?= $(shell git rev-parse HEAD)
ECR_REGISTRY     := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
# The host's SHA, passed to the trainer container (which cannot see .git): it
# becomes every MLflow run's git_commit_hash tag (mlflow_utils.get_git_commit_hash).
GIT_SHA          := $(shell git rev-parse HEAD 2>/dev/null)
TRAINER_RUN      := docker compose run --rm -e GIT_COMMIT_SHA=$(GIT_SHA) trainer
# Bucket created by localstack/init/01-bootstrap.sh (local stack only).
LOCAL_BUCKET     := mlops-portafolio-proj3-models

.PHONY: help install install-api install-core-ml install-core-ml-ci lock \
	format format-check lint type-check test \
	hooks pre-commit-run ci clean \
	dvc-pull dvc-push dvc-status data-download data-toy data-raw train train-toy quality-gate reload-api batch-local mlflow-ui \
	up down restart reset-local logs ps \
	docker-build docker-push deploy \
	tf-fmt tf-validate tf-test tf-plan tf-apply k8s-build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------
# Deterministic dependencies (Poetry + lockfile)
# ------------------------------------------------------------------------------
install: install-api install-core-ml ## Install the exact poetry.lock dependencies of both sub-projects

install-api: ## Install the api/ service dependencies (from poetry.lock)
	cd $(API_DIR) && poetry install --sync --no-interaction

install-core-ml: ## Install the core_ml/ pipeline dependencies (from poetry.lock)
	cd $(CORE_DIR) && poetry install --sync --no-interaction

# "poetry install" of core_ml's lockfile fails intermittently in some Linux
# containers; "poetry export" avoids that code path: it writes a pinned
# requirements.txt from the lockfile and pip installs from it.
install-core-ml-ci: ## Install core_ml/ through pip from the exported lockfile (Linux containers: CI, training)
	cd $(CORE_DIR) && poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt --with dev \
		&& pip install --no-cache-dir -q -r /tmp/requirements.txt

lock: ## Regenerate poetry.lock in both sub-projects (after editing pyproject.toml)
	@for d in $(PROJECTS); do \
		echo "==> poetry lock ($$d)"; \
		(cd $$d && poetry lock) || exit 1; \
	done

# ------------------------------------------------------------------------------
# Validation: formatting, lint, types, tests
# ------------------------------------------------------------------------------
format: ## Format the code (ruff) in both sub-projects
	@for d in $(PROJECTS); do \
		echo "==> Formatting $$d"; \
		(cd $$d && poetry run ruff check --fix . && poetry run ruff format .) || exit 1; \
	done

format-check: ## Check formatting without changing files (used in CI)
	@for d in $(PROJECTS); do \
		echo "==> Checking format $$d"; \
		(cd $$d && poetry run ruff format --check .) || exit 1; \
	done

lint: ## Run Ruff (lint + import order) in both sub-projects
	@for d in $(PROJECTS); do \
		echo "==> Lint $$d"; \
		(cd $$d && poetry run ruff check .) || exit 1; \
	done

type-check: ## Run mypy in both sub-projects (advisory, not part of `make ci`)
	@for d in $(PROJECTS); do \
		echo "==> mypy $$d"; \
		(cd $$d && poetry run mypy .) || exit 1; \
	done

test: ## Run pytest with coverage in both sub-projects
	@for d in $(PROJECTS); do \
		echo "==> pytest $$d"; \
		(cd $$d && poetry run pytest) || exit 1; \
	done

# ------------------------------------------------------------------------------
# Versioned data and models (DVC + MLflow)
# ------------------------------------------------------------------------------
dvc-pull: ## Download the versioned datasets (toy + raw) from the S3/LocalStack remote
	cd $(CORE_DIR) && poetry run dvc pull

dvc-push: ## Upload the versioned datasets to the S3/LocalStack remote
	cd $(CORE_DIR) && poetry run dvc push

dvc-status: ## Show whether the local datasets match the remote
	cd $(CORE_DIR) && poetry run dvc status -c

data-download: ## Download ETTh1 from the public repo and build the toy set (dvc-pull without a remote)
	cd $(CORE_DIR) && poetry run python -m scripts.download_data

data-toy: ## Process the toy dataset (~1,000 rows) for a quick end-to-end cycle
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset toy --output_dir artifacts_toy

data-raw: ## Process the full production dataset
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset raw --output_dir artifacts

# train.py and quality_gate.py run inside the Linux "trainer" container, NOT with
# "poetry run" on the host: mlflow.pyfunc.log_model() records artifact paths with
# the training OS's separator, and Windows paths do not resolve in the Linux API
# container.
train-toy: data-toy ## Train end to end on the toy dataset in seconds (smoke test, no GPU)
	$(TRAINER_RUN) python -m src.train --dataset toy --artifact_dir artifacts_toy --epochs 2 --n_trials 1

train: data-raw ## Train end to end on the full production dataset
	$(TRAINER_RUN) python -m src.train --dataset raw

quality-gate: ## Run the quality gate and promote to production if it passes
	$(TRAINER_RUN) python -m src.quality_gate

# The API loads the model once, at startup (lifespan): after a promotion it must
# restart to serve the new production version.
reload-api: ## Restart the local API so it loads the current "production" model
	docker compose restart api
	docker compose up -d --wait api

# Same entrypoint and image as the Kubernetes CronJob (kubernetes/base/cronjob.yaml),
# against the LocalStack bucket.
batch-local: ## Run batch scoring (src.batch_inference) locally against LocalStack
	cd $(CORE_DIR) && poetry run python -m scripts.make_batch_input --output artifacts_batch/input_data.csv
	docker compose cp $(CORE_DIR)/artifacts_batch/input_data.csv localstack:/tmp/input_data.csv
	docker compose exec -T localstack awslocal s3 cp /tmp/input_data.csv s3://$(LOCAL_BUCKET)/batch/input_data.csv
	docker compose run --rm --no-deps -e AWS_ENDPOINT_URL=http://localstack:4566 \
		-e MODEL_BUCKET_NAME=$(LOCAL_BUCKET) api python -m src.batch_inference
	docker compose exec -T localstack awslocal s3 cp s3://$(LOCAL_BUCKET)/batch/predictions_output.csv - | head -n 3

mlflow-ui: ## Start the local MLflow UI on ./core_ml/mlruns (http://localhost:5000)
	cd $(CORE_DIR) && poetry run mlflow ui --backend-store-uri ./mlruns

# ------------------------------------------------------------------------------
# Local environment (docker-compose): Postgres, LocalStack, MLflow, API
# ------------------------------------------------------------------------------
up: ## Start the whole local stack (Postgres, LocalStack, MLflow, API)
	docker compose up -d --build
	docker compose up -d --wait postgres localstack mlflow api

down: ## Stop and remove the local stack (data volumes are kept)
	docker compose down

restart: down up ## Restart the whole local stack

# LocalStack Community does not persist S3 but Postgres does: after a LocalStack
# restart MLflow would list runs whose artifacts are gone. This wipes both
# volumes back to a consistent state (then retrain).
reset-local: ## Delete the local stack AND its volumes (MLflow + simulated S3) and start clean
	docker compose down -v
	$(MAKE) up

logs: ## Follow the logs of every local service
	docker compose logs -f

ps: ## Show the state of the local services
	docker compose ps

# ------------------------------------------------------------------------------
# API image build/push and Kubernetes deployment: the procedure .gitlab-ci.yml
# runs, for manual use when needed.
# ------------------------------------------------------------------------------
docker-build: ## Build the API image locally
	docker build -t $(ECR_REPOSITORY):$(IMAGE_TAG) -f Dockerfile .

docker-push: ## Push the image to ECR (needs a prior `aws ecr get-login-password` login and AWS_ACCOUNT_ID)
	@if [ -z "$(AWS_ACCOUNT_ID)" ]; then echo "Usage: make docker-push AWS_ACCOUNT_ID=123456789012"; exit 1; fi
	docker tag $(ECR_REPOSITORY):$(IMAGE_TAG) $(ECR_REGISTRY)/$(ECR_REPOSITORY):$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/$(ECR_REPOSITORY):$(IMAGE_TAG)

deploy: ## Apply the Kubernetes manifests (kustomize) to the cluster kubectl points at
	kubectl apply -k kubernetes/overlays/production

# ------------------------------------------------------------------------------
# Terraform / Kubernetes: infrastructure validation and operations
# ------------------------------------------------------------------------------
tf-fmt: ## Check Terraform formatting (no AWS credentials)
	terraform -chdir=terraform fmt -check -diff -recursive

tf-validate: ## Validate Terraform syntax and consistency (local init, no remote backend or AWS)
	terraform -chdir=terraform init -backend=false -input=false
	terraform -chdir=terraform validate

tf-test: tf-validate ## Terraform tests with mocked providers (offline, no AWS credentials)
	terraform -chdir=terraform test

tf-plan: ## Show what Terraform would change (needs AWS credentials)
	terraform -chdir=terraform plan

tf-apply: ## Apply the Terraform changes to the real environment (needs AWS credentials)
	terraform -chdir=terraform apply

k8s-build: ## Render the production overlay (kustomize, no real cluster)
	kubectl kustomize kubernetes/overlays/production

# ------------------------------------------------------------------------------
# Git hooks
# ------------------------------------------------------------------------------
hooks: ## Install the pre-commit git hooks
	pre-commit install --hook-type pre-commit

pre-commit-run: ## Run every pre-commit hook over the whole repo
	pre-commit run --all-files

# ------------------------------------------------------------------------------
# Aggregate: the same target CI runs (.gitlab-ci.yml::lint_test)
# ------------------------------------------------------------------------------
ci: format-check lint test ## Run the full validation (identical to CI)

clean: ## Remove local tool caches and artifacts
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(API_DIR)/.coverage $(CORE_DIR)/.coverage
