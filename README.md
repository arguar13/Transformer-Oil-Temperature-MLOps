# Transformer Oil Temperature Forecasting: MLOps Project

*Read this in other languages: [Español](README_es.md)*

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![CI/CD: GitLab](https://img.shields.io/badge/CI%2FCD-GitLab-fc6d26?logo=gitlab&logoColor=white)](https://about.gitlab.com/)
[![Dependency management: Poetry](https://img.shields.io/badge/dependencies-poetry-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![Container: Docker](https://img.shields.io/badge/container-docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Orchestration: Kubernetes](https://img.shields.io/badge/orchestration-kubernetes-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Cloud: AWS](https://img.shields.io/badge/cloud-AWS-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![IaC: Terraform](https://img.shields.io/badge/IaC-terraform-844FBA?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Tracking: MLflow](https://img.shields.io/badge/tracking-mlflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

## What this project is

An end-to-end MLOps platform that trains and serves a **DLinear** model to
predict a power transformer's oil temperature **48 hours ahead**, from its
own load and sensor readings. The dataset is
[ETTh1](https://github.com/zhouhaoyi/ETDataset) (Electricity Transformer
Temperature), a widely used public benchmark for long-horizon time series
forecasting.

Oil temperature is a leading indicator of transformer health: it tracks
electrical load and ambient conditions, and sustained deviations from
expected behavior are an early signal of overload or degradation. A model
that forecasts it two days out gives operators a window to act before a
threshold is crossed, instead of reacting after the fact.

**Why DLinear.** ["Are Transformers Effective for Time Series
Forecasting?"](https://arxiv.org/abs/2205.13504) showed that a single linear
layer per output channel matches or beats considerably more elaborate
architectures on exactly this kind of benchmark (ETT included). A model
that trains in seconds on a laptop keeps the focus on the problems every
forecasting system has to solve regardless of model choice: reproducible
training, experiment tracking, a promotion gate before a model reaches
production, a served API, batch scoring, and a deployment path to
Kubernetes.

## Results

Measured once, on 3,389 held-out test windows (1 February to 26 June
2018), against two naive baselines scored on the same windows.

| Method | MAE (°C) | RMSE (°C) | WAPE |
|---|---|---|---|
| **DLinear** | **1.720** | **2.333** | **22.0%** |
| Persistence (repeat the last reading) | 1.825 | 2.423 | 23.4% |
| Seasonal naive (repeat the last 24 hours) | 2.047 | 2.661 | 26.2% |

DLinear removes **5.8%** of the persistence error and **16.0%** of the
seasonal naive error. The gain is not uniform across the horizon: the model
beats persistence at 42 of the 48 hours ahead, peaking at +14.4% around
+10 hours, but loses over the first three hours (0.701 °C against 0.450 °C
at +1 hour) and, marginally, at +23 to +25 hours. Oil temperature has a lot
of inertia, so the last reading is hard to beat a few hours out; the model
earns its place on the longer part of the horizon.

## Architecture

```mermaid
flowchart TB
    subgraph dev["Local development"]
        DVC["DVC\n(versioned dataset)"]
        Trainer["core_ml (train.py)"]
        MLflowLocal["MLflow (docker-compose)"]
        LocalStack["LocalStack\n(fake S3)"]
        DVC --> Trainer
        Trainer -- logs runs/metrics --> MLflowLocal
        MLflowLocal -- artifacts --> LocalStack
    end

    subgraph ci["GitLab CI"]
        LintTest["lint_test"]
        BuildPush["build_push"]
        Deploy["deploy"]
        Train["train (manual)"]
        LintTest --> BuildPush --> Deploy
    end

    subgraph aws["AWS"]
        ECR["ECR\n(API image)"]
        subgraph eks["EKS cluster"]
            API["dlinear-api\n(FastAPI, Deployment + HPA)"]
            CronJob["dlinear-batch-inference\n(CronJob)"]
            MLflowProd["mlflow\n(Deployment)"]
            API -. resolves model .-> MLflowProd
            CronJob -. resolves model .-> MLflowProd
        end
        RDS["RDS Postgres\n(MLflow backend store)"]
        S3["S3\n(model artifacts + DVC remote)"]
        MLflowProd --> RDS
        MLflowProd --> S3
        CronJob --> S3
    end

    Trainer -. dvc push / mlflow .-> S3
    BuildPush --> ECR
    Deploy --> eks
    ECR --> API
    ECR --> CronJob
    Train -. python -m src.train + quality_gate .-> MLflowProd
```

Every major component maps to a specific design decision:

- **The model never travels as a loose `.pth`/`.pkl` file.** `train.py`
  packages the DLinear network with its input/output scalers into a single
  `mlflow.pyfunc` artifact (`core_ml/src/mlflow_utils.py::DLinearForecaster`)
  and registers it in the MLflow Model Registry. Both the API and the batch
  job load it the same way, by name and alias
  (`models:/dlinear-ett-forecaster@production`): an immutable, versioned
  reference instead of a path on disk that could drift out of sync with
  the code that produced it.
- **A quality gate stands between "trained" and "serving traffic."**
  `core_ml/src/quality_gate.py` first requires the candidate to beat a
  naive persistence baseline (repeat the last observed reading) on the same
  held-out test windows, then compares its test MSE against whatever is
  currently aliased `production`, and only moves the alias if the candidate
  is at least as good: promotion is a decision the pipeline enforces, not
  a side effect of training.
- **One container image, two entry points.** The image built by CI
  (`Dockerfile`) serves `/predict` through `uvicorn` and runs
  `python -m src.batch_inference` as a Kubernetes CronJob. Both paths load
  the model through the same registry reference and the same
  `DLinearForecaster.predict()` code: one inference implementation, not
  two that can quietly diverge.
- **The API treats the model registry as a dependency that can fail.**
  Model loading goes through a bounded exponential-backoff retry wrapped in
  a circuit breaker (`pybreaker`): if the registry is unreachable, the pod
  stays alive and reports itself unready (`/health` vs `/ready`) rather
  than crash-looping or serving stale results.
- **Drift detection is a statistical check, not a subsystem.**
  `core_ml/src/monitoring/drift_check.py` flags a sensor when a batch's
  mean falls outside the p1–p99 band of the rolling weekly means seen
  during training (i.e. when the model would be extrapolating) and logs
  the result: no external queue, no persisted history, no automatic
  action; deciding what to do about a drift signal is a call for a human,
  not a heuristic.
- **Infrastructure is provisioned once, applied continuously.**
  `terraform/` owns the VPC, EKS cluster, RDS instance, S3 bucket, ECR
  repository and IAM roles as versioned state; `.gitlab-ci.yml` only
  touches the application layer (`kubectl apply -k`) on top of that fixed
  infrastructure, so the two change on different cadences.
- **CI authenticates to AWS with short-lived, scoped credentials.** The
  GitLab OIDC identity federates into a dedicated IAM role
  (`GitLabCI_OIDC_Role`, `terraform/iam.tf`) restricted to this project's
  `main` branch, assumed for a one-hour session; no long-lived AWS access
  key stored as a CI variable to rotate or leak.

## Repository structure

```
.
├── api/                       # FastAPI inference service
│   ├── main.py
│   └── tests/
├── core_ml/                   # Data, training, MLflow, quality gate
│   ├── src/
│   │   ├── data_processing.py
│   │   ├── train.py
│   │   ├── model_architecture.py   # DLinear (PyTorch)
│   │   ├── mlflow_utils.py
│   │   ├── quality_gate.py
│   │   ├── batch_inference.py
│   │   └── monitoring/
│   │       └── drift_check.py      # drift check (training weekly-mean envelope)
│   ├── data/                  # DVC-tracked datasets (toy + raw)
│   └── tests/
├── terraform/                 # AWS infrastructure (VPC, EKS, RDS, S3, ECR, IAM)
├── kubernetes/
│   ├── base/                  # Deployment, Service, HPA, CronJob, MLflow, namespace
│   └── overlays/production/   # per-environment values (image tag, bucket, RDS endpoint)
├── docker-compose.yml         # local stack: Postgres, LocalStack, MLflow, API
├── Dockerfile                 # API + batch inference image
├── core_ml/train.Dockerfile   # training image
└── .gitlab-ci.yml
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Model | PyTorch, DLinear | Strong, fast baseline for long-horizon forecasting; keeps the focus on the platform |
| Hyperparameter search | Optuna | Learning-rate search over a validation proxy, logged as part of the run |
| Experiment tracking & registry | MLflow (Postgres backend, S3 artifact store) | Single source of truth for run history and which model version is `production` |
| Data versioning | DVC (S3/LocalStack remote) | Every training run is reproducible from a content hash, not just a filename |
| Serving | FastAPI + Uvicorn | Small, typed, async-friendly HTTP surface for `/predict` |
| Resilience | `tenacity` (retry), `pybreaker` (circuit breaker) | Bounded retries and fail-fast behavior against a registry dependency |
| Containerization | Docker | One image for the API and the batch CronJob |
| Orchestration | Kubernetes (EKS), Kustomize | Declarative manifests, per-environment overlays, no templating engine needed at this scale |
| Infrastructure as Code | Terraform (AWS provider + community modules) | VPC, EKS, RDS, S3, ECR, IAM as versioned, reviewable state |
| CI/CD | GitLab CI, OIDC to AWS | Lint/test, build & push, deploy; short-lived cloud credentials, no static keys |
| Dependency management | Poetry (`api/`, `core_ml/` as separate projects) | Deterministic installs from a committed lockfile |
| Code quality | Ruff, mypy, pytest, pre-commit | Fast lint/format, optional typing, unit tests with mocks |

## Running it locally

Requirements: Docker + Docker Compose, Python 3.10, [Poetry](https://python-poetry.org/).

```bash
# Install dependencies for both Python subprojects
make install

# Bring up Postgres, LocalStack (fake S3), MLflow and the API
make up

# Pull the DVC-tracked datasets (toy + raw), first time only,
# or point .dvc/config.local at LocalStack (see .dvc/config.local.example)
cp .dvc/config.local.example .dvc/config.local
make dvc-pull

# No DVC remote reachable (e.g. a fresh clone: the LocalStack bucket starts
# empty)? Download ETTh1 from the public ETDataset repo and build the toy set
make data-download
```

`make data-download` fetches the same data, but not byte-identical to the
DVC-tracked copy (that one was re-saved with pandas), so training runs
tag `data_matches_dvc_pointer=false` in MLflow: the DVC hash recorded
then describes the pointer, not the bytes actually used.

`docker-compose.yml` stands in for AWS end to end: LocalStack simulates S3
(model artifacts and the DVC remote), Postgres simulates RDS, and MLflow
and the API run the same way they do in production, against the same
`MLFLOW_TRACKING_URI`/`MODEL_NAME`/`MODEL_ALIAS` contract.

Once the stack is up:

- API: http://localhost:8000/docs (or the port from
  `docker-compose.override.yml`, if present)
- MLflow UI: http://localhost:5000

`make down` stops the stack (data volumes are preserved). `make logs` /
`make ps` follow logs and check the state of each service. Note that
LocalStack Community does not persist S3 state (not even across a plain
container restart) while Postgres does: afterwards MLflow still lists
runs whose artifacts are gone. `make reset-local` wipes both volumes and
brings the stack back up clean; then retrain.

## Training a model

```bash
# Fast smoke test on the ~1,000-row toy dataset (seconds, no GPU needed)
make train-toy

# Full training run on the complete ETTh1 dataset
make train

# Compare the newly trained model against the current "production" one,
# and promote it if it's at least as good
make quality-gate

# The API loads the model once, at startup: restart it to serve the newly
# promoted version
make reload-api

# Run the batch scoring job (same image and entrypoint as the Kubernetes
# CronJob) against LocalStack
make batch-local
```

The API stays alive but answers `/ready` and `/predict` with 503 until a
`production` version exists and it has been (re)loaded.

Training runs inside the `trainer` container (`core_ml/train.Dockerfile`)
so that file paths recorded by MLflow are consistent regardless of the
host OS: a model trained directly on Windows would record artifact paths
with backslashes that the Linux serving container can't resolve. Every run
is seeded for reproducibility and logs to MLflow: hyperparameters, test
metrics for both the model and the persistence baseline
(`final_test_mse`, `final_test_mae`, `final_test_mse_persistence`,
`final_test_mae_skill`, ...), the model + its scalers as a single
versioned artifact, and a reference profile (per-sensor weekly-mean
envelope) consumed later by the drift check.

The pipeline keeps three splits strictly separated: train fits the
weights, validation drives the Optuna learning-rate search and early
stopping, and test is touched exactly once, at the very end, to produce
the metric the quality gate actually decides on, so that metric reflects
genuine generalization, not a number the selection process already
optimized toward.

`core_ml/src/quality_gate.py` is the promotion gate. Its first check is
against a naive baseline: a candidate trained on real data must have a
lower `final_test_mse` than `final_test_mse_persistence`, the error of
simply repeating the last observed oil temperature for the whole horizon,
measured on the same test windows. Oil temperature has a lot of inertia,
so persistence is hard to beat a few hours ahead; a model that does not
beat it does not justify the cost of running it, not even as the first
version. It then compares the candidate's `final_test_mse` against
whatever is currently aliased `production` in the MLflow Model Registry,
and only moves the `production` alias if the new version is at least as
good. If there is no `production` version yet, the first candidate that
beats persistence becomes the baseline. A worse model never gets promoted, and the script exits non-zero
when it rejects a candidate, so a CI pipeline built around it would stop
before deploying a regression.

The comparison only happens between comparable runs: both versions must
share `dataset`, `seq_len` and `pred_len`, otherwise their test MSEs
measure different things and the candidate is rejected (promote by hand
if the configuration change is intentional). A `toy` smoke-test model can
never replace a model trained on real data, while a real model always
replaces a `toy` baseline. The `toy` run is also exempt from the
persistence rule: its ~100 test windows exist to exercise the plumbing
end to end, not to measure skill.

## Batch inference and drift monitoring

`core_ml/src/batch_inference.py` runs as a daily Kubernetes CronJob
(`kubernetes/base/cronjob.yaml`, 02:00 UTC). Each run:

1. Loads the current `production` model from the MLflow Model Registry,
the exact same artifact and code path the online API uses.
2. Downloads the input batch from S3 and validates it against the same
   data contract (`core_ml/src/data_contracts.py`) the training pipeline
   enforces, failing fast before spending any compute on inference.
3. Builds overlapping sliding windows and scores them in a single batched,
   vectorized pass through the model.
4. Uploads the predictions back to S3.
5. Runs the drift check against the batch it just scored and logs the
   result.

The drift check itself (`core_ml/src/monitoring/drift_check.py`) is
intentionally a small, self-contained statistical test. The readings are
strongly seasonal and autocorrelated: a batch covers days while the
training set covers seasons, so testing a batch against the whole training
distribution would alert on almost every normal week. Instead, the
reference profile stores, per sensor, the p1–p99 band of the rolling
weekly means over the training period (the same time scale as a batch)
and a sensor is flagged when the batch mean falls outside it, meaning the
model is being asked to extrapolate. It carries no state between runs and
takes no action beyond logging, enough to answer "does
this batch still resemble what the model was trained on?", with a human
deciding what to do about a positive signal. The CronJob's own
`backoffLimit` and `activeDeadlineSeconds` keep a persistently failing run
from retrying forever.

## Deploying to AWS

> **Deployed on a real AWS account.** This stack has been deployed end to end on AWS: Terraform provisioned the infrastructure, the image was pushed to ECR and the service ran on EKS, with MLflow backed by RDS and S3. It was then torn down with `terraform destroy` to avoid ongoing costs (EKS, RDS and the NAT Gateway bill by the hour), so there is no live endpoint. The steps below are the ones used for that deployment.

This is a two-step process: provision the infrastructure with Terraform,
then deploy the application with `kubectl`/`kustomize`.

```bash
# 1. Infrastructure (one-time / whenever terraform/ changes)
cd terraform
terraform init
terraform plan -var="db_password=<a strong password>"
terraform apply -var="db_password=<the same password>"
```

This creates the VPC (two AZs, one NAT Gateway), the EKS cluster (a single
managed node group, `metrics-server` installed as a cluster add-on so the
HPA has CPU metrics to scale on), the RDS Postgres instance backing
MLflow, the S3 bucket used for both model artifacts and the DVC remote,
the ECR repository, and the IAM roles the pipeline needs, including the
OIDC trust relationship that lets GitLab CI assume an AWS role without a
stored access key. `db_password` has no default on purpose; never commit
a database password to Git.

The RDS instance is created with `storage_encrypted = true`. RDS cannot
toggle encryption in place (changing it destroys and recreates the
instance), so `terraform/rds.tf` ignores that attribute after creation: an
instance created unencrypted before this setting existed is **not**
replaced. To encrypt such an instance without losing MLflow's backend
store: take a snapshot, copy it with encryption enabled, and restore the
instance from the encrypted copy. `make tf-test` checks this contract
offline, with mocked providers and no AWS credentials.

```bash
# 2. One-time manual setup per cluster (not managed by Terraform or Git):
#    - a Kubernetes Secret with the RDS credentials MLflow uses
#    - map GitLabCI_OIDC_Role into the cluster's aws-auth ConfigMap so CI
#      can run kubectl
kubectl create namespace dlinear-production
kubectl create secret generic mlflow-db-credentials \
  --namespace dlinear-production \
  --from-literal=username=<same as var.db_username> \
  --from-literal=password=<same as var.db_password>

# 3. Deploy the application. The public repo keeps placeholders for the
#    account-specific values (REDACTED_ACCOUNT_ID, REDACTED-RDS-ENDPOINT in
#    kubernetes/overlays/production and .dvc/config). CI fills them from the
#    protected variables AWS_ACCOUNT_ID and RDS_ENDPOINT; by hand, the same
#    `kustomize edit set configmap/image` commands as the `deploy` job.
make deploy   # kubectl apply -k kubernetes/overlays/production
```

From there, pushing to `main` runs the CI pipeline: lint/test → build &
push the image to ECR → `kubectl apply -k` the new image tag, with
Kustomize's `images:` transformer rewriting the tag structurally instead
of a `sed` pass or `kubectl set image`. Training in CI is a separate,
manually triggered job; it needs network access to MLflow's in-cluster
Service, which a shared GitLab.com runner doesn't have by default; `make
train` runs the same pipeline locally against the docker-compose MLflow
instance for day-to-day iteration.

## Testing & code quality

```bash
make lint          # ruff check (api/ + core_ml/)
make format        # ruff check --fix + ruff format
make test           # pytest, both subprojects
make type-check     # mypy
make ci             # what CI runs: format-check + lint + test
```

Tests are unit tests using mocks and FastAPI's `TestClient`: no Docker,
no real AWS, no real MLflow server required to run them. `make hooks`
installs a pre-commit hook that runs the same formatting, lint, type and
test checks before every commit (`.pre-commit-config.yaml`), so issues
surface locally before they reach CI.
