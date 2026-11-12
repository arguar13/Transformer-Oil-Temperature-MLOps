#!/bin/sh
# Runs once LocalStack is ready (mounted at /etc/localstack/init/ready.d/, see
# docker-compose.yml) and creates the S3 bucket the app expects, with no manual
# step on first start.
set -e

BUCKET="mlops-portafolio-proj3-models"

# Idempotent: under `set -e`, `s3 mb` on an existing bucket (state that survived
# a restart) would abort the script.
if awslocal s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1; then
  echo "[localstack-init] Bucket s3://${BUCKET} already exists, keeping it."
else
  echo "[localstack-init] Creating S3 bucket s3://${BUCKET} ..."
  awslocal s3 mb "s3://${BUCKET}"
fi
awslocal s3api put-bucket-versioning --bucket "${BUCKET}" --versioning-configuration Status=Enabled

echo "[localstack-init] Bootstrap complete."
