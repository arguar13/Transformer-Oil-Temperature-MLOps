# A deterministic name (account ID, not a random suffix): a `random_id` would
# need a manual step (copy the real name from `terraform output` into the K8s
# manifests) every time the bucket is recreated. With the account ID the name is
# known in advance to both Terraform and the Kustomize-generated ConfigMap
# (MODEL_BUCKET_NAME in kubernetes/overlays/production/kustomization.yaml).
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-model-artifacts-${data.aws_caller_identity.current.account_id}"

  # Otherwise `terraform destroy` fails with "BucketNotEmpty" once the bucket
  # holds any object (dvc-store/, mlflow/ and batch/ always exist after use).
  # Versioning is on, so force_destroy purges every version, not just the current one.
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "model_artifacts_bucket_name" {
  description = "Deterministic bucket name, referenced by kubernetes/overlays/production/kustomization.yaml (MODEL_BUCKET_NAME)"
  value       = aws_s3_bucket.model_artifacts.bucket
}
