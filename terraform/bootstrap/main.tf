# ==============================================================================
# Bootstrap
#
# Creates, with one-off admin credentials, what the rest of the project
# (../*.tf) needs *before* its own `terraform init`: the remote backend (S3 +
# DynamoDB), the RDS password and an IAM operator user, all as code, with no
# manual CLI or console steps.
# ==============================================================================

# ------------------------------------------------------------------------------
# Remote backend of the main state (../provider.tf::backend "s3")
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "tf_state" {
  bucket = var.state_bucket_name

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tf_locks" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ------------------------------------------------------------------------------
# RDS password, generated here once and passed by hand as var.db_password to the
# main stack (../rds.tf) and as the value of the `mlflow-db-credentials`
# Kubernetes Secret (kubernetes/base/mlflow.yaml). No Secrets Manager: one
# sensitive value, generated once, copied to the two places that use it.
# ------------------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # safe inside a PostgreSQL connection URI
}

# ------------------------------------------------------------------------------
# IAM operator user: runs the main stack's first `terraform apply` (before the
# GitLab OIDC role exists) and later manual kubectl/AWS CLI work. Created by
# Terraform, not in the console.
# ------------------------------------------------------------------------------
resource "aws_iam_user" "operator" {
  name = var.operator_user_name
  tags = {
    Purpose = "dlinear-mlops-bootstrap-operator"
  }
}

resource "aws_iam_access_key" "operator" {
  user = aws_iam_user.operator.name
}

resource "aws_iam_user_policy_attachment" "operator_admin" {
  user       = aws_iam_user.operator.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

output "state_bucket_name" {
  value = aws_s3_bucket.tf_state.bucket
}

output "lock_table_name" {
  value = aws_dynamodb_table.tf_locks.name
}

output "db_password" {
  value     = random_password.db_password.result
  sensitive = true
}

output "operator_access_key_id" {
  value = aws_iam_access_key.operator.id
}

output "operator_secret_access_key" {
  value     = aws_iam_access_key.operator.secret
  sensitive = true
}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}
