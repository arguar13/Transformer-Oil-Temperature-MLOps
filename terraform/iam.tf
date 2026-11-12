# The project's IAM, in two parts:
#
#   - dlinear_api_s3_read: an S3 policy attached to the EKS node group's role
#     (eks.tf); every pod inherits it from its node instead of an IAM role per
#     ServiceAccount (IRSA).
#   - gitlab_ci_role: the role GitLab CI assumes through OIDC (no long-lived
#     credentials) to push the image to ECR, read/write the artifacts bucket
#     (DVC remote included) and deploy to the cluster.

# Least privilege: read on this project's artifacts bucket only (not an
# account-wide AmazonS3ReadOnlyAccess), plus write on the batch/ prefix, where the
# batch-inference CronJob stores its output. Scoped to that prefix so the model
# artifacts (mlflow/) and the DVC store (dvc-store/) cannot be overwritten by
# mistake.
resource "aws_iam_policy" "dlinear_api_s3_read" {
  name = "dlinear-api-s3-read-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.model_artifacts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/batch/*"
      }
    ]
  })
}

# ============================================================
# OIDC federation with GitLab.com: GitLab CI assumes an AWS role with temporary
# credentials (sts:AssumeRoleWithWebIdentity) instead of a long-lived access key
# stored as a CI variable.
# ============================================================
data "tls_certificate" "gitlab" {
  url = "https://gitlab.com"
}

resource "aws_iam_openid_connect_provider" "gitlab" {
  url             = "https://gitlab.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.gitlab.certificates[0].sha1_fingerprint]
}

# The exact name .gitlab-ci.yml uses in `sts assume-role-with-web-identity`.
resource "aws_iam_role" "gitlab_ci_role" {
  name = "GitLabCI_OIDC_Role"

  # `sub` is pinned to the exact project and branch that deploy: no other
  # GitLab.com project or branch can assume this role.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.gitlab.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "gitlab.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "gitlab.com:sub" = "project_path:${var.gitlab_project_path}:ref_type:branch:ref:main"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "gitlab_ci_ecr" {
  name = "ecr-push-pull-dlinear-repo"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = aws_ecr_repository.dlinear_api.arn
      }
    ]
  })
}

# Least privilege: S3 limited to this project's bucket (not AmazonS3FullAccess).
# Covers both the MLflow model artifacts and the DVC remote (same bucket,
# dvc-store/ prefix, see .dvc/config).
resource "aws_iam_role_policy" "gitlab_ci_s3" {
  name = "s3-rw-model-artifacts-bucket"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.model_artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
      }
    ]
  })
}

# The deploy job (.gitlab-ci.yml::deploy) must describe the cluster to build a
# kubeconfig (`aws eks update-kubeconfig`). Permission to actually operate on
# Kubernetes resources (RBAC) is granted separately by mapping this role in the
# cluster's aws-auth ConfigMap, a manual one-time step per cluster (README.md,
# "Deploying to AWS").
resource "aws_iam_role_policy" "gitlab_ci_eks_describe" {
  name = "eks-describe-cluster"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster"]
        Resource = module.eks.cluster_arn
      }
    ]
  })
}
