# A single repository: the API image, which serves both the HTTP endpoint and
# the batch-inference CronJob (see Dockerfile). MLflow runs the official
# ghcr.io/mlflow/mlflow image (kubernetes/base/mlflow.yaml), so nothing of it is
# pushed to ECR.

resource "aws_ecr_repository" "dlinear_api" {
  name                 = "dlinear-forecast-api"
  image_tag_mutability = "IMMUTABLE"

  # Otherwise `terraform destroy` fails with RepositoryNotEmptyException as soon
  # as one image has been pushed.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep the last 10 images (for rollback) and expire the rest.
resource "aws_ecr_lifecycle_policy" "cleanup_policy" {
  repository = aws_ecr_repository.dlinear_api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 images (rollback); expire the rest"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

output "ecr_repository_url" {
  value = aws_ecr_repository.dlinear_api.repository_url
}
