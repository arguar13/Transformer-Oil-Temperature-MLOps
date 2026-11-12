terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    # For data "tls_certificate" (iam.tf, GitLab OIDC thumbprint).
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Remote state in S3 with DynamoDB locking; terraform/bootstrap creates both.
  backend "s3" {
    bucket         = "dlinear-terraform-state-prod"
    key            = "mlops/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
