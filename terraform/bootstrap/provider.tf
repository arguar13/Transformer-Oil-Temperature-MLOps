terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  # Local state on purpose: this module creates the remote backend (S3 bucket +
  # DynamoDB table) the rest of the project uses (../provider.tf), so it cannot
  # depend on it. It runs once, with admin credentials, and its git-ignored
  # .tfstate stays on this machine as the record of the bootstrap resources.
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
