# Fully offline Terraform tests: every provider is mocked (mock_provider), so no
# credentials are needed and AWS is never called.
#
#   make tf-test
#
# They pin the encryption contract of MLflow's RDS (rds.tf): a new instance is
# born encrypted, and enabling encryption on an existing unencrypted instance
# does NOT replace it (which would wipe MLflow's backend store).

# Only what the root stack really reads is pinned: the account_id builds the
# bucket name (s3.tf). The community modules, which would read much more, are
# not executed (override_module below).
mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
      arn        = "arn:aws:iam::123456789012:user/mock"
    }
  }
}

# The community modules (VPC, EKS) are replaced by the outputs the rest of the
# stack consumes: these tests are about RDS, not terraform-aws-modules internals.
override_module {
  target = module.vpc
  outputs = {
    vpc_id                     = "vpc-0123456789abcdef0"
    private_subnets            = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
    database_subnet_group_name = "mock-db-subnet-group"
  }
}

override_module {
  target = module.eks
  outputs = {
    node_security_group_id = "sg-0123456789abcdef0"
    cluster_arn            = "arn:aws:eks:us-east-1:123456789012:cluster/mock"
  }
}

mock_provider "tls" {
  mock_data "tls_certificate" {
    defaults = {
      certificates = [{ sha1_fingerprint = "0000000000000000000000000000000000000000" }]
    }
  }
}

mock_provider "kubernetes" {}
mock_provider "cloudinit" {}
mock_provider "time" {}

variables {
  db_password = "mock-password-not-a-secret" # pragma: allowlist secret
}

run "new_instance_is_created_encrypted" {
  command = plan

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == true
    error_message = "A new RDS instance must be created with storage_encrypted = true."
  }
}

# A legacy RDS instance, created before encryption existed in the stack.
run "legacy_unencrypted_instance_exists" {
  command = apply

  variables {
    db_storage_encrypted = false
  }

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == false
    error_message = "The starting scenario must be an unencrypted instance."
  }
}

# On that state the default (true) must not change storage_encrypted: a change
# there would force destroying and recreating the instance.
run "enabling_encryption_does_not_replace_legacy_instance" {
  command = plan

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == false
    error_message = "storage_encrypted changed on an existing instance, which forces its replacement."
  }
}
