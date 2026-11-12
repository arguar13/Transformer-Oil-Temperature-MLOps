variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = "S3 bucket for the remote Terraform state (must match backend \"s3\" in ../provider.tf)"
  type        = string
  default     = "dlinear-terraform-state-prod"
}

variable "lock_table_name" {
  description = "DynamoDB table for state locking (must match dynamodb_table in ../provider.tf)"
  type        = string
  default     = "terraform-state-locks"
}

variable "db_username" {
  description = "RDS admin user (must match db_username in ../variables.tf)"
  type        = string
  default     = "mlopsadmin"
}

variable "operator_user_name" {
  description = "IAM operator user that runs the first terraform apply and later manual operations"
  type        = string
  default     = "dlinear-mlops-admin"
}
