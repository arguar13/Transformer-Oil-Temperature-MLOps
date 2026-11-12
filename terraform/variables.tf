variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name, used for tagging"
  type        = string
  default     = "dlinear-forecast-cluster"
}

variable "db_username" {
  description = "RDS admin user (MLflow backend store)"
  type        = string
  default     = "mlopsadmin"
}

# No default on purpose: a default password would be a secret committed to git.
# Pass it with -var="db_password=..." or TF_VAR_db_password, never hardcoded.
variable "db_password" {
  description = "Password of the RDS admin user (MLflow backend store)"
  type        = string
  sensitive   = true
}

variable "db_storage_encrypted" {
  description = "Encryption at rest for MLflow's RDS. Only takes effect when the instance is CREATED (see lifecycle.ignore_changes in rds.tf)"
  type        = bool
  default     = true
}

variable "gitlab_project_path" {
  description = "GitLab project path (namespace/project) used in GitLabCI_OIDC_Role's OIDC trust policy"
  type        = string
  default     = "personal-group7745334/612-forecasting-oil-temperature-mlops"
}
