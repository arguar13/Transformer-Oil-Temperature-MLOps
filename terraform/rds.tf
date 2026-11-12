# RDS security group: PostgreSQL from the EKS nodes only
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Allow PostgreSQL traffic exclusively from EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# Single RDS instance for MLflow's backend store, with no Multi-AZ or read
# replicas: Multi-AZ doubles the cost for availability this project does not need.
#
# The password arrives as a Terraform variable (var.db_password, no default, see
# variables.tf) rather than from Secrets Manager: fewer moving parts. Pods never
# read it from Terraform or AWS; it is created once, by hand, as a plain
# Kubernetes Secret (kubernetes/base/mlflow.yaml).
resource "aws_db_instance" "mlflow_db" {
  identifier        = "${var.project_name}-backend-db"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  engine            = "postgres"
  engine_version    = "18.3"
  # Without `db_name` RDS creates no initial database, and
  # `mlflow server --backend-store-uri postgresql://.../mlflow_db` fails with
  # "database does not exist". Same name as docker-compose.yml's POSTGRES_DB, so
  # one `--backend-store-uri` shape works locally and in production.
  db_name                = "mlflow_db"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = true
  multi_az               = false
  # Encryption at rest (AWS-managed KMS key, no extra cost): the backend store
  # holds run parameters, metrics and metadata.
  storage_encrypted = var.db_storage_encrypted

  lifecycle {
    # RDS cannot change storage_encrypted in place: it forces a replacement, and
    # with skip_final_snapshot = true that wipes MLflow's backend with no
    # snapshot. Ignored after creation: a new instance is born encrypted, and an
    # existing unencrypted one is NOT replaced. To encrypt an existing one
    # without loss: snapshot -> encrypted copy -> restore from the copy (README,
    # "Deploying to AWS").
    ignore_changes = [storage_encrypted]
  }
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}
