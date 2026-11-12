# Two AZs: enough for this cluster and cheaper than three. A single NAT gateway
# gives the private nodes egress for images and packages, at the cost of losing
# it if that AZ goes down.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  # Dedicated RDS subnets: EKS and RDS share the VPC but not the subnets.
  database_subnets             = ["10.0.201.0/24", "10.0.202.0/24"]
  create_database_subnet_group = true

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  tags = {
    Environment = "production"
    Project     = var.project_name
  }
}
