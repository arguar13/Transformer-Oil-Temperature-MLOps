module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name = "${var.project_name}-eks"
  # 1.36 is in standard support; older versions lose their managed AMIs once
  # extended support ends.
  cluster_version = "1.36"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS does not install metrics-server by default (unlike coredns, kube-proxy and
  # vpc-cni). Without it the HPA (kubernetes/base/hpa.yaml) cannot read CPU
  # metrics and stays "Degraded"; installed as an AWS-managed add-on so the
  # whole cluster is reproducible from Terraform.
  cluster_addons = {
    metrics-server = {
      most_recent = true
    }
  }

  # The module opens the node security group to the control plane for the usual
  # webhook and kubelet ports, but not for metrics-server's 10251: without this
  # rule the metrics.k8s.io API is unreachable.
  node_security_group_additional_rules = {
    ingress_cluster_to_node_metrics_server = {
      description                   = "Cluster API to node metrics-server webhook"
      protocol                      = "tcp"
      from_port                     = 10251
      to_port                       = 10251
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }

  # One node group: the HPA (kubernetes/base/hpa.yaml) scales pods within this
  # node range, with no need for a specialised second group.
  eks_managed_node_groups = {
    inference = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      # t3.medium hit MemoryPressure with MLflow plus several API replicas
      # (each with PyTorch loaded); t3.large leaves enough headroom.
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      # With a custom launch template (this case) the submodule silently ignores
      # "disk_size"; block_device_mappings is honoured, and avoids running out of
      # space pulling the PyTorch image on the AMI's default 20 GB root volume.
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 50
            volume_type           = "gp3"
            delete_on_termination = true
          }
        }
      }

      # The S3 permission is attached to the node group's role, so every pod
      # inherits it from its node (which is why kubernetes/base/serviceaccount.yaml
      # has no IRSA annotation).
      iam_role_additional_policies = {
        s3_access = aws_iam_policy.dlinear_api_s3_read.arn
      }
    }
  }

  tags = {
    Environment = "production"
  }
}
