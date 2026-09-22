#!/usr/bin/env bash
# Bootstrap an AWS EC2 g5.xlarge for QatabasisStack multi-cloud validation.
# Uses the same Docker path as Lambda Cloud (make build / make run), via
# scripts/cloud_bootstrap.sh, so both clouds run identical, reproducible
# infrastructure rather than two different install methods per provider.
# Design + rationale: docs/AWS_DEPLOYMENT.md (qatabasis-internal repo).
#
# Usage:
#   AWS_KEY=~/.ssh/aws-vqe.pem AWS_KEY_NAME=aws-vqe AWS_SG=sg-xxxxxxxx \
#       scripts/aws_deploy.sh
#
# Env vars (all required unless a default is shown):
#   AWS_KEY        Local path to the .pem key (chmod 400).
#   AWS_KEY_NAME   The EC2 key pair name (no path, no extension).
#   AWS_SG         Security group id allowing SSH from your IP.
#   AWS_REGION     Default: us-east-1.
#   AWS_AMI        Default: latest DL AMI Ubuntu 22.04 in the region.
#   INSTANCE_TYPE  Default: g5.xlarge.
#
# Cleanup is manual (see docs/AWS_DEPLOYMENT.md — deliberate).

set -eo pipefail   # NOT -u; conda activate scripts hit unset vars

: "${AWS_KEY:?set AWS_KEY to the .pem path}"
: "${AWS_KEY_NAME:?set AWS_KEY_NAME to the EC2 key pair name}"
: "${AWS_SG:?set AWS_SG to a security group id allowing SSH from your IP}"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"

if [[ -z "${AWS_AMI:-}" ]]; then
    echo "[deploy] resolving latest GPU-capable Ubuntu DL AMI in ${AWS_REGION}..."
    # AWS renames/reversions this family often (e.g. dropped Ubuntu 22.04 for
    # 24.04/26.04, retitled "GPU PyTorch" -> "OSS Nvidia Driver GPU"). Match
    # loosely on "Deep Learning...GPU...Ubuntu" rather than an exact string so
    # this doesn't silently break again on the next AWS rename -- this stack
    # builds its own Docker image via `make build`, so the AMI only needs a
    # working NVIDIA driver + Ubuntu base, not a specific preinstalled
    # framework version.
    AWS_AMI=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning*GPU*Ubuntu*" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
        --output text)
    AWS_AMI_NAME=$(echo "$AWS_AMI" | cut -f2-)
    AWS_AMI=$(echo "$AWS_AMI" | cut -f1)
    if [[ -z "$AWS_AMI" || "$AWS_AMI" == "None" ]]; then
        echo "[deploy] ERROR: no matching AMI found. AWS may have renamed this" >&2
        echo "         family again -- check available images with:" >&2
        echo "         aws ec2 describe-images --region $AWS_REGION --owners amazon \\" >&2
        echo "             --filters 'Name=name,Values=Deep Learning*GPU*' 'Name=state,Values=available' \\" >&2
        echo "             --query 'sort_by(Images, &CreationDate)[-10:].[ImageId,Name]' --output table" >&2
        exit 1
    fi
    echo "[deploy] AMI: $AWS_AMI ($AWS_AMI_NAME)"
fi

echo "[deploy] launching ${INSTANCE_TYPE} in ${AWS_REGION}..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AWS_AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$AWS_KEY_NAME" \
    --security-group-ids "$AWS_SG" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hpchybrid-validation}]' \
    --query 'Instances[0].InstanceId' \
    --output text)
echo "[deploy] instance $INSTANCE_ID launching..."

aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)
echo "[deploy] running at ${PUBLIC_IP}"

echo "[deploy] waiting for SSH..."
until ssh -i "$AWS_KEY" -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
          ubuntu@"$PUBLIC_IP" true 2>/dev/null; do
    sleep 5
done
echo "[deploy] SSH up."

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "[deploy] uploading repo from $REPO_ROOT..."
rsync -az \
    --exclude=.git \
    --exclude=results \
    --exclude=__pycache__ \
    --exclude=.pubchem_cache \
    --exclude=build \
    --exclude='*.pyc' \
    -e "ssh -i $AWS_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$REPO_ROOT/" ubuntu@"$PUBLIC_IP":~/qatabasis/

echo "[deploy] running cloud_bootstrap.sh on the instance (docker group + GPU-in-Docker check)..."
ssh -i "$AWS_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && bash scripts/cloud_bootstrap.sh"

echo "[deploy] building the Docker image (same path used on Lambda -- keeps"
echo "         AWS and Lambda runs on identical, reproducible infrastructure)..."
echo "[deploy] NOTE: if cloud_bootstrap.sh just added this user to the docker"
echo "         group for the first time, this SSH command runs in a NEW"
echo "         connection (not the bootstrap's own shell), so group membership"
echo "         is already active here -- no separate re-login needed."
ssh -i "$AWS_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && make build"

echo "[deploy] running smoke test (make pytest)..."
ssh -i "$AWS_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && make pytest" \
    || echo "[deploy] WARNING: pytest did not exit 0 — investigate before running the workload."

cat <<EOF

[deploy] ready. instance: $INSTANCE_ID at $PUBLIC_IP

Follow-up:
  ssh -i $AWS_KEY ubuntu@$PUBLIC_IP
  cd ~/qatabasis
  make run NP=2 MOLECULES="H2 LiH BeH2 H2O"

Before terminating (do this every time):
  rsync -av -e "ssh -i $AWS_KEY" \\
      ubuntu@$PUBLIC_IP:~/qatabasis/results/ \\
      results/

Terminate when done:
  aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID
EOF
