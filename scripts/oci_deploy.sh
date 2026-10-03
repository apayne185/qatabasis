#!/usr/bin/env bash
# Bootstrap an Oracle Cloud Infrastructure (OCI) GPU instance for
# QatabasisStack multi-cloud validation. Uses the same Docker path as
# Lambda/AWS/GCP (make build / make run), via scripts/cloud_bootstrap.sh,
# so every cloud runs identical, reproducible infrastructure rather than a
# different install method per provider. Mirrors scripts/aws_deploy.sh and
# scripts/gcp_deploy.sh's structure.
#
# OCI differs from AWS/GCP in two ways this script has to handle:
#   1. GPU images are Oracle-Linux-only (no Ubuntu GPU image family like
#      AWS/GCP's Deep Learning AMI/image family) -- fine, since this stack's
#      GPU support comes entirely from Docker + NVIDIA Container Toolkit,
#      not anything OS-specific, and Oracle Linux ships both preinstalled.
#   2. No default VCN/subnet exists on a fresh tenancy (unlike AWS's default
#      VPC or GCP's default network) -- this script creates a minimal one
#      (VCN + public subnet + internet gateway + security list allowing SSH)
#      if none exists yet, and reuses it on subsequent runs.
#
# Usage:
#   OCI_COMPARTMENT_ID=ocid1.tenancy.oc1..xxx scripts/oci_deploy.sh
#
# Env vars (all optional except OCI_COMPARTMENT_ID, sensible defaults shown):
#   OCI_COMPARTMENT_ID  Required. Compartment (or tenancy) OCID to launch into.
#   OCI_REGION          Default: region from ~/.oci/config.
#   OCI_AD              Availability domain. Default: first AD in the region.
#   OCI_SHAPE            Default: VM.GPU.A10.1 (single A10, cheapest GPU shape).
#   OCI_IMAGE_ID          Default: latest Oracle-Linux-8.10-Gen2-GPU image.
#   INSTANCE_NAME        Default: qatabasis-validation.
#   OCI_SSH_KEY          Path to a public key for instance access.
#                        Default: ~/.ssh/id_rsa.pub (generated if missing).
#
# Cleanup is manual (see the printed follow-up command) -- deliberate, so a
# rsync-back-results step is never skipped (same rationale as aws/gcp).
#
# STATUS: written 2026-09-29, NOT yet run end-to-end -- this tenancy's
# gpu-a10-count service limit is 0 (fresh account, same as AWS/GCP's
# zero-quota default) and a limit-increase request is pending. Verify this
# script actually works once that clears -- flagging this explicitly rather
# than presenting it as tested.

set -eo pipefail   # NOT -u; conda activate scripts hit unset vars

: "${OCI_COMPARTMENT_ID:?set OCI_COMPARTMENT_ID to a compartment or tenancy OCID}"
OCI_REGION="${OCI_REGION:-$(oci iam region-subscription list --query 'data[0]."region-name"' --raw-output 2>/dev/null)}"
OCI_SHAPE="${OCI_SHAPE:-VM.GPU.A10.1}"
INSTANCE_NAME="${INSTANCE_NAME:-qatabasis-validation}"
OCI_SSH_KEY="${OCI_SSH_KEY:-$HOME/.ssh/id_rsa.pub}"

if [[ -z "$OCI_REGION" ]]; then
    echo "[deploy] ERROR: could not determine OCI region. Set OCI_REGION explicitly." >&2
    exit 1
fi

if [[ ! -f "$OCI_SSH_KEY" ]]; then
    echo "[deploy] no SSH key at $OCI_SSH_KEY -- generating one..."
    ssh-keygen -t rsa -b 4096 -f "${OCI_SSH_KEY%.pub}" -N "" -q
fi

if [[ -z "${OCI_AD:-}" ]]; then
    echo "[deploy] resolving first availability domain in ${OCI_REGION}..."
    OCI_AD=$(oci iam availability-domain list \
        --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
        --query 'data[0].name' --raw-output)
    echo "[deploy] availability domain: $OCI_AD"
fi

if [[ -z "${OCI_IMAGE_ID:-}" ]]; then
    echo "[deploy] resolving latest GPU-capable Oracle Linux image..."
    # OCI has no Ubuntu GPU-preinstalled-driver image family (unlike AWS/GCP's
    # DL AMI/image family) -- Oracle Linux is the only OS with one. This
    # stack's GPU support is entirely Docker + NVIDIA Container Toolkit, so
    # the host OS doesn't need to match the Ubuntu-based Dockerfile.
    OCI_IMAGE_ID=$(oci compute image list \
        --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
        --operating-system "Oracle Linux" --shape "$OCI_SHAPE" \
        --sort-by TIMECREATED --sort-order DESC \
        --query 'data[0].id' --raw-output 2>&1)
    if [[ -z "$OCI_IMAGE_ID" || "$OCI_IMAGE_ID" == "null" ]]; then
        echo "[deploy] ERROR: no matching GPU image found for shape $OCI_SHAPE." >&2
        echo "         Check available images with:" >&2
        echo "         oci compute image list --compartment-id $OCI_COMPARTMENT_ID --operating-system 'Oracle Linux' --shape $OCI_SHAPE" >&2
        exit 1
    fi
    echo "[deploy] image: $OCI_IMAGE_ID"
fi

echo "[deploy] checking for an existing VCN (OCI has no default network)..."
VCN_ID=$(oci network vcn list --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
    --display-name "qatabasis-vcn" --query 'data[0].id' --raw-output 2>&1)

if [[ -z "$VCN_ID" || "$VCN_ID" == "null" ]]; then
    echo "[deploy] no existing qatabasis-vcn found -- creating minimal network"
    echo "         (VCN + public subnet + internet gateway + SSH-only security list)..."
    VCN_ID=$(oci network vcn create \
        --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
        --display-name "qatabasis-vcn" --cidr-block "10.0.0.0/16" \
        --query 'data.id' --raw-output)

    IGW_ID=$(oci network internet-gateway create \
        --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
        --vcn-id "$VCN_ID" --is-enabled true --display-name "qatabasis-igw" \
        --query 'data.id' --raw-output)

    RT_ID=$(oci network route-table list --compartment-id "$OCI_COMPARTMENT_ID" \
        --region "$OCI_REGION" --vcn-id "$VCN_ID" --query 'data[0].id' --raw-output)
    oci network route-table update --rt-id "$RT_ID" --region "$OCI_REGION" \
        --route-rules "[{\"destination\": \"0.0.0.0/0\", \"networkEntityId\": \"$IGW_ID\"}]" \
        --force >/dev/null

    SL_ID=$(oci network security-list list --compartment-id "$OCI_COMPARTMENT_ID" \
        --region "$OCI_REGION" --vcn-id "$VCN_ID" --query 'data[0].id' --raw-output)
    MY_IP=$(curl -s ifconfig.me)
    oci network security-list update --security-list-id "$SL_ID" --region "$OCI_REGION" \
        --ingress-security-rules "[{\"source\": \"${MY_IP}/32\", \"protocol\": \"6\", \"tcpOptions\": {\"destinationPortRange\": {\"min\": 22, \"max\": 22}}}]" \
        --force >/dev/null

    SUBNET_ID=$(oci network subnet create \
        --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
        --vcn-id "$VCN_ID" --display-name "qatabasis-subnet" \
        --cidr-block "10.0.0.0/24" \
        --query 'data.id' --raw-output)
else
    echo "[deploy] reusing existing qatabasis-vcn ($VCN_ID)"
    SUBNET_ID=$(oci network subnet list --compartment-id "$OCI_COMPARTMENT_ID" \
        --region "$OCI_REGION" --vcn-id "$VCN_ID" --display-name "qatabasis-subnet" \
        --query 'data[0].id' --raw-output)
fi

echo "[deploy] launching ${OCI_SHAPE} in ${OCI_AD}..."
INSTANCE_ID=$(oci compute instance launch \
    --compartment-id "$OCI_COMPARTMENT_ID" --region "$OCI_REGION" \
    --availability-domain "$OCI_AD" \
    --shape "$OCI_SHAPE" \
    --image-id "$OCI_IMAGE_ID" \
    --subnet-id "$SUBNET_ID" \
    --assign-public-ip true \
    --ssh-authorized-keys-file "$OCI_SSH_KEY" \
    --display-name "$INSTANCE_NAME" \
    --wait-for-state RUNNING \
    --query 'data.id' --raw-output)
echo "[deploy] instance $INSTANCE_ID launching..."

PUBLIC_IP=$(oci compute instance list-vnics \
    --instance-id "$INSTANCE_ID" --region "$OCI_REGION" \
    --query 'data[0]."public-ip"' --raw-output)
echo "[deploy] running at ${PUBLIC_IP}"

SSH_KEY_PRIV="${OCI_SSH_KEY%.pub}"
echo "[deploy] waiting for SSH..."
until ssh -i "$SSH_KEY_PRIV" -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
          opc@"$PUBLIC_IP" true 2>/dev/null; do
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
    -e "ssh -i $SSH_KEY_PRIV -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$REPO_ROOT/" opc@"$PUBLIC_IP":~/qatabasis/

# Oracle Linux ships Docker but the default user (opc) isn't in the docker
# group any more than Ubuntu's default user is on AWS/GCP -- same fix.
echo "[deploy] running cloud_bootstrap.sh on the instance (docker group + GPU-in-Docker check)..."
ssh -i "$SSH_KEY_PRIV" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    opc@"$PUBLIC_IP" \
    "cd ~/qatabasis && bash scripts/cloud_bootstrap.sh"

echo "[deploy] building the Docker image (same path used on every other cloud --"
echo "         keeps all providers on identical, reproducible infrastructure)..."
ssh -i "$SSH_KEY_PRIV" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    opc@"$PUBLIC_IP" \
    "cd ~/qatabasis && make build"

echo "[deploy] running readiness check (make doctor)..."
ssh -i "$SSH_KEY_PRIV" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    opc@"$PUBLIC_IP" \
    "cd ~/qatabasis && make doctor" \
    || echo "[deploy] WARNING: make doctor reported issues -- review before running the workload."

echo "[deploy] running smoke test (make pytest)..."
ssh -i "$SSH_KEY_PRIV" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    opc@"$PUBLIC_IP" \
    "cd ~/qatabasis && make pytest" \
    || echo "[deploy] WARNING: pytest did not exit 0 — investigate before running the workload."

cat <<EOF

[deploy] ready. instance: $INSTANCE_ID at $PUBLIC_IP

Follow-up:
  ssh -i $SSH_KEY_PRIV opc@$PUBLIC_IP
  cd ~/qatabasis
  make run NP=2 MOLECULES="H2 LiH BeH2 H2O"

Note: A10 is workstation-class in HardwareProfile (fp64:fp32 ~1/32) --
for a valid cross-cloud precision comparison against the fp64 Lambda A100
baseline, force VQE_PRECISION=fp64 explicitly:
  VQE_PRECISION=fp64 make run NP=2 MOLECULES="H2 LiH BeH2 H2O"

Before terminating (do this every time):
  rsync -av -e "ssh -i $SSH_KEY_PRIV" \\
      opc@$PUBLIC_IP:~/qatabasis/results/ results/

Terminate when done:
  oci compute instance terminate --instance-id $INSTANCE_ID --region $OCI_REGION --force
EOF
