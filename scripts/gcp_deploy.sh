#!/usr/bin/env bash
# Bootstrap a GCP Compute Engine GPU instance for QatabasisStack multi-cloud
# validation. Uses the same Docker path as Lambda/AWS (make build / make
# run), via scripts/cloud_bootstrap.sh, so all three clouds run identical,
# reproducible infrastructure rather than a different install method per
# provider. Mirrors scripts/aws_deploy.sh's structure.
#
# GCP's default instance type has NO GPU attached -- unlike AWS/Lambda where
# the instance type itself implies a GPU (g5.xlarge, A100 instance, etc),
# GCP GPUs are a separate --accelerator flag on a generic machine type. This
# script defaults to a single T4 (n1-standard-4 + nvidia-tesla-t4), the
# cheapest GPU tier with a non-zero free-trial quota (see docs/*_DEPLOYMENT.md
# -- unlike A100, T4/L4/V100/P100 default to quota=1 on a fresh project,
# no quota-increase request needed).
#
# Usage:
#   GCP_PROJECT=my-project GCP_ZONE=us-central1-a scripts/gcp_deploy.sh
#
# Env vars (all optional, sensible defaults shown):
#   GCP_PROJECT      GCP project id. Default: current `gcloud config` project.
#   GCP_ZONE         Default: us-central1-a.
#   GCP_MACHINE_TYPE Default: n1-standard-4 (pairs with T4; use g2-standard-4 for L4).
#   GCP_GPU_TYPE     Default: nvidia-tesla-t4.
#   GCP_GPU_COUNT    Default: 1.
#   GCP_IMAGE_FAMILY Default: latest common-cuXXX-ubuntu-2204-nvidia-* in the
#                    public `ml-images` project (Google's DL VM image family).
#   INSTANCE_NAME    Default: qatabasis-validation.
#
# Cleanup is manual (see the printed follow-up command) -- deliberate, so a
# rsync-back-results step is never skipped (same rationale as aws_deploy.sh).

set -eo pipefail   # NOT -u; conda activate scripts hit unset vars

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-standard-4}"
GCP_GPU_TYPE="${GCP_GPU_TYPE:-nvidia-tesla-t4}"
GCP_GPU_COUNT="${GCP_GPU_COUNT:-1}"
INSTANCE_NAME="${INSTANCE_NAME:-qatabasis-validation}"

if [[ -z "$GCP_PROJECT" ]]; then
    echo "[deploy] ERROR: no GCP project set. Run 'gcloud config set project <id>'" >&2
    echo "         or pass GCP_PROJECT=<id>." >&2
    exit 1
fi

if [[ -z "${GCP_IMAGE_FAMILY:-}" ]]; then
    echo "[deploy] resolving latest GPU-capable Ubuntu DL image family..."
    # Google's ml-images project periodically revs the CUDA version in the
    # family name (e.g. common-cu129-ubuntu-2204-nvidia-580) -- match loosely
    # rather than hardcoding an exact family name, same rationale as
    # aws_deploy.sh's AMI resolution (this stack builds its own Docker image,
    # so the base image only needs a working NVIDIA driver + Ubuntu base).
    GCP_IMAGE_FAMILY=$(gcloud compute images list --project=ml-images \
        --filter="family~common-cu.*ubuntu-2204" \
        --format="value(family)" 2>&1 | sort -V | tail -1)
    if [[ -z "$GCP_IMAGE_FAMILY" ]]; then
        echo "[deploy] ERROR: no matching image family found in ml-images." >&2
        echo "         Check available families with:" >&2
        echo "         gcloud compute images list --project=ml-images --filter='family~common-cu'" >&2
        exit 1
    fi
    echo "[deploy] image family: $GCP_IMAGE_FAMILY"
fi

echo "[deploy] launching ${GCP_MACHINE_TYPE} + ${GCP_GPU_COUNT}x ${GCP_GPU_TYPE} in ${GCP_ZONE}..."
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="$GCP_MACHINE_TYPE" \
    --accelerator="type=${GCP_GPU_TYPE},count=${GCP_GPU_COUNT}" \
    --image-family="$GCP_IMAGE_FAMILY" \
    --image-project=ml-images \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd

echo "[deploy] waiting for SSH..."
until gcloud compute ssh "$INSTANCE_NAME" \
        --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
        --command="true" -- -o ConnectTimeout=5 2>/dev/null; do
    sleep 5
done
echo "[deploy] SSH up."

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "[deploy] uploading repo from $REPO_ROOT..."
gcloud compute scp --recurse \
    --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --scp-flag="-o StrictHostKeyChecking=no" \
    "$REPO_ROOT" "${INSTANCE_NAME}:~/qatabasis-upload" 2>&1 | grep -v "^Warning:" || true

# gcloud scp --recurse copies the local dir itself (not just its contents),
# and has no --exclude flag -- unlike aws_deploy.sh's targeted rsync
# excludes (.git/results/__pycache__/build), so move+clean on the remote
# side instead of trying to filter the upload.
gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="
    rm -rf ~/qatabasis
    mv ~/qatabasis-upload ~/qatabasis
    cd ~/qatabasis
    rm -rf .git results __pycache__ build .pubchem_cache
    find . -name '*.pyc' -delete
"

echo "[deploy] running cloud_bootstrap.sh on the instance (docker group + GPU-in-Docker check)..."
gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --command="cd ~/qatabasis && bash scripts/cloud_bootstrap.sh"

echo "[deploy] building the Docker image (same path used on Lambda/AWS -- keeps"
echo "         all three clouds on identical, reproducible infrastructure)..."
echo "[deploy] NOTE: if cloud_bootstrap.sh just added this user to the docker"
echo "         group for the first time, this SSH command runs in a NEW"
echo "         connection (not the bootstrap's own shell), so group membership"
echo "         is already active here -- no separate re-login needed."
gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --command="cd ~/qatabasis && make build"

echo "[deploy] running readiness check (make doctor)..."
gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --command="cd ~/qatabasis && make doctor" \
    || echo "[deploy] WARNING: make doctor reported issues -- review before running the workload."

echo "[deploy] running smoke test (make pytest)..."
gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --command="cd ~/qatabasis && make pytest" \
    || echo "[deploy] WARNING: pytest did not exit 0 — investigate before running the workload."

cat <<EOF

[deploy] ready. instance: $INSTANCE_NAME in $GCP_ZONE (project $GCP_PROJECT)

Follow-up:
  gcloud compute ssh $INSTANCE_NAME --project=$GCP_PROJECT --zone=$GCP_ZONE
  cd ~/qatabasis
  make run NP=2 MOLECULES="H2 LiH BeH2 H2O"

Note: T4/L4 classify as workstation-class in HardwareProfile (fp64:fp32
~1/64) -- for a valid cross-cloud precision comparison against the fp64
Lambda A100 baseline, force VQE_PRECISION=fp64 explicitly:
  VQE_PRECISION=fp64 make run NP=2 MOLECULES="H2 LiH BeH2 H2O"

Before terminating (do this every time):
  gcloud compute scp --recurse --project=$GCP_PROJECT --zone=$GCP_ZONE \\
      ${INSTANCE_NAME}:~/qatabasis/results/ results/

Terminate when done:
  gcloud compute instances delete $INSTANCE_NAME --project=$GCP_PROJECT --zone=$GCP_ZONE
EOF
