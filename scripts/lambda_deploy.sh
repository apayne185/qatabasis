#!/usr/bin/env bash
# Bootstrap a Lambda Cloud A100-SXM4-40GB instance for QatabasisStack --
# this is the paper's PRIMARY hardware (see CLAUDE.md), so unlike
# aws_deploy.sh/gcp_deploy.sh (secondary, multi-cloud-validation-only
# providers), this script's default instance type is the one all of the
# paper's headline numbers were measured on.
#
# Uses the same Docker path as AWS/GCP (make build / make run), via
# scripts/cloud_bootstrap.sh, so every provider runs identical,
# reproducible infrastructure.
#
# Lambda Cloud has no AMI/image-family resolution step and no VPC/security
# group setup, unlike AWS/GCP -- every Lambda instance boots a fixed
# Ubuntu + NVIDIA driver image with SSH open by default, so this script is
# shorter than its AWS/GCP siblings by exactly that much.
#
# Usage:
#   LAMBDA_API_KEY=... LAMBDA_SSH_KEY_NAME=my-key scripts/lambda_deploy.sh
#
# Env vars:
#   LAMBDA_API_KEY      Required. From the Lambda Cloud dashboard
#                        (cloud.lambdalabs.com -> API keys).
#   LAMBDA_SSH_KEY_NAME  Required. The SSH key name already registered in
#                        your Lambda Cloud account (dashboard -> SSH keys) --
#                        this script does not upload a new key, it launches
#                        with one that already exists there.
#   LAMBDA_SSH_KEY_PATH  Local private key path matching LAMBDA_SSH_KEY_NAME.
#                        Default: ~/.ssh/<LAMBDA_SSH_KEY_NAME>.pem
#   LAMBDA_INSTANCE_TYPE  Default: gpu_1x_a100_sxm4 (the paper's primary
#                        hardware -- A100-SXM4-40GB). Pass
#                        gpu_1x_a100 for the PCIe variant if SXM4 capacity
#                        is unavailable in every region (rare, but Lambda's
#                        on-demand A100-SXM4 stock does run out).
#   LAMBDA_REGION        Default: us-east-1. Script tries other regions
#                        automatically if this one is sold out (see below).
#   INSTANCE_NAME        Default: qatabasis-exp1-sweep.
#
# Cleanup is manual (see the printed follow-up command) -- deliberate, so a
# rsync-back-results step is never skipped (same rationale as the other
# three cloud deploy scripts).

set -eo pipefail   # NOT -u; conda activate scripts hit unset vars

: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY (Lambda Cloud dashboard -> API keys)}"
: "${LAMBDA_SSH_KEY_NAME:?set LAMBDA_SSH_KEY_NAME (an SSH key name already registered in your Lambda Cloud account)}"
LAMBDA_SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:-$HOME/.ssh/${LAMBDA_SSH_KEY_NAME}.pem}"
LAMBDA_INSTANCE_TYPE="${LAMBDA_INSTANCE_TYPE:-gpu_1x_a100_sxm4}"
LAMBDA_REGION="${LAMBDA_REGION:-us-east-1}"
INSTANCE_NAME="${INSTANCE_NAME:-qatabasis-exp1-sweep}"
API="https://cloud.lambdalabs.com/api/v1"

if [[ ! -f "$LAMBDA_SSH_KEY_PATH" ]]; then
    echo "[deploy] ERROR: SSH private key not found at $LAMBDA_SSH_KEY_PATH" >&2
    echo "         Set LAMBDA_SSH_KEY_PATH to the local .pem matching the" >&2
    echo "         Lambda Cloud SSH key named '$LAMBDA_SSH_KEY_NAME'." >&2
    exit 1
fi
chmod 400 "$LAMBDA_SSH_KEY_PATH"

_api() {
    curl -sf -u "${LAMBDA_API_KEY}:" "$@"  # gitleaks:allow -- env var reference, not a literal key
}

echo "[deploy] checking A100-SXM4 availability..."
AVAILABLE_REGIONS=$(_api "$API/instance-types" \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
entry = d.get('$LAMBDA_INSTANCE_TYPE')
if not entry:
    print('')
else:
    print(' '.join(r['name'] for r in entry.get('regions_with_capacity_available', [])))
")

if [[ -z "$AVAILABLE_REGIONS" ]]; then
    echo "[deploy] ERROR: no region currently has capacity for $LAMBDA_INSTANCE_TYPE." >&2
    echo "         Check https://cloud.lambdalabs.com/instances for live stock," >&2
    echo "         or retry later -- A100-SXM4 on-demand stock does sell out." >&2
    exit 1
fi

if [[ ! " $AVAILABLE_REGIONS " =~ " $LAMBDA_REGION " ]]; then
    FALLBACK=$(echo "$AVAILABLE_REGIONS" | awk '{print $1}')
    echo "[deploy] $LAMBDA_REGION has no capacity right now. Available: $AVAILABLE_REGIONS"
    echo "[deploy] falling back to region: $FALLBACK"
    LAMBDA_REGION="$FALLBACK"
fi
echo "[deploy] region: $LAMBDA_REGION"

echo "[deploy] launching $LAMBDA_INSTANCE_TYPE in $LAMBDA_REGION..."
INSTANCE_ID=$(_api -X POST "$API/instance-operations/launch" \
    -H "Content-Type: application/json" \
    -d "{
        \"region_name\": \"$LAMBDA_REGION\",
        \"instance_type_name\": \"$LAMBDA_INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$LAMBDA_SSH_KEY_NAME\"],
        \"name\": \"$INSTANCE_NAME\"
    }" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['instance_ids'][0])")
echo "[deploy] instance $INSTANCE_ID launching..."

echo "[deploy] waiting for it to come up..."
PUBLIC_IP=""
until [[ -n "$PUBLIC_IP" && "$PUBLIC_IP" != "null" ]]; do
    sleep 10
    STATUS_JSON=$(_api "$API/instances/$INSTANCE_ID")
    STATUS=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['status'])")
    echo "[deploy] status: $STATUS"
    if [[ "$STATUS" == "active" ]]; then
        PUBLIC_IP=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'].get('ip') or 'null')")
    fi
done
echo "[deploy] running at ${PUBLIC_IP}"

echo "[deploy] waiting for SSH..."
until ssh -i "$LAMBDA_SSH_KEY_PATH" -o StrictHostKeyChecking=no \
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
    -e "ssh -i $LAMBDA_SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$REPO_ROOT/" ubuntu@"$PUBLIC_IP":~/qatabasis/

echo "[deploy] running cloud_bootstrap.sh on the instance (docker group + GPU-in-Docker check)..."
ssh -i "$LAMBDA_SSH_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && bash scripts/cloud_bootstrap.sh"

echo "[deploy] building the Docker image..."
echo "[deploy] NOTE: if cloud_bootstrap.sh just added this user to the docker"
echo "         group for the first time, this SSH command runs in a NEW"
echo "         connection (not the bootstrap's own shell), so group membership"
echo "         is already active here -- no separate re-login needed."
ssh -i "$LAMBDA_SSH_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && make build"

echo "[deploy] running readiness check (make doctor)..."
ssh -i "$LAMBDA_SSH_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && make doctor" \
    || echo "[deploy] WARNING: make doctor reported issues — review before running the workload."

echo "[deploy] running smoke test (make pytest)..."
ssh -i "$LAMBDA_SSH_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@"$PUBLIC_IP" \
    "cd ~/qatabasis && make pytest" \
    || echo "[deploy] WARNING: pytest did not exit 0 — investigate before running the workload."

cat <<EOF

[deploy] ready. instance: $INSTANCE_ID at $PUBLIC_IP (region $LAMBDA_REGION)

Follow-up -- Experiment 1 clean sweep, batch 1 (H2, LiH, BeH2, H2O):
  ssh -i $LAMBDA_SSH_KEY_PATH ubuntu@$PUBLIC_IP
  cd ~/qatabasis
  make exp1-sweep

Then batch 2 (NH3 -- the expensive one, run separately, see
scripts/run_exp1_sweep.sh's note on interrupting/resuming it):
  make exp1-sweep EXP1_BATCH=2

Then batch 3 (N2 -- optional, see the script's note on whether you even
need this; it has no serial comparison regardless):
  make exp1-sweep EXP1_BATCH=3

Copy results back after EACH batch, not just at the end -- if the
instance is interrupted mid-NH3, you keep whatever batches already
finished instead of losing the whole session:
  rsync -av -e "ssh -i $LAMBDA_SSH_KEY_PATH" \\
      ubuntu@$PUBLIC_IP:~/qatabasis/results/ \\
      results/

Terminate when done (Lambda bills per-minute while running), using the
same basic-auth convention as this script's own _api() helper above
(LAMBDA_API_KEY as username, empty password) with LAMBDA_API_KEY still
set in your shell:
  curl -sf --user "LAMBDA_API_KEY_here_followed_by_a_colon_and_nothing" \\
      -X POST "$API/instance-operations/terminate" \\
      -H "Content-Type: application/json" \\
      -d '{"instance_ids": ["$INSTANCE_ID"]}'
EOF
