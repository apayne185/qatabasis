#!/usr/bin/env bash
# One-time setup for a fresh cloud GPU instance (Lambda, AWS, or any other
# Docker+NVIDIA host) using the Docker path (make build / make trial / make run).
#
# Fixes the specific friction hit repeatedly when spinning up a new instance:
# the default cloud user isn't in the `docker` group yet, so `docker build`/
# `docker run` fail with "permission denied ... /var/run/docker.sock" -- and
# critically, the Makefile's own GPU probe (`docker run --gpus all ... nvidia-smi`)
# fails the EXACT SAME WAY, so a Docker permission problem silently looks
# identical to "no GPU detected" and falls back to CPU with no explanation.
# This script tells the two apart explicitly instead of leaving you to
# discover it the hard way.
#
# Usage (run once per fresh instance, right after cloning, before make build):
#   bash scripts/cloud_bootstrap.sh
#
# Safe to re-run -- every step is idempotent (checks before acting).

set -eo pipefail   # NOT -u; some distros' profile scripts reference unset vars

echo "=== Qatabasis cloud instance bootstrap ==="
echo

# ------------------------------------------------------------------
# Step 1: is Docker even installed?
# ------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "[bootstrap] ERROR: docker is not installed on this instance."
    echo "            Lambda/AWS Deep Learning AMIs normally ship it preinstalled --"
    echo "            if it's missing, this may be a bare Ubuntu image; install"
    echo "            Docker + the NVIDIA Container Toolkit before continuing."
    exit 1
fi
echo "[bootstrap] docker found: $(docker --version)"

# ------------------------------------------------------------------
# Step 2: docker group membership. This is the friction point:
# a fresh cloud user is often NOT in the docker group, so docker
# build/run need sudo until the user re-logs in after being added.
# ------------------------------------------------------------------
CURRENT_USER="$(whoami)"
NEEDS_RELOGIN=0

if groups "$CURRENT_USER" | grep -qw docker; then
    echo "[bootstrap] $CURRENT_USER is already in the docker group."
    DOCKER_CMD="docker"
else
    echo "[bootstrap] $CURRENT_USER is NOT in the docker group yet -- adding."
    sudo usermod -aG docker "$CURRENT_USER"
    echo "[bootstrap] Added. This takes effect on your NEXT login, not this shell --"
    echo "            using 'sudo docker' for the rest of THIS bootstrap run only."
    DOCKER_CMD="sudo docker"
    NEEDS_RELOGIN=1
fi

# ------------------------------------------------------------------
# Step 3: can Docker actually reach the daemon? (belt-and-suspenders --
# catches a genuinely broken/unstarted daemon, not just a group issue)
# ------------------------------------------------------------------
if ! $DOCKER_CMD info &>/dev/null; then
    echo "[bootstrap] ERROR: '$DOCKER_CMD info' failed -- the Docker daemon"
    echo "            itself may not be running. Try: sudo systemctl start docker"
    exit 1
fi
echo "[bootstrap] Docker daemon is reachable."

# ------------------------------------------------------------------
# Step 4: is the NVIDIA GPU actually visible to a Docker container?
# This is the check the Makefile's GPU_AVAILABLE was silently getting
# wrong when it was really a permission problem, not a missing GPU --
# do it explicitly here, with the right permissions, so the answer is
# trustworthy before you ever run `make build`.
# ------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "[bootstrap] nvidia-smi on host: OK"
else
    echo "[bootstrap] WARNING: no nvidia-smi on the host itself -- this instance"
    echo "            may not actually have a GPU attached. Continuing anyway;"
    echo "            the stack will fall back to CPU-only if so."
fi

echo "[bootstrap] Checking GPU visibility INSIDE a container (this is the check"
echo "            that actually matters for make build/trial/run) ..."
if $DOCKER_CMD run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/tmp/bootstrap_gpu_check.log; then
    echo "[bootstrap] GPU IS visible inside Docker containers. make build/run will"
    echo "            correctly detect and use the GPU."
else
    echo "[bootstrap] GPU is NOT visible inside Docker containers."
    echo "            Host nvidia-smi output above tells you whether a GPU exists"
    echo "            at all; if it does, the NVIDIA Container Toolkit is likely"
    echo "            missing or misconfigured. Full error:"
    sed 's/^/              /' /tmp/bootstrap_gpu_check.log
    echo "            The stack will still run (CPU fallback), just without GPU"
    echo "            acceleration -- fix this before running anything you intend"
    echo "            to cite in the paper as a GPU result."
fi
rm -f /tmp/bootstrap_gpu_check.log

echo
echo "=== Bootstrap summary ==="
if [ "$NEEDS_RELOGIN" -eq 1 ]; then
    echo "[bootstrap] IMPORTANT: you were just added to the docker group. This"
    echo "            bootstrap script itself used 'sudo docker' to get you a"
    echo "            correct answer above, but your NEXT manual 'make build' /"
    echo "            'docker run' command in THIS shell will still fail with"
    echo "            the same permission error, because group membership only"
    echo "            applies to new login sessions."
    echo
    echo "            Do ONE of the following before running 'make build':"
    echo "              (a) exit this SSH session and reconnect, or"
    echo "              (b) run: newgrp docker"
    echo "                  (starts a subshell with the new group active now --"
    echo "                  stay in that subshell for the rest of this session)"
else
    echo "[bootstrap] Already set up correctly -- go ahead and run 'make build'."
fi
