#!/usr/bin/env bash
# Native install for HPC cluster (no Docker).
# Tested on: IE University capstone cluster (Debian 13, CUDA 12.4, EESSI, no module system).
# For local / reproducible runs, prefer the Docker path: `make build`.
#
# Usage:
#   bash scripts/install_native.sh
#
# Key overrides:
#   SCRATCH      scratch filesystem root     (default: /scratch/$USER)
#   ENV_PATH     full path for conda env     (default: $SCRATCH/hybrid-vqe)
#   MINIFORGE    conda root                  (default: $HOME/miniforge3)
#   CUDA_HOME    CUDA toolkit root           (default: /usr/local/cuda)
#   CUDA_ARCH    semicolon-separated SM list (default: 70;75;80;86;89;90)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/scratch/$USER}"
ENV_PATH="${ENV_PATH:-$SCRATCH/hybrid-vqe}"
MINIFORGE="${MINIFORGE:-$HOME/miniforge3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCH="${CUDA_ARCH:-70;75;80;86;89;90}"

echo "=== VQE Native Install ==="
echo "  Repo:      $REPO_ROOT"
echo "  Env:       $ENV_PATH"
echo "  CUDA:      $CUDA_HOME"
echo ""

# ------------------------------------------------------------------
# Step 1: CUDA — try module system first, then fall back to fixed path.
# On the IE capstone cluster CUDA is at /usr/local/cuda (no modules).
# ------------------------------------------------------------------
if command -v module &>/dev/null; then
    for _mod in "cuda/12.6" "cuda/12.4" "cuda/12.2" "cuda/12.0" "cuda/11.8" \
                "CUDA/12.6"  "CUDA/12.4"  "CUDA/12.2"  "CUDA/12.0" "cuda" "CUDA"; do
        if module load "$_mod" 2>/dev/null; then
            echo "[install] Loaded CUDA module: $_mod"
            break
        fi
    done
fi

# Add the fixed CUDA path regardless (harmless if already in PATH).
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc &>/dev/null; then
    echo "[install] ERROR: nvcc not found. Set CUDA_HOME or load the CUDA module."
    exit 1
fi
echo "[install] nvcc: $(nvcc --version | grep release)"

# ------------------------------------------------------------------
# Step 2: Conda env
# ------------------------------------------------------------------
if [ ! -d "$MINIFORGE" ]; then
    echo "[install] ERROR: miniforge not found at $MINIFORGE"
    echo "  Install with:"
    echo "    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "    bash Miniforge3-Linux-x86_64.sh -b -p $MINIFORGE"
    exit 1
fi

# shellcheck source=/dev/null
source "$MINIFORGE/bin/activate"

if [ -d "$ENV_PATH" ]; then
    echo "[install] Updating existing env at $ENV_PATH ..."
    conda env update --prefix "$ENV_PATH" --file "$REPO_ROOT/environment.yml" --prune -q
else
    echo "[install] Creating new env at $ENV_PATH ..."
    mkdir -p "$(dirname "$ENV_PATH")"
    conda env create --prefix "$ENV_PATH" --file "$REPO_ROOT/environment.yml" -q
fi

conda activate "$ENV_PATH"

# ------------------------------------------------------------------
# Step 3: Debian sysroot fix.
# The conda GCC cross-compiler expects glibc at /lib64/ (RedHat layout)
# but Debian puts it at /lib/x86_64-linux-gnu/. Symlink the two libs
# the linker needs into the conda sysroot so the final link step works.
# This is a no-op on clusters where /lib64 already exists.
# ------------------------------------------------------------------
if [ ! -e /lib64/libm.so.6 ] && [ -e /lib/x86_64-linux-gnu/libm.so.6 ]; then
    echo "[install] Applying Debian sysroot fix (libm/libmvec) ..."
    SYSROOT="$ENV_PATH/x86_64-conda-linux-gnu/sysroot/lib64"
    mkdir -p "$SYSROOT"
    ln -sf /lib/x86_64-linux-gnu/libm.so.6    "$SYSROOT/libm.so.6"    2>/dev/null || true
    ln -sf /lib/x86_64-linux-gnu/libmvec.so.1 "$SYSROOT/libmvec.so.1" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# Step 4: CMake build.
# Key flags:
#   -DCMAKE_CXX_COMPILER  — force conda GCC (avoids EESSI g++ / linker conflict)
#   -DMPI_CXX_SKIP_MPICXX — skip compile test; pass MPI paths directly instead
#   -DMPI_*               — use conda mpich headers + library (system OpenMPI
#                           may be broken on Debian/EESSI clusters)
# ------------------------------------------------------------------
echo "[install] Building hpc_core via CMake (archs: $CUDA_ARCH) ..."
mkdir -p "$REPO_ROOT/build"
cd "$REPO_ROOT/build"

CONDA_GXX="$ENV_PATH/bin/x86_64-conda-linux-gnu-g++"
if [ ! -x "$CONDA_GXX" ]; then
    # Fallback: let cmake pick the system compiler (works on non-EESSI clusters)
    CONDA_GXX=""
fi

/usr/bin/cmake "$REPO_ROOT" \
    -DPython_EXECUTABLE="$ENV_PATH/bin/python" \
    ${CONDA_GXX:+-DCMAKE_CXX_COMPILER="$CONDA_GXX"} \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    ${CONDA_GXX:+-DCMAKE_CUDA_HOST_COMPILER="$CONDA_GXX"} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DMPI_CXX_SKIP_MPICXX=TRUE \
    -DMPI_CXX_HEADER_DIR="$ENV_PATH/include" \
    -DMPI_CXX_LIB_NAMES=mpi \
    -DMPI_mpi_LIBRARY="$ENV_PATH/lib/libmpi.so"

make -j"$(nproc)"

# ------------------------------------------------------------------
# Step 4b: qiskit-aer with GPU support, built from source.
# environment.yml used to depend on a separate `qiskit-aer-gpu` PyPI
# package for this -- that package no longer exists (removed from PyPI;
# `conda env create` would fail outright on that line). GPU support now
# comes from building plain qiskit-aer with AER_THRUST_BACKEND=CUDA, same
# approach the Docker path already uses (see Dockerfile). Without this
# step, environment.yml's plain `qiskit-aer` pip install resolves to a
# CPU-only wheel -- this stack would still run, just silently without GPU
# acceleration despite CUDA/nvcc having already been confirmed present
# above. Safe to skip only if you deliberately want CPU-only here.
# ------------------------------------------------------------------
echo "[install] Building qiskit-aer from source with CUDA support ..."
AER_THRUST_BACKEND=CUDA python -m pip install --no-cache-dir --force-reinstall \
    --no-binary qiskit-aer "qiskit-aer==0.17.2"

# ------------------------------------------------------------------
# UCX sanity check - conda-forge ucx 1.20.0 ships a broken libuct
# (missing ucs_netif_is_ipoib symbol) that breaks mpi4py import.
# Auto-downgrade rather than letting the user hit it later.
# ------------------------------------------------------------------
echo "[install] Checking UCX compatibility ..."
if ! python -c "from mpi4py import MPI" 2>/dev/null; then
    echo "[install] mpi4py import failed - downgrading ucx<1.20 ..."
    conda install -c conda-forge "ucx<1.20" --force-reinstall -y
    python -c "from mpi4py import MPI" || {
        echo "[install] ERROR: mpi4py still broken after ucx downgrade."
        exit 1
    }
    echo "[install] UCX downgrade resolved the import."
fi

# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
cd "$REPO_ROOT"
echo "[install] Smoke testing import ..."
PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT" python -c "
import hpc_core
print('[install] hpc_core import OK')
"

SO_FILE="$(ls "$REPO_ROOT/build"/hpc_core*.so 2>/dev/null | head -1)"
echo ""
echo "=== Install complete ==="
echo "  Module: $SO_FILE"
echo ""
echo "Runtime env vars needed for mpirun (already in SLURM scripts):"
echo "  export UCX_TLS=sm,self"
echo "  export UCX_NET_DEVICES="
echo ""
echo "Next steps:"
echo "  Diagnostic (interactive): make native-trial NP=2"
echo "  SLURM trial job:          make slurm-trial"
echo "  Full scaling sweep:       make slurm-scaling"