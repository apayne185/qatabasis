#!/usr/bin/env bash
# Capture an Nsight Systems timeline of one short VQE run on H2O.
# Design + rationale: docs/PROFILING.md.
#
# Assumes: running inside vqe-mpi-gpu:profiling (or a host with nsys
# installed + this repo checked out). GPU visible. Conda env active if not
# using the container.
#
# Output: results/profiling/<timestamp>.nsys-rep, opened later in the
# Nsight Systems GUI locally.

set -eo pipefail

MOLECULE="${MOLECULE:-H2O}"
MAX_ITERS="${MAX_ITERS:-10}"
NP="${NP:-2}"
OUT_DIR="${OUT_DIR:-results/profiling}"

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
REPORT="${OUT_DIR}/${MOLECULE}_np${NP}_iters${MAX_ITERS}_${TS}"

echo "[nsys] profiling ${MOLECULE} NP=${NP} MAX_ITERS=${MAX_ITERS}"
echo "[nsys] output → ${REPORT}.nsys-rep"

# Env vars read by benchmarks/local_test_run.py — export before the mpirun
# subprocess so both ranks see them.
export MOLECULES="$MOLECULE"
export MAX_ITERS="$MAX_ITERS"

# --trace: what to record.
#   cuda      — CUDA API calls + kernels (the core of what we want).
#   nvtx      — hot-region annotations from src/api/interface.py (once wired).
#   mpi       — MPI calls, so scaling-limit diagnosis is possible from the
#               timeline. Requires OpenMPI's PMPI hooks — Lambda AMI has this.
#   osrt      — OS runtime (thread state); modest cost, useful for spotting
#               Python-GIL contention.
# --sample=cpu: also sample the CPU-side call stack at 1kHz. Cheap and lets
#               you spot Python overhead between kernel launches.
# --stats=true: print a summary table at the end so you don't need the GUI
#               to sanity-check the run before rsync-ing.
#
# Two real, independent bugs were found and fixed here (2026-11-03) --
# this script had apparently never produced a usable CUDA-trace capture:
#
# 1. `mpirun` had no `--allow-run-as-root` (every OTHER mpirun invocation
#    in this repo's Makefile has it -- this script was the one exception).
#    Inside the Docker container, the shell runs as root, so bare `mpirun`
#    refuses to launch at all: prints its root-refusal warning and exits
#    with ZERO ranks ever started. The `nsys profile` wrapping it still
#    "succeeds" (it traced a real, near-instant process -- just one that
#    did nothing), producing a tiny, empty-of-CUDA-data report with no
#    error that points at the actual cause. First symptom noticed: a
#    captured .nsys-rep's --stats=true output showed "SKIPPED: ... does
#    not contain CUDA trace/kernel/memory data" for every CUDA table,
#    despite the real underlying workload (NH3 at P=2) definitely using
#    the GPU when run normally (confirmed separately, outside nsys).
#
# 2. Even with (1) fixed, nsys MUST be launched INSIDE the mpirun
#    invocation (one nsys process per rank), not wrapped AROUND mpirun
#    from the outside. mpirun forks its rank processes via fork+exec;
#    nsys attached to the mpirun PID itself does not automatically trace
#    those child processes' CUDA activity -- nsys --help's
#    --trace-fork-before-exec flag note explains why: child-process
#    tracing after fork is NOT automatic and is documented as relying on
#    undefined behavior even when explicitly enabled. `mpirun -np N nsys
#    profile ... python ...` instead gives each rank its own nsys
#    instance tracing its own process directly -- the standard,
#    documented pattern for MPI+CUDA profiling. --output gets a
#    %q{OMPI_COMM_WORLD_RANK} suffix so rank 0 and rank 1 (etc.) each get
#    a distinct, non-colliding report file instead of one rank's write
#    clobbering the other's.
mpirun --allow-run-as-root -np "$NP" \
    nsys profile \
        --trace=cuda,nvtx,mpi,osrt \
        --sample=cpu \
        --stats=true \
        --output "${REPORT}_rank%q{OMPI_COMM_WORLD_RANK}" \
        --force-overwrite=true \
        python -m benchmarks.local_test_run

echo ""
echo "[nsys] done."
echo "[nsys] reports: ${REPORT}_rank0.nsys-rep, ${REPORT}_rank1.nsys-rep, ..."
echo "[nsys] rsync back to laptop before terminating the instance:"
echo "  rsync -av <instance>:$(pwd)/${OUT_DIR}/ ${OUT_DIR}/"
echo "[nsys] open with: nsight-sys ${REPORT}_rank0.nsys-rep"
