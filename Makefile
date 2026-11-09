IMAGE_NAME = vqe-mpi-gpu
NP ?= 2       						#override with -  make run NP=4
SEED ?= 42                          #override with - make run SEED=43
MOLECULES ?=                        #override with - make run MOLECULES="H2 LiH"  (default: H2 LiH BeH2 H2O)
MAX_ITERS ?=                        #override with - make run MAX_ITERS=50
VQE_PRECISION ?= auto                #override with - make run VQE_PRECISION=fp32

ifneq (,$(wildcard .env))
  include .env
  export
endif
 
# Single invocation, output captured once -- checked for both success and
# (on failure) the permission-denied signature, instead of running the
# probe twice with two different failure-detection strategies.
GPU_PROBE_OUTPUT := $(shell docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi 2>&1; echo "EXIT:$$?")
GPU_AVAILABLE := $(if $(findstring EXIT:0,$(GPU_PROBE_OUTPUT)),yes,no)
DOCKER_PERM_DENIED := $(if $(findstring permission denied,$(GPU_PROBE_OUTPUT)),yes,no)

ifeq ($(GPU_AVAILABLE),yes)
  GPU_FLAG = --gpus all
  $(info [Make] GPU detected — CUDA acceleration enabled.)
else
  GPU_FLAG =
  # A Docker permission problem (user not in the `docker` group yet) looks
  # IDENTICAL to "no GPU present" here unless we actually inspect the
  # error -- this exact confusion cost real debugging time on a fresh
  # Lambda instance this session. Check for the permission-denied
  # signature and point at the real fix instead of the misleading
  # "No GPU detected" message.
  ifeq ($(DOCKER_PERM_DENIED),yes)
    $(info [Make] GPU probe failed with a Docker PERMISSION error, not necessarily)
    $(info [Make] a missing GPU. Run: bash scripts/cloud_bootstrap.sh)
    $(info [Make] Falling back to CPU mode for now.)
  else
    $(info [Make] No GPU detected — falling back to CPU mode.)
  endif
endif


.PHONY: build trial run run-ibm scaling baseline exp1-sweep clean shell test pytest doctor \
        test-sparse-pauli-reuse \
        native-install native-trial native-run \
        slurm-trial slurm-run slurm-scaling slurm-weak-scaling slurm-ibm \
        slurm-multi-seed slurm-ibm-seeds aggregate-seeds aggregate-scaling \
        backup-results check-docker-perms

# GUARD - every Docker-invoking target depends on this. A fresh cloud
# instance's default user is never in the `docker` group yet, so `docker
# build`/`docker run` fail with "permission denied ... docker.sock" --
# and critically, this is the EXACT SAME error signature the GPU probe
# above already detects and silently works around (by falling back to
# CPU mode) rather than failing loudly. Without this guard, `make build`
# "succeeds" at printing a hint buried above a raw Docker error, then
# fails anyway -- costing real debugging time on a fresh Lambda instance.
# This guard turns that into one unmissable stop with the actual fix.
check-docker-perms:
ifeq ($(DOCKER_PERM_DENIED),yes)
	$(error [Make] Docker permission denied -- this user is not in the \
`docker` group yet (common on a fresh cloud instance). Run: \
bash scripts/cloud_bootstrap.sh -- then either reconnect over SSH or run \
`newgrp docker` in this shell, and re-run this command.)
endif

build: check-docker-perms
	@echo "[Make] Building Docker image '$(IMAGE_NAME)' ..."
	docker build -t $(IMAGE_NAME) .
	@echo "[Make] Build complete."


# READINESS CHECK - single command to answer "is this environment ready
# to use this stack" for any target: cloud GPU, IBM QPU, or CPU-only.
# Runs inside the container with the same --gpus flag `make trial`/`make
# run` use, so GPU visibility is checked exactly as the real workload
# would see it. See scripts/doctor.py for what's actually checked.
doctor: check-docker-perms
	@echo "[Make] Running readiness check ..."
	@docker run --rm \
	  $(GPU_FLAG) \
	  -e IBM_QUANTUM_TOKEN="$(IBM_QUANTUM_TOKEN)" \
	  -e IBM_QUANTUM_INSTANCE="$(IBM_QUANTUM_INSTANCE)" \
	  -e IBM_QUANTUM_BACKEND="$(IBM_QUANTUM_BACKEND)" \
	  -e IBM_QUANTUM_REGION="$(IBM_QUANTUM_REGION)" \
	  $(IMAGE_NAME) \
	  python3 scripts/doctor.py


# DIAGNOSTIC - tests the 6 layers on simulator
trial: check-docker-perms
	@echo "[Make] Running diagnostic trial (simulator, $(NP) ranks) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
 	  mpirun --allow-run-as-root -np $(NP) python3 tests/test_layers_run.py       


# RUN TEMPLATE SCRIPT
example: check-docker-perms
	@echo "[Make] Running template (simulator, $(NP) ranks) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -e VQE_PRECISION=$(VQE_PRECISION) \
	  -v "$$(pwd)/results:/workspace/results" \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np $(NP) python3 template.py

# FULL BENCHMARK - simualtor only
run: check-docker-perms
	@echo "[Make] Running full benchmark (simulator, $(NP) ranks) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -e SEED=$(SEED) \
	  -e MOLECULES="$(MOLECULES)" \
	  -e MAX_ITERS=$(MAX_ITERS) \
	  -e VQE_PRECISION=$(VQE_PRECISION) \
	  -e RESUME=$(RESUME) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np $(NP) python3 benchmarks/local_test_run.py




# # FULL BENCHMARK - IBM quantum QPU 
run-ibm: check-docker-perms
	@[ -n "$(IBM_QUANTUM_TOKEN)" ] || (echo "ERROR: IBM_QUANTUM_TOKEN not set in .env"; exit 1)
	@[ -n "$(IBM_QUANTUM_INSTANCE)" ] || (echo "ERROR: IBM_QUANTUM_INSTANCE not set in .env"; exit 1)
	@echo "[Make] Running $(NP) ranks -> IBM Quantum ($(IBM_QUANTUM_BACKEND)) ..."
	@docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=ibm_cloud \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -e IBM_QUANTUM_TOKEN="$(IBM_QUANTUM_TOKEN)" \
	  -e IBM_QUANTUM_INSTANCE="$(IBM_QUANTUM_INSTANCE)" \
	  -e IBM_QUANTUM_BACKEND="$(IBM_QUANTUM_BACKEND)" \
	  -e IBM_QUANTUM_REGION="$(IBM_QUANTUM_REGION)" \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np $(NP) python3 benchmarks/ibm_test_run.py


# STRONG SCALAING SWEEP - simulator      
scaling: check-docker-perms
	@echo "[Make] Starting strong scaling analysis ..."
	@mkdir -p results/scaling
	@for p in 1 2 4 8; do \
	  echo "  Running P=$$p ..."; \
	  docker run --rm \
	    $(GPU_FLAG) \
	    -e BACKEND=simulator \
		-e USE_GPU=$(GPU_AVAILABLE) \
	    -e RUN_SCALING=1 \
	    -v "$$(pwd)/results:/workspace/results" \
	    $(IMAGE_NAME) \
	    mpirun --allow-run-as-root -np $$p python3 benchmarks/local_test_run.py \
	    > results/scaling/scaling_p$$p.log 2>&1; \
	  echo "  P=$$p done."; \
	done
	@echo "[Make] Scaling logs saved to results/scaling/. Check T_total and M-metric."

# WEAK SCALING SWEEP - problem size grows with P
# P=16 added for NH3 (see local_test_run.py:run_weak_scaling). N2 has no
# tier -- would need P=32 on the same single Docker host as everything
# else, deeper into the shared-memory contention regime than any other
# scaling data in the paper goes; deliberately left out, disclosed in text.
weak-scaling: check-docker-perms
	@echo "[Make] Starting weak scaling analysis ..."
	@mkdir -p results/scaling
	@for p in 1 2 4 8 16; do \
	  echo "  Running P=$$p (weak scaling) ..."; \
	  docker run --rm \
	    $(GPU_FLAG) \
	    -e BACKEND=simulator \
		-e USE_GPU=$(GPU_AVAILABLE) \
	    -v "$$(pwd)/results:/workspace/results" \
	    -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	    $(IMAGE_NAME) \
	    mpirun --allow-run-as-root -np $$p python3 -c \
	    "import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,'build'); \
	     from src.api.interface import QatabasisStack; \
	     from benchmarks.local_test_run import run_weak_scaling; \
	     stack = QatabasisStack(use_gpu=os.environ.get('USE_GPU','no')=='yes', backend='simulator'); \
	     run_weak_scaling(stack); stack.finalize()" \
	    > results/scaling/weak_scaling_p$$p.log 2>&1; \
	  echo "  P=$$p done."; \
	done
	@echo "[Make] Weak scaling results saved to results/scaling/."



# SERIAL BASELINE - single-core Qiskit VQE for comparison (no MPI)
baseline: check-docker-perms
	@echo "[Make] Running serial Qiskit baseline (no MPI, no GPU) ..."
	docker run --rm \
	  -e USE_GPU=no \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  python3 benchmarks/serial_baseline.py
	@echo "[Make] Serial baseline complete."


# EXPERIMENT 1 CLEAN SWEEP - AerSimulator-CPU serial baseline + distributed
# P=1,2, 5 seeds/molecule (10 for LiH), all in one container session so
# every file shares the same git_commit/hostname and there is no
# cross-session file-pairing judgment call left for later. Run in batches
# (EXP1_BATCH=1/2/3/all) so a multi-hour NH3 sweep can be safely
# interrupted without re-paying for the cheaper molecules. See
# qatabasis-internal/docs/RESULT_PROVENANCE.md ("Experiment 1 root cause")
# for why this target exists instead of composing `baseline` + `scaling`
# by hand, and scripts/run_exp1_sweep.sh for the batch breakdown.
#   make exp1-sweep                  # batch 1: H2, LiH, BeH2, H2O
#   make exp1-sweep EXP1_BATCH=2     # batch 2: NH3
#   make exp1-sweep EXP1_BATCH=3     # batch 3: N2 (optional -- see script's note)
EXP1_BATCH ?= 1
exp1-sweep: check-docker-perms
	@echo "[Make] Running Experiment 1 clean sweep, batch $(EXP1_BATCH) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -e EXP1_BATCH=$(EXP1_BATCH) \
	  -v "$$(pwd)/results:/workspace/results" \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  bash scripts/run_exp1_sweep.sh
	@echo "[Make] Experiment 1 clean sweep batch $(EXP1_BATCH) complete. See results/*-exp1-clean-*/MANIFEST.md"


# RUN ALL TESTS- resolver + layer diagnostic
test:
	@echo "[Make] Running test suite ..."
	docker run --rm $(IMAGE_NAME) python3 -m pytest
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np 2 python3 tests/test_layers_run.py
	@echo "[Make] All tests complete."

# pytest suite only (hardware profile + molecule resolver), no MPI layer test.
# Runs inside Docker -- the host Python has no qiskit/pyscf install.
pytest:
	docker run --rm $(IMAGE_NAME) python3 -m pytest


# CORRECTNESS CHECK for the SparsePauliOp-reuse caching fix (NH3/N2 P=2
# slowdown mitigation, FUTURE_WORK.md #7.1) -- requires GPU, verifies
# reusing one SparsePauliOp across save_expectation_value calls is
# numerically safe before trusting the speed fix in interface.py.
test-sparse-pauli-reuse: check-docker-perms
	@echo "[Make] Running SparsePauliOp-reuse correctness check ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  $(IMAGE_NAME) \
	  python3 tests/test_sparse_pauli_op_reuse.py
	@echo "[Make] SparsePauliOp-reuse check complete."



# LIST AVAILABLE MOLECULES - from the live registry
molecules:
	@docker run --rm $(IMAGE_NAME) python3 -c "\
	from src.api.problems import MOLECULE_REGISTRY; \
	print('Available molecules:'); \
	print(f'{\"Name\":<8} {\"Qubits\":<8} {\"FCI (Ha)\":<14} {\"Description\"}'); \
	print('-' * 60); \
	[print(f'{k:<8} {\"--\":<8} {v[\"fci_energy\"]:<14.4f} {v[\"description\"]}') for k, v in MOLECULE_REGISTRY.items()]"


shell: check-docker-perms
	docker run --rm -it \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  /bin/bash


clean:
	docker rmi $(IMAGE_NAME) || true
	rm -rf results/scaling/
	rm -f *.log *.npy
	find checkpoints/ -name "*.npy" -delete 2>/dev/null || true


# ============================================================
# NATIVE (conda) PATH - for HPC clusters where Docker is unavailable.
# Uses environment.yml + a native CMake build of the C++/CUDA module.
# For local reproducible runs, prefer the Docker targets above.
# ============================================================

# Install miniforge env + build hpc_core natively.
# Set SCRATCH=/scratch/$USER to install the env on fast local SSD (HPC recommended).
native-install:
	@echo "[Make] Native install (conda + CMake)..."
	bash scripts/install_native.sh

# Run the 7-layer diagnostic natively (no Docker, no Slurm).
native-trial:
	@echo "[Make] Native diagnostic trial ($(NP) ranks) ..."
	PYTHONPATH=./build:. mpirun -np $(NP) python tests/test_layers_run.py

# Run full simulator benchmark natively.
native-run:
	@echo "[Make] Native benchmark ($(NP) ranks) ..."
	PYTHONPATH=./build:. mpirun -np $(NP) python benchmarks/local_test_run.py

# Submit IBM QPU run to Slurm. Requires .env with IBM credentials.
# First-time setup: cp .env.example .env  then fill in your token.
slurm-ibm:
	@[ -f .env ] || (echo "ERROR: .env not found. Run: cp .env.example .env  then add your IBM credentials"; exit 1)
	@mkdir -p results/slurm
	sbatch scripts/slurm_ibm.sh

# Submit the 7-layer diagnostic to Slurm.
slurm-trial:
	@mkdir -p results/slurm
	sbatch scripts/slurm_trial.sh

# Submit the full simulator benchmark to Slurm (1 GPU).
slurm-run:
	@mkdir -p results/slurm
	sbatch scripts/slurm_gpu.sh

# Submit 4 jobs for the strong-scaling sweep (P=1,2,4,8).
# local_test_run.py runs the full benchmark + weak-scaling routine per job.
slurm-scaling:
	@mkdir -p results/slurm
	JOB_PREFIX=vqe-scale bash scripts/slurm_scaling.sh

# Weak-scaling sweep. Uses the same script path; named separately for
# clarity in the log filenames so results don't get mixed up.
slurm-weak-scaling:
	@mkdir -p results/slurm
	JOB_PREFIX=vqe-weak bash scripts/slurm_scaling.sh

# Multi-seed sweep for publication statistics. Submits one job per seed.
# Default seeds: 42 43 44. Override with SEEDS="42 43 44 45".
slurm-multi-seed:
	@mkdir -p results/slurm
	bash scripts/submit_multi_seed.sh

# Multi-seed IBM QPU runs (3 seeds, 10 iters each = ~90s billed QPU budget).
# Override with SEEDS="42" MAX_ITERS=5 for a smoke test.
slurm-ibm-seeds:
	@[ -f .env ] || (echo "ERROR: .env not found"; exit 1)
	@mkdir -p results/slurm
	bash scripts/submit_ibm_seeds.sh

# Aggregate seeded results into median +/- range statistics (at P=2 by default).
aggregate-seeds:
	python3 benchmarks/aggregate_seeds.py

# Build strong-scaling table from scaling sweep JSONs (filtered to seed=42).
aggregate-scaling:
	python3 benchmarks/aggregate_scaling.py


# Commit + push results/ so a run's output survives a local disk failure.
# Safe to run after any make target above -- no-ops cleanly if nothing changed.
# Does NOT touch checkpoints/ (deliberately gitignored -- ephemeral,
# regenerate-on-crash by rerunning rather than resuming) or .env (secrets).
backup-results:
	@git add results/
	@if git diff --cached --quiet -- results/; then \
	  echo "[Make] backup-results: nothing new under results/ to commit."; \
	else \
	  git commit -m "results: backup $$(date -u +%Y-%m-%dT%H:%M:%SZ)" -- results/; \
	  echo "[Make] backup-results: committed."; \
	fi
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	if git rev-parse --abbrev-ref --symbolic-full-name "$$branch@{upstream}" > /dev/null 2>&1; then \
	  git push; \
	  echo "[Make] backup-results: pushed $$branch."; \
	else \
	  echo "[Make] backup-results: '$$branch' has no upstream yet -- run 'git push -u origin $$branch' once to enable auto-push here."; \
	fi