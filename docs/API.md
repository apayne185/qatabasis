# API Reference

Public API for the hybrid quantum-classical VQE middleware. This document
covers the Python entry points, environment-variable configuration contract,
and extension points for adding new molecules, backends, and ansatzes.

For a task-oriented walkthrough (deploy on a laptop, run on a cluster,
extend with a new molecule), see the **Tutorials** section at the bottom.

**Table of contents:**

- [Configuration contract (environment variables)](#configuration-contract-environment-variables)
- [Core classes](#core-classes)
  - [`QatabasisStack`](#qatabasisstack)
  - [`HardwareProfile`](#hardwareprofile)
  - [`ChemistryProblem`](#chemistryproblem)
  - [`MoleculeResolver`](#moleculeresolver)
  - [`FinanceProblem`](#financeproblem)
- [Result artefacts](#result-artefacts)
  - [JSON output schema](#json-output-schema)
  - [Aggregators](#aggregators)
- [Extension points](#extension-points)
  - [Add a new molecule](#add-a-new-molecule)
  - [Add a new ansatz tier](#add-a-new-ansatz-tier)
  - [Add a new quantum backend](#add-a-new-quantum-backend)
- [Tutorials](#tutorials)

---

## Configuration contract (environment variables)

Every runtime decision the stack makes — whether to use GPU, which precision
to run at, how many SPSA iterations, which molecule subset — is controlled by
environment variables. This is the primary user-facing API: no code changes
are needed to switch hardware, backends, or workloads.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `USE_GPU` | `yes` \| `no` | auto-detect | Force GPU on or off. `HardwareProfile.detect()` decides otherwise |
| `BACKEND` | `simulator` \| `ibm_cloud` | `simulator` | Simulator vs cloud QPU |
| `VQE_PRECISION` | `auto` \| `fp32` \| `fp64` | `auto` (fp64 on datacenter GPU / ≥20 qubits, fp32 on consumer/workstation < 20 qubits) | Statevector precision |
| `VQE_BACKEND` | `auto` \| `custatevec` \| `aer_gpu` \| `aer_cpu` | `auto` | Force a specific simulator backend |
| `MOLECULES` | space-separated names | `H2 LiH BeH2 H2O` | Subset of `MOLECULE_REGISTRY` to run in `local_test_run.py` |
| `SEED` | integer | `42` | SPSA random seed (override for multi-seed runs) |
| `MAX_ITERS` | integer | auto (`max(200, 8 × n_params)`) | Cap on iterations per molecule (used for hardware-ceiling tests) |
| `NP` | integer | `2` | MPI rank count (also settable via `sbatch --ntasks=N`) |
| `IBM_QUANTUM_TOKEN` | string | (required for `ibm_cloud`) | IBM Quantum API token |
| `IBM_QUANTUM_INSTANCE` | CRN string | (required for `ibm_cloud`) | From the SAME account as the token |
| `IBM_QUANTUM_BACKEND` | backend name | (required for `ibm_cloud`) | e.g. `ibm_marrakesh`, `ibm_torino`, `ibm_kyiv` |
| `VQE_LEGACY_EXPECT` | `1` \| unset | unset | Force the legacy CPU-side expectation path (GPU→CPU statevector copy + numpy) even at `NP=1`, where the GPU-native path is normally faster. Useful for A/B measurement or regression checks. |
| `VQE_GPU_EXPECT_MPI` | `1` \| unset | unset | Force the GPU-native expectation path (`save_expectation_value`) even under MPI at `NP>=2`, where it is **not** the default due to a measured 3.6x per-iteration regression (per-rank Aer transpile cost amplifies when multiple ranks contend for one GPU). Only for validating a future fix — do not use for published results until that regression is resolved. |

**GPU-native vs. legacy expectation routing**: the stack picks between two
ways of computing Pauli expectation values on GPU (`_expectation_on_gpu` vs.
the legacy build-statevector-then-numpy path in
[`interface.py`](../src/api/interface.py)). By default it uses the faster
GPU-native path at `NP=1` (~1.5x speedup, validated) and falls back to the
legacy path at `NP>=2` (the GPU-native path regresses under MPI contention
for the same GPU). See [`GPU_EXPECTATION_FIX.md`](GPU_EXPECTATION_FIX.md)
for the full measurement and rationale. The two env vars above override
this default for A/B testing.

Example — run only H₂O on GPU with 4 ranks, seed 43:

    USE_GPU=yes MOLECULES="H2O" SEED=43 make run NP=4

Example — force fp32 precision on a datacenter GPU:

    VQE_PRECISION=fp32 make run

Example — IBM Quantum run with credentials loaded from `.env`:

    make run-ibm NP=2

---

## Core classes

### `QatabasisStack`

Main entry point for VQE runs. Manages MPI initialization, hardware
detection, checkpoint I/O, SPSA optimizer state, and dispatch to the
correct backend (statevector CPU, statevector GPU via cuStateVec, or
IBM Quantum QPU).

Location: [`src/api/interface.py`](../src/api/interface.py)

#### Constructor

    QatabasisStack(use_gpu: bool | None = None, backend: str = 'simulator')

**Parameters:**

- `use_gpu` (`bool | None`) — Force GPU (`True`), force CPU (`False`),
  or auto-detect (`None`, default). Auto-detection consults `HardwareProfile`
  and the `USE_GPU` env var. If `hpc_core` was compiled without CUDA, a
  `True` value is downgraded to `False` with a warning.
- `backend` (`str`) — `'simulator'` (default) for local/GPU statevector
  simulation, or `'ibm_cloud'` for IBM Quantum QPU submission via
  Qiskit IBM Runtime EstimatorV2.

**Initializes:**

- MPI (`MPI_Init_thread` via `hpc_core.init_mpi()`, then `mpi4py`)
- CUDA device (round-robin `rank % gpu_count`)
- Qiskit Aer GPU probe (tests `AerSimulator(method='statevector', device='GPU')`)
- Prints the `HardwareProfile.describe()` banner on rank 0

**Attributes populated:**

- `hw: HardwareProfile` — the detected hardware profile
- `use_gpu: bool` — final GPU decision after fallbacks
- `backend: str` — as passed
- `precision: str` — resolved per-problem in `vqe_optimize()`
- `comm: MPI.Comm` — `MPI.COMM_WORLD`
- `rank: int` — this rank's ID
- `size: int` — total ranks
- `_gpu_sv: bool` — whether cuStateVec statevector is available

#### `vqe_optimize(problem, ...) -> tuple[np.ndarray, list[float]]`

Runs the SPSA optimization loop for a given problem.

    vqe_optimize(
        problem: QuantumProblem,
        max_iterations: int = 100,
        tolerance: float = 1.6e-3,
        restart_from: str | None = None,
        checkpoint_dir: str = "checkpoints",
        start_iter: int = 0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[float]]

**Parameters:**

- `problem` — instance of `QuantumProblem` (typically `ChemistryProblem` or `FinanceProblem`)
- `max_iterations` — hard cap on SPSA iterations
- `tolerance` — convergence threshold in Ha (default = chemical accuracy)
- `restart_from` — path to a `.npy` checkpoint to resume from
- `checkpoint_dir` — where to write incremental theta checkpoints
- `start_iter` — SPSA schedule offset (auto-inferred from checkpoint filename)
- `seed` — SPSA random seed (overrides `SEED` env var if set)

**Returns:**

- `theta` (`np.ndarray`) — final variational parameters
- `history` (`list[float]`) — per-iteration energy trajectory

**Side effects:**

- Writes `checkpoint_iter_XXXX.npy` every 5 iterations, keeps the last 5
- Broadcasts theta across MPI ranks via `comm.Bcast`
- Aggregates partial energies via `MPI_Allreduce(SUM)`
- Tracks best-physical-energy (below-FCI mitigation for HWE ansatz)

#### `finalize()`

Cleanly closes IBM Runtime session (if open) and calls `hpc_core.finalize_mpi()`.
Called automatically in the `__exit__` handler when used as a context manager.

**Usage pattern (recommended):**

    with QatabasisStack(backend='simulator') as stack:
        theta, history = stack.vqe_optimize(problem, max_iterations=200, seed=42)
    # finalize() runs automatically here

---

### `HardwareProfile`

Immutable-after-detection dataclass that probes the runtime environment
(GPU vendor/class/memory, MPI availability, installed libraries) and
exposes policy decisions to the rest of the stack.

Location: [`src/api/hardware.py`](../src/api/hardware.py)

#### `HardwareProfile.detect() -> HardwareProfile`

Class method. Runs all probes and returns a populated instance. Called
once by `QatabasisStack.__init__`.

Probes:

- `nvidia-smi --query-gpu=name,memory.total,compute_cap` (3 s timeout)
- Attempted `AerSimulator(method='statevector', device='GPU')` construction
- `cuquantum` import check
- `mpi4py.MPI.Is_initialized()` and `.Get_size()` if so

#### Policy methods

    want_gpu() -> bool
        # True iff a CUDA device exists and USE_GPU != 'no'

    recommend_precision(num_qubits: int) -> str
        # 'fp32' | 'fp64' — see docstring for rules

    recommend_backend() -> str
        # 'custatevec' | 'aer_gpu' | 'aer_cpu'

    max_qubits_fit(precision: str = 'fp64', mpi_size: int = 1) -> int
        # Upper bound on qubit count that fits in GPU memory.
        # mpi_size matters: round-robin CUDA device assignment means N ranks
        # on 1 GPU each build their OWN full statevector -> budget divided by N.

    describe() -> str
        # Human-readable summary printed at stack init.
        # Example: "[hw] NVIDIA A100-SXM4-40GB (datacenter, 40.0 GB, fp64:fp32≈0.500)
        #          | libs: aer-gpu, cuquantum | MPI=2 | backend=custatevec"

#### GPU database

`_GPU_DATABASE` in `hardware.py` maps GPU-name substrings to
`(class, fp64_ratio)` tuples. Currently covers: A100, H100, V100, A40,
RTX 6000 Ada, RTX A6000, RTX 4090/4080/3090/3080, GTX 1650/1660.

To add a new GPU, append a tuple like:

    ("H200", "datacenter", 1 / 2),

and it will be recognised on next `detect()` call.

---

### `ChemistryProblem`

Encapsulates a molecular Hamiltonian and its variational ansatz. Wraps
PySCF (for the electronic-structure problem) and Qiskit Nature (for
Jordan-Wigner mapping to a qubit Hamiltonian).

Location: [`src/api/problems.py`](../src/api/problems.py)

#### Constructor

    ChemistryProblem(
        atom_coordinates: str,
        reps: int = 1,
        name: str = "custom",
        force_tier: str | None = None,
    )

**Parameters:**

- `atom_coordinates` — PySCF geometry string (e.g. `"H 0 0 0; H 0 0 0.74"`)
- `reps` — HWE ansatz depth
- `name` — used for checkpoint directory naming
- `force_tier` — override auto-selected ansatz tier (`'hwe'`, `'hwe_adaptive'`, `'uccsd'`)

#### Class method

    ChemistryProblem.from_name(molecule_name: str, force_tier=None) -> ChemistryProblem

Look up a built-in molecule from `MOLECULE_REGISTRY` by name.
Registered names: `H2`, `LiH`, `BeH2`, `H2O`, `NH3`, `N2`, `CO2`.
See [`MOLECULES.md`](../MOLECULES.md) for full metadata.

#### `prepare()`

Populates `pauli_terms`, `num_qubits`, `num_params`, `ansatz_circuit`,
`fci_energy`, and `diagnostics`. Idempotent (safe to call multiple times;
does nothing after the first).

Uses PySCF's `PySCFDriver` for the electronic-structure problem, then
`JordanWignerMapper` for the qubit mapping. FCI is computed via
PySCF's `fci.FCI` solver — this becomes the ground-truth reference.

#### Attributes (after `prepare()`)

- `pauli_terms: list[tuple[str, float]]` — Hamiltonian as list of Pauli strings + coeffs
- `num_qubits: int`
- `num_params: int`
- `ansatz_circuit: QuantumCircuit`
- `fci_energy: float` — reference ground-state energy (Ha)
- `ansatz_tier: str` — `'hwe'`, `'hwe_adaptive'`, or `'uccsd'`
- `diagnostics: dict` — correlation score + reasoning
- `circuit_qasm: str` — serialized QASM 3 for the C++ dispatcher

---

### `MoleculeResolver`

Cascade resolver that turns arbitrary molecule input into a validated
geometry + electron count + qubit estimate. Tries in order:

1. Local `MOLECULE_REGISTRY` lookup
2. Raw geometry string (heuristic: matches `[A-Z][a-z]?\s+[-\d.]`)
3. SMILES notation (RDKit — optional dependency)
4. PubChem online lookup (with local cache)

Location: [`src/api/molecule_resolver.py`](../src/api/molecule_resolver.py)

#### Constructor

    MoleculeResolver(
        max_qubits: int = 20,
        basis: str = "sto-3g",
        allow_network: bool = True,
        cache_dir: str | None = None,
    )

Raise `MoleculeTooBigError` for anything exceeding `max_qubits` after
active-space reduction. `local_test_run.py` uses `max_qubits=30` to
accommodate ceiling tests up to CO₂.

#### `resolve(molecule_input, freeze_core=True) -> ResolutionResult`

Returns a `ResolutionResult` dataclass with `geometry`, `source`,
`total_electrons`, `active_electrons`, `estimated_qubits`, `freeze_core`,
and metadata.

#### `resolve_batch(molecules: list[str], freeze_core=True) -> dict[str, ResolutionResult | None]`

Batch version. Failed resolutions become `None` in the returned dict
rather than raising.

---

### `FinanceProblem`

Portfolio optimization problem mapped to a QUBO/Ising Hamiltonian.
Kept for testing the middleware's non-chemistry extensibility.

Location: [`src/api/problems.py`](../src/api/problems.py)

    FinanceProblem(
        covariance: np.ndarray,
        expected_returns: np.ndarray,
        risk_factor: float = 1.0,
    )

Same `.prepare()` / `.vqe_optimize()` interface as `ChemistryProblem`.
Ansatz defaults to `TwoLocal` (RY + RZ rotations, CX entanglement).

---

## Result artefacts

### JSON output schema

Every run writes a timestamped JSON to
`results/<hardware-slug>/<category>/<backend>_<YYYYMMDD_HHMMSS>.json`.
`<hardware-slug>` is derived from `nvidia-smi` by
`HardwareProfile.results_slug()` (e.g. `a100-sxm4-40gb`,
`rtx-6000-ada-generation`, or `cpu-only` with no GPU) — results from
different hardware are always routed to sibling folders, never mixed. See
[`results/README.md`](../results/README.md) for the full layout and the
per-hardware folder index.

**Simulator output** (`results/<hardware-slug>/simulator/simulator_*.json`):

    {
      "timestamp": "2026-08-06T14:00:03",
      "backend": "simulator",
      "git_commit": "abc1234",
      "mpi_ranks": 2,
      "gpu": true,
      "seed": 42,
      "gpu_name": "NVIDIA A100-SXM4-40GB",
      "gpu_class": "datacenter",
      "hostname": "129-213-151-110",
      "molecules": {
        "H2": {
          "energy": -1.134896,
          "unperturbed_energy": -1.134901,
          "fci": -1.13727,
          "iters": 200,
          "wall_time": 1.4,
          "history": [-0.037, -0.034, ...],
          "tier": "hwe_adaptive",
          "score": 0.34
        },
        ...
      },
      "scaling": {"ranks": 2, "wall_time": 0.68, ...},
      "weak_scaling": {"ranks": 2, "wall_time": 1.05, ...}
    }

**`energy` vs. `unperturbed_energy` — which one to cite for accuracy claims:**

- `energy` is the value SPSA itself used internally: `(E(θ+cₖδ) + E(θ-cₖδ)) / 2`
  at whichever iteration was selected for reporting (either the final iteration,
  or `_best_physical_energy` if the trajectory ended below the FCI reference).
  This is the correct quantity for driving the optimizer, but it carries a small,
  non-vanishing bias of order `cₖ²·tr(Hessian)/2` relative to the true energy at
  that point — `cₖ` decays too slowly (`gamma=0.101`) to make this negligible
  over a realistic run. **Do not cite `energy` alone as a chemical-accuracy
  number without also reporting `unperturbed_energy`.**
- `unperturbed_energy` is a single, separate, exact statevector evaluation of
  `E(θ)` at the same `θ` that produced `energy` — computed once via
  `QatabasisStack.evaluate_unperturbed_energy()` after the SPSA loop finishes,
  not per-iteration. This is the reproducible number to use for any accuracy
  claim against FCI. It costs one extra statevector build per molecule (not
  per iteration), so it is cheap even for the larger benchmark molecules.
- Both fields report the energy at the *same* `θ` — the only difference is
  the perturbed-average vs. unperturbed evaluation of that one point. Neither
  field is "the converged answer": both are still subject to whatever
  iteration/seed selected `θ` in the first place (see `docs/API.md`'s
  Aggregators section and `benchmarks/aggregate_seeds.py` for the multi-seed
  statistics this feeds into).

**IBM output** (`results/<hardware-slug>/ibm/ibm_cloud_*.json`):

    {
      "timestamp": "2026-07-27T22:01:30",
      "backend": "ibm_cloud",
      "mpi_ranks": 2,
      "gpu": true,
      "seed": 42,
      "max_iters": 10,
      "chemistry": {
        "molecule": "H2",
        "energy": -0.0987,
        "fci": -1.1373,
        "error": 1.0386,
        "iterations": 10,
        "wall_time": 2957.9,
        "time_per_iter": 295.79,
        "history": [-0.191, -0.175, ...]
      }
    }

**Provenance fields** (`gpu_name`, `gpu_class`, `hostname`, `git_commit`)
are stamped by `src/api/results.py:save_results()` so a JSON file remains
self-identifying even if moved between filesystems.

### Aggregators

    python3 benchmarks/aggregate_seeds.py                # median + range across seeds at P=2
    python3 benchmarks/aggregate_seeds.py --backend ibm  # same for IBM QPU runs
    python3 benchmarks/aggregate_scaling.py              # strong-scaling table (seed=42)

Both scripts filter by `--hw <hardware-slug>` to prevent mixing runs
across GPU classes (e.g. `--hw a100-sxm4-40gb` vs `--hw rtx-6000-ada-generation`).

---

## Extension points

### Add a new molecule

Two ways.

**Option 1 — registry entry** (permanent, checked into git):

Edit `src/api/problems.py:MOLECULE_REGISTRY`:

    "N2": {
        "geometry": "N 0 0 0; N 0 0 1.09",
        "fci_energy": -108.9544,
        "reps": 2,
        "description": "Nitrogen (N2), 14 electrons, 20 qubits",
    },

Then `ChemistryProblem.from_name("N2")` works, and `N2` becomes valid
in the `MOLECULES` env var: `MOLECULES="N2" make run`.

**Option 2 — ad-hoc via MoleculeResolver** (SMILES, PubChem, or raw geometry):

    from src.api.molecule_resolver import MoleculeResolver
    resolver = MoleculeResolver(max_qubits=30)
    result = resolver.resolve("CCO")                # SMILES for ethanol
    problem = result.to_chemistry_problem()

### Add a new ansatz tier

Extend `build_ansatz()` in `src/api/problems.py`. The current tier system
routes on the correlation score returned by `estimate_correlation_strength()`:

    if score < 0.25:  tier = "hwe"           # weak correlation, HWE is fine
    elif score < 0.55: tier = "hwe_adaptive" # moderate correlation
    else:              tier = "uccsd"        # strong correlation, UCCSD needed

To add a new tier (e.g. `"qcc"` for Qubit Coupled Cluster), add a branch
to `build_ansatz()` and update `ANSATZ_TIERS` metadata dict at the top of
the file.

### Add a new quantum backend

Currently the backend contract is a single `if/elif` chain in
`QatabasisStack.vqe_optimize()`:

    if self.backend == "simulator":
        e_plus, e_minus, M, path = self._evaluate_distributed_statevector(...)
    elif self.backend == "ibm_cloud":
        e_plus, e_minus, M, path = self._evaluate_ibm_estimator(...)

To add e.g. Amazon Braket or IonQ, add:

1. A new `_evaluate_<vendor>_...` method with the same signature
2. A new `elif self.backend == "<vendor>"` branch
3. A new `_init_<vendor>_session()` method for one-time setup
4. Optional: extend `HardwareProfile` to detect the vendor's SDK

A future refactor would extract these into a `QPUBackend` `Protocol` +
registry pattern; see [`docs/FUTURE_WORK.md`](FUTURE_WORK.md) for details.

---

## Tutorials

### Tutorial 1 — Run a benchmark on your laptop (Docker, no GPU required)

    git clone https://github.com/apayne185/qatabasis.git
    cd qatabasis
    make build              # ~10 min first time
    make trial NP=2         # ~5 min, expect: Tests passed: 7 / 7
    make run NP=2           # full 4-molecule benchmark

Output lands in `results/cpu-only/simulator/simulator_<timestamp>.json` (or
`results/<hardware-slug>/simulator/...` if a GPU is detected). On a laptop
with no CUDA-capable GPU, the stack transparently falls back to
`aer_cpu` — the run completes but is slower.

### Tutorial 2 — Same code on a cloud GPU

Same commands, different host. On Lambda Cloud or any host with the
NVIDIA Container Toolkit installed:

    ssh ubuntu@<gpu-instance-ip>
    git clone https://github.com/apayne185/qatabasis.git
    cd qatabasis
    sudo make build
    sudo make trial NP=2

The Makefile's Docker invocation includes `--gpus all` when it detects
a GPU. `HardwareProfile` picks up the GPU class and enables
`cuStateVec` automatically.

### Tutorial 3 — Multi-seed statistical run

    for s in 42 43 44 45 46; do
        SEED=$s make run NP=2
    done
    python3 benchmarks/aggregate_seeds.py

Prints a median + [min, max] table across seeds. Use this pattern for
publication-grade statistics.

### Tutorial 4 — Hardware ceiling test (bigger molecule than the defaults)

    # Registry already has N2 (20q) and CO2 (30q)
    MOLECULES="CO2" MAX_ITERS=10 make run NP=1

`NP=1` matters at 30 qubits: at fp64 the statevector is 16 GB, and two
ranks sharing one 40 GB GPU (default `NP=2`) would each build their own
full copy (redundant, OOM-inducing). The `MPI-aware max_qubits_fit()`
check will warn if you exceed the rank-adjusted budget.

### Tutorial 5 — Cloud QPU (IBM Quantum)

    # First: put valid IBM credentials in .env
    cp .env.example .env
    # Edit .env: set IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE
    # (both must be from the SAME IBM Cloud account)

    make run-ibm NP=2

Uses EstimatorV2 with `mode=backend` (open-plan compatible),
4096 shots, T-REx measurement error mitigation. Results go to
`results/<hardware-slug>/ibm/ibm_cloud_<timestamp>.json`.

### Tutorial 6 — Distributed via Slurm on an HPC cluster

    # One-time install on cluster:
    bash scripts/install_native.sh

    # Submit jobs:
    make slurm-trial            # 7-layer diagnostic on 1 GPU node
    make slurm-run              # full benchmark
    make slurm-scaling          # strong scaling sweep P=1,2,4,8
    make slurm-multi-seed       # 3-seed statistical run

`scripts/slurm_*.sh` are tested on Debian 13 + Slurm + CUDA 12.4.
Adapt partition names, time limits, and CUDA paths for your site.
