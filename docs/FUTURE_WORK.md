# Future Work

This document expands on the "Future Work" section of the README with detailed
technical explanations of the deferred features. Each item explains:

- **What** is deferred and why it matters
- **Why not now** (technical blocker or scope decision)
- **What would be needed** to close the gap

The paper-ready single-paragraph versions are marked `<!-- paper -->` for
direct citation in the manuscript.

**Contents:**

1. [Particle-conserving ansatzes (UCCSD)](#1-particle-conserving-ansatzes-uccsd) — highest priority
2. [Distributed statevector (multi-node MPI + cuStateVec tiling)](#2-distributed-statevector)
3. [Advanced error mitigation (ZNE, PEC)](#3-advanced-error-mitigation)
4. [QPU backend plugin system](#4-qpu-backend-plugin-system) (IonQ, Rigetti, Braket, Azure)
5. [Application extensibility](#5-application-extensibility) (materials, combinatorics)
6. [Formal profiling data](#6-formal-profiling-data) (Nsight/nvprof)

---

## 1. Particle-conserving ansatzes (UCCSD)

### What is deferred

The Unitary Coupled-Cluster Singles-and-Doubles (UCCSD) ansatz preserves
particle number by construction: its excitation operators map an
N-electron state to an N-electron state, never a superposition of
different N. The Hardware-Efficient Ansatz (HWE) used throughout this
work has no such constraint — during SPSA optimization it can drive the
prepared state into unphysical Hilbert-space sectors with the wrong
particle count.

### Why it matters

This directly limits the chemistry accuracy of the reported results:

| Molecule | Params | Best-seed error | Median-seed error (n=5 or n=10) |
|---|---:|---:|---:|
| H₂ | 24 | 0.2 mHa | 2.4 mHa |
| LiH | 96 | 0.6 mHa | 492 mHa (n=10, bimodal) |
| BeH₂ | 112 | 0.7 mHa | 3.4 mHa |
| H₂O | 112 | 96 mHa | 220 mHa |

The best-of-N seed reaches or approaches chemical accuracy (1.6 mHa) for
H₂, LiH, and BeH₂. The **median** convergence is much worse — clear
evidence of particle-number violation over the SPSA trajectory. UCCSD
would eliminate this failure mode by construction.

### Why not now — the technical blocker

UCCSD integration was attempted during the thesis phase and encountered
a concrete compatibility issue with Qiskit Nature 0.7's
`UCCSD` circuit library:

The UCC amplitude parameterization uses **tied parameters** — the same
variational parameter appears in multiple gates within one excitation
term (this is what enforces the particle-conservation symmetry). Qiskit
Nature builds this correctly at the abstract `QuantumCircuit` level.

However, the stack's statevector evaluation path calls
`transpile()` on the ansatz to decompose it into hardware-native gates.
Qiskit's transpile pipeline **breaks the tied-parameter structure**
during optimization: it treats each occurrence of the parameter as
independent, producing an unrolled circuit that no longer conserves
particle number when the parameters are bound to concrete values.

Grepping the entire result-log history of this repo:

    grep -r "UCCSD built" results*/  # returns nothing

confirms that no run has ever successfully exercised the UCCSD tier
end-to-end — the code path structurally exists in `build_ansatz()`
(`src/api/problems.py`) but has never produced a valid VQE trajectory.

### Impact scope: this is NOT a middleware issue

The middleware contributions of this work — hardware auto-detection,
MPI-distributed Pauli-term evaluation, masking metric, async QPU dispatch,
cloud-portable deployment — are **orthogonal** to the ansatz choice. UCCSD
is a drop-in replacement in the existing ansatz tier system; when the
transpile-decomposition issue is fixed, no other stack code needs to
change. Chemistry accuracy improvements from UCCSD would be delivered by
the same middleware, on the same hardware, with the same reproducibility
guarantees.

### What would be needed

Three plausible fixes, from lightest to heaviest:

1. **Explicit Trotterization with manual parameter tying.** Bypass
   Qiskit's `UCCSD` circuit library entirely: build the exponential
   `exp(-i * θ_j * T_j)` for each excitation `T_j` as a hand-rolled
   sequence of Pauli rotations, tying the θ vector at construction
   time so `transpile()` sees only one parameter per excitation.
   ~1-2 weeks of work, no upstream changes needed.

2. **Skip transpile for the ansatz.** For statevector simulation there
   is no coupling-map / native-gate constraint — the transpile pass is
   only useful for real hardware. If UCCSD's abstract circuit runs
   directly through `AerSimulator.run()` without transpilation, the
   parameter ties may survive. Cheapest option, needs experimental
   verification that Aer accepts the un-transpiled UCC gates.

3. **Upstream fix in Qiskit Nature.** File an issue against qiskit-nature
   about UCCSD + transpile parameter-tie loss. Waiting on this is
   unbounded; not a viable primary plan.

### Suggested paper text

<!-- paper -->

> "The Hardware-Efficient Ansatz (HWE) used throughout this work does not
> conserve particle number, permitting the SPSA optimizer to explore
> unphysical Hilbert-space sectors. This manifests as final energies
> falling below the Full Configuration Interaction reference at moderate
> parameter counts, evident in LiH's 492 mHa bimodal spread across n=10
> seeds (best-of-10 seed reached 0.6 mHa error; median 492 mHa error).
> The Unitary Coupled-Cluster Singles-and-Doubles (UCCSD) ansatz conserves
> particle number by construction and would eliminate this failure mode.
> UCCSD integration was attempted but encountered a compatibility issue
> with Qiskit Nature 0.7's UCCSD circuit library: the tied-parameter
> structure of UCC amplitudes does not survive Qiskit's transpile
> pipeline, producing an unrolled circuit that no longer conserves
> particle number when parameters are bound. The stack retains UCCSD in
> its ansatz tier system; resolution — via explicit Trotterization with
> manual parameter tying, or by bypassing transpilation on the statevector
> path — is a priority follow-up. Until then, all chemistry results in
> this work use HWE with best-physical-energy tracking as a mitigation.
> The middleware contributions reported here are orthogonal to the ansatz
> choice: UCCSD is a drop-in replacement in the existing tier system when
> the transpilation issue is resolved."

<!-- /paper -->

---

## 2. Distributed statevector

### What is deferred

The current stack distributes **Pauli-term evaluations** across MPI ranks,
but each rank independently constructs the full 2ⁿ statevector. Two
higher-scale distribution modes are missing:

1. **Multi-node MPI**: all reported scaling data is single-host
   (multiple ranks on one machine, sharing CPU cache/memory bandwidth
   or one GPU). True multi-node MPI over InfiniBand-interconnected
   nodes has not been demonstrated.

2. **Statevector tiling**: splitting the 2ⁿ amplitudes across GPUs
   (via cuStateVec's `custatevecMultiGPU_*` API), so each GPU stores
   only 2ⁿ/P amplitudes and gates are applied via distributed
   collectives. This lifts the qubit ceiling from ~30 (single 40 GB
   A100) to ~34 on an 8×A100 node, and higher across multiple nodes.

### Why it matters

Without statevector tiling, the per-rank memory cost is unchanged by
adding ranks — you can't run bigger molecules by adding hardware, only
by getting a bigger single GPU. This bounds the science story of the
paper at 30 qubits (CO₂ on 40 GB A100). Statevector tiling would enable
40+ qubit molecules on 8×A100/8×H100 nodes.

**Measured, not hypothetical — Aer's single-GPU `blocking_enable` mode is
not the answer.** A 2026-09-01 A100 sweep tested whether Qiskit Aer's own
distributed-statevector mode (`AerSimulator(blocking_enable=True)`) closes
this gap without a stack rearchitecture. It does not: `aer-mpi` lost to the
current replicated-SV design at every tested size from H2 (4q) through N2
(20q), and the gap widens exponentially with qubit count (aer-mpi was 8.6x
*slower* than hpchybrid at N2). See
[`GPU_EXPECTATION_FIX.md`](GPU_EXPECTATION_FIX.md)'s "Crossover measurement"
section for the full table. The reason is architectural: Aer's blocking
mode trades single-kernel-launch efficiency for communication overhead
*within the same physical GPU*, which loses for Pauli-heavy VQE workloads
at any tested size. This means the fix described below — genuine
**multi-GPU** cuStateVec tiling, where each GPU holds only 2ⁿ/P amplitudes
— remains the correct target; a config-only fix (routing to Aer's existing
blocking mode above some qubit threshold) was tested and ruled out.

Multi-node MPI validation matters separately for the "HPC middleware"
claim: currently the paper says "distributed VQE" but demonstrates only
single-host multi-rank runs. A reviewer will ask whether the code
actually handles cross-node network transport correctly.

### Why not now

- **Multi-node hardware access**: Lambda 1-Click Clusters cost ~$8/hr
  for a 2-node 2×A100 setup; university HPC allocations require
  administrative overhead. The IE University capstone cluster has one
  compute node (`haskell`), so multi-node was structurally impossible
  there.
- **cuStateVec multi-GPU API**: not a drop-in replacement — requires
  a substantial refactor of `_build_statevector()` to call
  `custatevecDistIndexBitSwapScheduler_*` and handle the tiled state
  during Pauli evaluation. ~2-4 weeks of work.

### What would be needed

For multi-node validation (cheap):

- 2-hour rental of a Lambda 1-Click Cluster (2 × A100) = ~$16
- Run `mpirun -n 2 --host node1,node2 python benchmarks/local_test_run.py`
- Capture one wall-time number, confirm no deadlock. Paper claim
  becomes: "validated on multi-node InfiniBand-interconnected GPU
  cluster."

For statevector tiling (expensive):

- Rewrite `_build_statevector()` to use cuStateVec's multi-GPU API
- Modify Pauli-term evaluation to work on tiled amplitudes
  (each rank evaluates its Pauli subset against its statevector tile;
  MPI_Allreduce as today)
- Regression test against single-GPU results at ≤30 qubits
- Benchmark at 32-40 qubits on multi-GPU nodes

### Suggested paper text

<!-- paper -->

> "The current implementation distributes Pauli-term evaluations across
> MPI ranks while each rank independently constructs the full 2ⁿ
> statevector. This limits the maximum tractable molecule to what fits
> in a single GPU's memory (approximately 30 qubits on the tested 40 GB
> A100 SXM4). Distributed-statevector simulation via cuStateVec's
> multi-GPU API would tile the 2ⁿ amplitudes across GPUs, lifting the
> ceiling to ~34 qubits per 8-GPU node and enabling larger molecules
> (e.g. formaldehyde, benzene fragments) with correspondingly better
> chemistry. Multi-node MPI validation on InfiniBand-interconnected
> hardware — planned on a Lambda 1-Click Cluster — will separately
> address the current single-host constraint on the reported scaling
> data."

<!-- /paper -->

---

## 3. Advanced error mitigation

### What is deferred

The IBM Quantum QPU path currently uses **T-REx** (Twirled Readout
Error eXtinction), Qiskit's resilience level 1. T-REx only corrects
measurement (readout) errors — it does nothing about gate errors or
decoherence during circuit execution.

Missing: **Zero Noise Extrapolation** (ZNE, resilience level 2) and
**Probabilistic Error Cancellation** (PEC, resilience level 3), both
of which target the dominant error sources on today's superconducting
QPUs.

### Why it matters

The H₂ QPU run in this work reached 1.04 Ha error vs FCI. Roughly:

- ~0.02 Ha of that is shot noise (from 4096 shots)
- ~1.0 Ha is gate errors + decoherence over the HWE circuit's ~30 CNOTs

ZNE alone typically closes 30-60% of the gate-error gap. Combined with
UCCSD (item 1), a well-mitigated QPU run of H₂ could realistically
reach ~50 mHa error — still not chemical accuracy but qualitatively
more useful.

### Why not now

- Resilience level 2 in `qiskit-ibm-runtime` requires more QPU shots
  per iteration (typically 4-8× at the same accuracy) — for our
  10-minute Open Plan budget this shrinks the accessible iteration
  count from 10 to 1-2.
- PEC additionally requires a per-backend noise model characterization
  (~30-60 minutes of QPU calibration time), which the Open Plan does
  not support.

Bumping resilience level is a one-line change in `_init_ibm_session()`:

    self._ibm_estimator.options.resilience_level = 2

but doing so responsibly requires either a paid IBM plan or a
dedicated evaluation budget.

### What would be needed

- IBM Premium / Startup Plan (~$1500/mo for hobby usage) OR
- Academic time allocation via IBM Quantum Network membership
- ~30 min of QPU characterization for PEC
- Rerun the H₂ QPU benchmark at resilience level 2 (~10 min execution)

---

## 4. QPU backend plugin system

### What is deferred

Currently `QatabasisStack.vqe_optimize()` dispatches by a hard-coded
`if/elif` chain:

    if self.backend == "simulator":  ...
    elif self.backend == "ibm_cloud":  ...

Adding a new QPU vendor (IonQ, Rigetti, Amazon Braket, Azure Quantum,
Quantinuum) requires editing this dispatch, adding a new `_evaluate_*`
method, adding a new `_init_*_session()` method — invasive changes
to a central class.

### Why it matters

The "hardware-agnostic middleware" claim of the paper is limited by
this — the stack is genuinely portable across CPU/GPU/IBM, but adding
a new QPU vendor is a several-hundred-LOC change scattered across the
central `interface.py`. A proper plugin architecture would let
researchers add new QPUs in one small file each.

### Why not now

Refactoring to a `QPUBackend` `Protocol` + registry is straightforward
(~1-2 weeks), but has zero user-facing benefit until at least one
non-IBM backend is added. For this paper, IBM is the only QPU tested.

### What would be needed

1. Extract a `QPUBackend` protocol (`src/api/qpu_backend.py`):

        class QPUBackend(Protocol):
            name: str
            def connect(self, credentials: dict) -> None: ...
            def transpile(self, circuit) -> Any: ...
            def submit(self, circuit, observable, params) -> "Job": ...
            def wait(self, job) -> tuple[float, float]: ...

2. Refactor existing IBM code into `src/api/qpu_plugins/ibm.py`
3. Registry: `REGISTRY = {"ibm_cloud": IBMBackend, ...}`
4. `QatabasisStack.__init__` looks up backend via registry

Once done, adding IonQ (via `qiskit-ionq` or Braket SDK) is a ~100 LOC
file. A follow-up paper could demonstrate "same code, four vendors."

---

## 5. Application extensibility

### What is deferred

The stack includes `FinanceProblem` (portfolio QUBO) as evidence that
the middleware isn't chemistry-specific. Two obvious next-application
domains are not yet demonstrated:

- **Combinatorial optimization**: MaxCut, TSP, graph coloring — all
  reducible to QUBO/Ising Hamiltonians
- **Materials / condensed matter**: periodic-boundary Hamiltonians,
  Hubbard-model VQE

Both would use the same MPI distribution and QPU dispatch — the
existing stack could run them today with only a new `Problem` subclass.

### Why not now

Scope. The paper's chemistry story is already substantial; adding two
more application domains would fragment the narrative. Better as
follow-up work with its own paper positioning.

### What would be needed

For each new domain: a subclass of `QuantumProblem` with:

- `prepare()` — build the Hamiltonian as a list of Pauli terms
- `ansatz_circuit` — an appropriate ansatz
- `fci_energy` — a reference solution (classical solver for QUBO,
  DMRG for materials)

No middleware changes needed.

---

## 6. Formal profiling data

### What is deferred

The paper reports wall-time speedups but does not decompose per-iteration
time into components:

- Statevector construction (bandwidth-bound)
- Pauli-term evaluation (compute-bound)
- MPI Allreduce (network/shared-memory-bound)
- Python framework overhead (Qiskit transpilation, PCIe transfers,
  result marshaling)

### Why it matters

Reviewers of systems papers commonly ask: "how do we know your
measured speedup isn't noise?" A profiled breakdown answers this with
per-kernel time attribution.

### Why not now

Simple time budget — nsight-systems adds ~5% overhead but requires
careful integration into the Docker image and re-running the full
benchmark under profiling. ~2 evenings of work.

### What would be needed

    # In Dockerfile: install nsight-systems
    RUN apt-get install -y nsight-systems-cli

    # Wrap the mpirun call in the Makefile with:
    nsys profile --stats=true -o /workspace/results/profile_%q{SLURM_JOB_ID} \
        mpirun --allow-run-as-root -n $NP python3 tests/test_layers_run.py

    # Post-process:
    nsys stats profile_<id>.qdrep

Output: per-kernel time breakdown showing the actual bandwidth-bound vs
compute-bound decomposition. Would strengthen every claim about "why
the A100 is faster than the RTX 6000 on H₂O."

---

## 7. Hot-path micro-optimizations (tabled from GPU-expectation review)

Three optimizations identified during the 2026-08-31 hot-path sweep for
`feature/gpu-expectation-fix` (which landed the primary GPU-native
expectation win). Each is potentially real but tabled to keep the paper
cycle's variable-attribution clean.

### 7.1 Cache `SparsePauliOp` per rank (medium win, medium risk)

Currently `SparsePauliOp.from_list(local_terms)` runs on every SPSA
iteration at line 153 of `_expectation_on_gpu` and line 617 of
`_evaluate_ibm_estimator`. `local_terms` is invariant across iterations
(fixed at problem construction), so the `SparsePauliOp` object could be
built once during `prepare()` and cached on the problem or stack.

At H2O (1086 Pauli terms) the `from_list` call likely costs 5–20 ms per
iter — a potential 5–10% wall-clock win on top of the GPU-expectation
fix. Untested; also unclear whether `SparsePauliOp` is safe to reuse
across multiple `save_expectation_value` circuit appends (Aer may
mutate the op internally).

**When to do this**: after the GPU-expectation fix is validated on
cloud GPU. If the fix delivers Lightning-parity wall-clock, this is
the natural next optimization pass.

### 7.2 Aer `parameter_binds` API (potentially significant, higher risk)

`_evaluate_distributed_statevector` currently calls
`ansatz.assign_parameters()` twice per iteration to produce fully-bound
copies of the circuit. Aer 0.14+ supports passing an unbound circuit
plus a `parameter_binds` list to `sim.run()`, letting Aer do the
parameter substitution internally — potentially on GPU, and batching
both plus/minus evaluations into a single kernel-launch stream.

Predicted wall-clock reduction: 10–30% on top of the GPU-expectation
fix. Real risk: this is a larger API change; Aer version compatibility
matrix for `parameter_binds + save_expectation_value` is not something
I can verify from documentation, only from cloud testing.

**When to do this**: only after 7.1 lands and shows measured benefit
(or lack thereof). Stacking multiple wall-clock changes into one cloud
run makes attribution impossible.

### 7.3 Preallocated MPI reduce buffers (cosmetic)

Every `comm.Allreduce` currently allocates a fresh 1-element numpy
array (lines 501–502 and 626–627). Cost is ~5 µs per iter — genuinely
unmeasurable at H2O's ~100 ms/iter. Only worth doing as a code-quality
cleanup, not a performance fix.

**When to do this**: never as a standalone change. If any other
refactor touches this code path, roll this in as part of it.

---

## Priority ordering for a follow-up paper cycle

If the current QAAS middleware paper is submitted first, the natural
sequence for follow-up work is:

1. **UCCSD** (chemistry credibility) — biggest single accuracy win
2. **Distributed statevector** (systems credibility) — biggest single
   scaling story
3. **Backend plugin system** (multiplies applicability) — enables
   4. Multi-vendor demonstration paper
5. **Advanced error mitigation** (only useful with UCCSD in place)

Application extensibility (materials, combinatorics) and profiling
data are supporting work that can accompany any of the above.
