# Baseline Comparison — QatabasisStack vs Pennylane Lightning-GPU vs Qiskit Aer MPI

Apples-to-apples wall-clock comparison of the three most relevant VQE
execution paths for our target problem set. Motivated by reviewer question 1
in the paper prep: *"how does this stack compare to widely used single-vendor
alternatives?"*

## Status: executed, results committed

The plan below ran to completion on an A100-SXM4-40GB Lambda instance: 18
runs across 3 backends (`hpchybrid`, `lightning`, `aer-mpi`) × 6 molecules
(H2, LiH, BeH2, H2O, NH3, N2). The live, citable results are in
`results/baseline_comparison_gpuexpect/paper_table.md` (this is the current
Table 1 candidate for the paper) — the original `results/baseline_comparison/`
directory (pre-Fix-A) is superseded, and
`results/baseline_comparison_legacy_check/` holds the `VQE_LEGACY_EXPECT=1`
A/B check.

Two real fairness bugs were found and fixed during this work, both worth
knowing before citing any number from this comparison:

1. **Lightning ansatz-parity bug**: the hand-ported Lightning ansatz in
   `benchmarks/baseline_comparison.py` used the wrong `reps` value and
   entanglement pattern, giving Lightning 25-33% fewer parameters than
   hpchybrid across all 6 molecules — every pre-fix Lightning number was
   measured on an easier problem. Fixed by asserting parameter-count parity
   before proceeding; results under `results/baseline_comparison_gpuexpect/`
   are post-fix. The old `results/baseline_comparison_gpuexpect/lightning/`
   pre-fix files are marked invalid in their own `README-INVALIDATED.md`.
2. **Serial-baseline path-parity bug**: `benchmarks/serial_baseline.py`
   used to re-implement its own ansatz-construction logic, which diverged
   from the distributed stack's for H2 and LiH. Fixed by routing it through
   the same `MoleculeResolver` → `ChemistryProblem` → `prepare()` path used
   everywhere else, so serial and distributed numbers are now built from
   bit-identical ansätze.

The rest of this document is the original comparison plan, kept as the
design record for what was measured and why. See
[`GPU_EXPECTATION_FIX.md`](GPU_EXPECTATION_FIX.md) for the related GPU-native
expectation fix that this comparison's numbers depend on.

Cross-refs: `docs/RELATED_WORK.md` (qualitative positioning),
`docs/KNOWN_GAPS.md` gap H (distributed statevector — the leading reason a
per-rank-replicated stack underperforms distributed simulators),
`docs/FUTURE_WORK.md` (rearchitecture path).

---

## What is being compared

Three execution paths against the same VQE problem:

| Label | Backend | Distributed statevector? | Notes |
|---|---|---|---|
| `hpchybrid-aer-gpu` | Qiskit Aer GPU + QatabasisStack MPI | No (per-rank replicated) | Our current path — Pauli-term parallelism only |
| `lightning-gpu-mpi` | Pennylane LightningGPU + Lightning-MPI | Yes (cuStateVec multi-GPU) | Baseline #1 |
| `aer-mpi-distributed` | Qiskit Aer with `blocking_enable=True` | Yes (Aer's own MPI mode) | Baseline #2 |

All three are configured to solve the same VQE problem on the same molecule
set with the same random seed and iteration budget. Nothing else varies.

## Fixed variables

To make the comparison apples-to-apples:

- **Hardware**: single Lambda A100-SXM4-40GB instance, `NP=2` (both ranks on
  the same physical GPU). Same-instance runs back-to-back to avoid cross-run
  drift.
- **Molecules**: H2 (4q), LiH (12q), BeH2 (14q), H2O (14q). Full canonical
  set. **CO2 (30q) deliberately excluded** from the baseline comparison for
  two reasons: (1) too close to A100 memory ceiling for the replicated
  hpchybrid path, unfair comparison to distributed-SV Aer-MPI; (2) at 16k
  Pauli terms × 240 params it would dominate the wall-clock budget for
  the whole comparison. Run CO2 separately via
  `VQE_ACCEPT_COST=1 VQE_PRECISION=fp32 MOLECULES=CO2 MAX_ITERS=10 NP=1 make run`
  (the pre-flight cost check now aborts by default above ~1e11
  amplitude-touches/iteration — CO2 is always over this regardless of
  `MAX_ITERS`, so the explicit opt-in is required) — see the
  LARGE-COST WARNING in `src/api/interface.py` and the
  CO2 section in `docs/AWS_DEPLOYMENT.md`.
- **Ansatz**: same HWE tier, same layer count. Pennylane version has to be
  hand-built to match; Qiskit versions share `AnsatzBuilder`.
- **Optimizer**: SPSA, `MAX_ITERS=100`, same seed=42. Fixed iteration count
  (not convergence-terminated) so wall-clock is directly comparable.
- **Precision**: fp64 across all three (A100 is unaffected by the fp32
  auto-selection change).
- **Warm-up**: one throwaway H2 run per backend before timing starts (loads
  CUDA libraries, primes caches).

## Measured quantities

Per (backend, molecule):

- **Wall-clock time per iteration** (median of the 100 iterations, robust
  against startup transients).
- **Total wall-clock time for the full 100 iterations**.
- **Peak GPU memory** (from `nvidia-smi --query-gpu=memory.used`).
- **Converged energy** (for sanity — should agree to <1 mHa across backends
  on the same ansatz).
- **Absolute error vs FCI** (chemistry accuracy check).

Output: one JSON per (backend, molecule) run into
`results/baseline_comparison/<backend>/<molecule>_<timestamp>.json`, with the
existing `save_results()` schema plus a `baseline_backend` field so
`aggregate_seeds.py` doesn't collide with real QatabasisStack runs.

## Expected outcomes (predictions to falsify)

1. **Aer-GPU replicated ≈ Aer MPI distributed** at ≤14 qubits — the
   replicated statevector fits comfortably in GPU memory, MPI overhead of
   distribution outweighs the benefit at this size. If this **doesn't** hold,
   something is wrong with our path.
2. **Lightning-GPU-MPI faster per iteration** — better-optimized GPU kernels
   at the simulator layer.
3. **All three agree on converged energy** to <1 mHa — sanity check that
   we're solving the same problem.
4. **Peak memory: Aer-GPU replicated ≈ 2× Aer MPI distributed** at NP=2 (each
   rank has its own copy).

## What this proves (and doesn't)

**Proves:**
- Whether our replicated-statevector architecture is competitive at the
  molecule sizes we actually publish, or whether distributed is meaningfully
  better even at 14 qubits.
- Whether Pennylane Lightning's GPU kernels are the raw-performance ceiling.

**Does not prove:**
- QPU integration parity — Lightning and Aer MPI don't do IBM Runtime.
- Statistical robustness — this is a single-seed run for wall-clock; the
  n=5-seed methodology in `benchmarks/local_test_run.py` still owns that
  story.
- Scaling behavior — this is NP=2 only. Full scaling replication under the
  baseline backends is deferred until the pattern above is validated.

## Execution plan (when cloud GPU access returns)

Estimated cost: ~$5 on a 1-hour Lambda A100 instance.

1. Bootstrap Lambda instance per existing `install_native.sh` path.
2. `pip install pennylane pennylane-lightning-gpu pennylane-lightning[mpi]`.
3. Verify Lightning-MPI loads:
   `mpirun -np 2 python -c "import pennylane as qml; print(qml.about())"`.
4. Verify Aer MPI mode loads:
   `python -c "from qiskit_aer import AerSimulator; s = AerSimulator(method='statevector', device='GPU', blocking_enable=True); print('ok')"`.
5. Run the warm-up per backend.
6. Run the three backends × four molecules = 12 timed runs. ~10 min at 100
   iterations per run.
7. `rsync` results back to laptop before terminating instance.
8. Feed into a new aggregator (`benchmarks/aggregate_baseline.py`, deferred)
   to produce the paper table.

## Skeleton entry point

`benchmarks/baseline_comparison.py` — CLI switches backends via a
`--backend={hpchybrid,lightning,aer-mpi}` flag. Currently only the
`hpchybrid` path is fully wired (imports QatabasisStack); the other two raise
`NotImplementedError` with an explicit message pointing at the pip install
step. The stub structure is intentional — locks in the CLI contract and the
result schema now so the actual cloud-GPU run is a one-liner later.

## Open questions before the cloud run

- **Ansatz matching**: Lightning uses Pennylane templates; matching the
  Qiskit HWE tier gate-for-gate requires either (a) exporting our circuit to
  OpenQASM and importing into Pennylane, or (b) hand-porting the ansatz.
  Option (a) is safer for gate-level parity but has a decompose-tied-parameter
  risk (same one that blocks UCCSD — see `docs/FUTURE_WORK.md`). Option (b)
  is what the skeleton assumes; ansatz parity is a `TODO` in the code.
- **SPSA seed portability**: Pennylane and Qiskit SPSA implementations may
  not use the seed the same way — the converged energies should still agree
  but the iteration-by-iteration trajectories will differ. Not a fairness
  issue, but worth calling out in the paper.
