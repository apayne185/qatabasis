"""Baseline comparison entry point: QatabasisStack vs Pennylane Lightning-GPU
vs Qiskit Aer MPI on the same VQE problem.

Design and rationale: docs/BASELINE_COMPARISON.md.

Three backends, same VQE problem, apples-to-apples wall-clock:
  - hpchybrid  : our stack (Aer-GPU, per-rank replicated statevector)
  - lightning  : Pennylane LightningGPU/Lightning (import-time capability probe)
  - aer-mpi    : Qiskit Aer with blocking_enable=True (distributed statevector)

The lightning + aer-mpi paths ARE wired but degrade gracefully:
if the required library isn't available, they run on the CPU fallback of
the same library (so local dry-runs work) and stamp the exact device used
into the JSON so the aggregator can spot mixed-device comparisons.

Usage (from repo root, inside the container/env):
    python -m benchmarks.baseline_comparison --backend hpchybrid \
        --molecule H2 --max-iters 100 --seed 42
    mpirun -np 2 python -m benchmarks.baseline_comparison \
        --backend lightning --molecule LiH --max-iters 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

BACKENDS = ("hpchybrid", "lightning", "aer-mpi")
DEFAULT_MOLECULES = ("H2", "LiH", "BeH2", "H2O", "NH3", "N2")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--molecule", default="H2",
                   help="Single molecule ID (H2 / LiH / BeH2 / H2O). "
                        "Loop externally for the full set.")
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", action="store_true",
                   help="Run one throwaway H2 iteration to prime caches "
                        "before the timed run.")
    p.add_argument("--out-dir", default="results/baseline_comparison")
    p.add_argument("--aer-blocking-qubits", type=int, default=None,
                   help="aer-mpi backend only: override blocking_qubits for "
                        "the AerSimulator(blocking_enable=True) config. "
                        "Default: max(n_qubits-2, n_qubits//2). Use this to "
                        "sweep blocking granularity for a proper "
                        "distributed-SV parameter study.")
    return p.parse_args()


# ------------------------------------------------------------------ hpchybrid

def _run_hpchybrid(molecule, max_iters, seed, out_dir):
    """Time QatabasisStack (Aer-GPU, per-rank replicated statevector).

    Mirrors benchmarks/local_test_run.py's flow but strips the multi-molecule
    loop and stamps a `baseline_backend` field so aggregation does not
    collide with real QatabasisStack runs.
    """
    from src.api.interface import QatabasisStack
    from src.api.molecule_resolver import MoleculeResolver

    resolver = MoleculeResolver(max_qubits=30, allow_network=True,
                                cache_dir=".pubchem_cache")
    problem = resolver.resolve(molecule, freeze_core=True).to_chemistry_problem()
    problem.prepare()

    stack = QatabasisStack(backend="simulator")

    t0 = time.perf_counter()
    _theta, history = stack.vqe_optimize(problem, max_iterations=max_iters, seed=seed)
    wall = time.perf_counter() - t0

    device_label = "gpu-custatevec" if stack._gpu_sv else "cpu"
    payload = {
        "baseline_backend": "hpchybrid",
        "device": device_label,
        "molecule": molecule,
        "num_qubits": problem.num_qubits,
        "num_pauli": len(problem.pauli_terms),
        "num_params": problem.num_params,
        "max_iters": max_iters,
        "iters_completed": len(history),
        "seed": seed,
        "wall_seconds": wall,
        "wall_per_iter": wall / max(len(history), 1),
        "energy_ha": history[-1] if history else None,
        "fci_ha": getattr(problem, "fci_energy", None),
        # Captured BEFORE stack.finalize() below -- _save()'s rank check
        # must not call MPI.COMM_WORLD.Get_rank() itself after finalize()
        # has run (MPI_FINALIZE makes Get_rank() disallowed by the
        # standard and aborts the job; confirmed for real, not
        # hypothetical: a fix attempt that re-derived rank post-finalize
        # crashed every run with "MPI_Comm_rank() called after
        # MPI_FINALIZE"). Captured once, here, while `stack.rank` is still
        # valid, and threaded through to _save() instead.
        "_mpi_rank": stack.rank,
    }
    stack.finalize() if hasattr(stack, "finalize") else None
    return payload


# ------------------------------------------------------------------ lightning

def _lightning_ansatz(num_qubits, params, reps, entanglement="full"):
    """Port EfficientSU2 to Pennylane matching the Qiskit path GATE-FOR-GATE.

    Ansatz-parity note (fixed 2026-09-03): earlier versions of this
    function used entanglement="linear" (n-1 CNOTs per layer) with reps
    taken directly from the registry, while the Qiskit path used
    entanglement="full" (n(n-1)/2 CNOTs per layer) with reps expanded
    to adaptive_reps=min(reps+1,3) via build_ansatz(). Result was Lightning
    running a ~25% smaller ansatz on ~85% fewer entangling gates than
    hpchybrid at n=14, invalidating the wall-clock comparison. Both dimensions
    are now caller-controlled and must be set to match the Qiskit path.

    Parameter layout matches Qiskit's EfficientSU2 ordering: for L=reps+1
    rotation blocks, block i uses params[2*i*n : 2*(i+1)*n] as (n RYs,
    then n RZs).
    """
    import pennylane as qml
    idx = 0
    for layer in range(reps + 1):
        for q in range(num_qubits):
            qml.RY(params[idx], wires=q); idx += 1
        for q in range(num_qubits):
            qml.RZ(params[idx], wires=q); idx += 1
        if layer < reps:
            if entanglement == "full":
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "linear":
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            else:
                raise ValueError(f"unknown entanglement pattern: {entanglement!r}")


def _run_lightning(molecule, max_iters, seed, out_dir):
    """Pennylane LightningGPU (falls back to lightning.qubit CPU if GPU
    device not installed). Uses QatabasisStack's problem construction so
    the Hamiltonian is bit-for-bit identical; only the simulator backend
    and SPSA implementation differ.
    """
    try:
        import pennylane as qml
    except ImportError:
        raise RuntimeError(
            "pennylane not installed. Install with:\n"
            "  pip install pennylane pennylane-lightning\n"
            "  # GPU variant (optional): pip install pennylane-lightning-gpu"
        )
    import numpy as np
    from src.api.molecule_resolver import MoleculeResolver

    # Build the same Hamiltonian our stack uses so this is apples-to-apples
    # on the physics, not just on the wall-clock.
    resolver = MoleculeResolver(max_qubits=30, allow_network=True,
                                cache_dir=".pubchem_cache")
    problem = resolver.resolve(molecule, freeze_core=True).to_chemistry_problem()
    problem.prepare()

    n_qubits = problem.num_qubits

    # Mirror src/api/problems.py:build_ansatz() hwe_adaptive tier so the
    # Lightning ansatz is gate-for-gate identical to the Qiskit path.
    # (Previously used problem.reps + linear entanglement, which under-
    # sized the Lightning ansatz by ~25% params and ~85% CNOT gates,
    # invalidating the wall-clock comparison. Now: same adaptive_reps
    # formula and same entanglement threshold as build_ansatz.)
    adaptive_reps = min(problem.reps + 1, 3)
    entanglement = "full" if n_qubits <= 20 else "linear"
    n_params = problem.num_params  # already set by build_ansatz via prepare()
    assert n_params == 2 * n_qubits * (adaptive_reps + 1), (
        f"Lightning ansatz param count {n_params} does not match Qiskit "
        f"EfficientSU2(reps={adaptive_reps}) prediction "
        f"{2 * n_qubits * (adaptive_reps + 1)} -- ansatz parity broken")

    # Prefer LightningGPU when the wheel is present, else the CPU fallback.
    device_label = "cpu-lightning.qubit"
    # Print the actual failure reason if lightning.gpu falls through --
    # bare `except Exception: pass` silently masked a driver mismatch on
    # the 2026-08-30 cloud session so the CPU fallback was hit unnoticed.
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        device_label = "gpu-lightning.gpu"
    except Exception as _e:
        print(f"[lightning] lightning.gpu unavailable "
              f"({type(_e).__name__}: {_e}) -- falling back to lightning.qubit (CPU)",
              flush=True)
        dev = qml.device("lightning.qubit", wires=n_qubits)

    # Assemble Pennylane Hamiltonian from our (pauli_string, coeff) list.
    coeffs = [c for _, c in problem.pauli_terms]
    obs = []
    for pauli_str, _c in problem.pauli_terms:
        # Qiskit stores strings little-endian: rightmost char = qubit 0.
        term = None
        for i, ch in enumerate(reversed(pauli_str)):
            op = {"I": qml.Identity, "X": qml.PauliX,
                  "Y": qml.PauliY, "Z": qml.PauliZ}[ch](i)
            term = op if term is None else term @ op
        obs.append(term if term is not None else qml.Identity(0))
    H = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev, diff_method="parameter-shift")
    def energy(params):
        _lightning_ansatz(n_qubits, params, adaptive_reps, entanglement=entanglement)
        return qml.expval(H)

    # SPSA loop matching src/api/interface.py:vqe_optimize hyperparams so
    # the trajectories are directly comparable.
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.1, 0.1, n_params)
    c = 0.1
    a = 0.628 / np.sqrt(n_params / 8.0)
    A = max_iters * 0.1
    alpha, gamma = 0.602, 0.101
    history = []

    t0 = time.perf_counter()
    for k in range(1, max_iters + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = rng.choice([-1.0, 1.0], size=n_params)
        e_plus = float(energy(theta + ck * delta))
        e_minus = float(energy(theta - ck * delta))
        g = (e_plus - e_minus) / (2 * ck * delta)
        if np.all(np.isfinite(g)):
            theta = theta - ak * g
        history.append((e_plus + e_minus) / 2.0)
    wall = time.perf_counter() - t0

    return {
        "baseline_backend": "lightning",
        "device": device_label,
        "molecule": molecule,
        "num_qubits": n_qubits,
        "num_pauli": len(problem.pauli_terms),
        "num_params": n_params,
        "max_iters": max_iters,
        "iters_completed": len(history),
        "seed": seed,
        "wall_seconds": wall,
        "wall_per_iter": wall / max(len(history), 1),
        "energy_ha": history[-1] if history else None,
        "fci_ha": getattr(problem, "fci_energy", None),
    }


# ------------------------------------------------------------------ aer-mpi

def _run_aer_mpi(molecule, max_iters, seed, out_dir, aer_blocking_qubits=None):
    """Qiskit Aer with distributed-statevector mode (blocking_enable=True).
    Uses our stack's problem construction so the Hamiltonian is identical;
    only the AerSimulator configuration differs.

    Under MPI this tiles the statevector across ranks (the thing gap H
    in docs/KNOWN_GAPS.md flags as our missing capability). Single-rank
    invocation still exercises the code path but does not tile.

    blocking_qubits controls tile size: total chunks = 2^(n - blocking_qubits).
    - n-1: 2 chunks (minimal splitting)
    - n-2: 4 chunks (default; may not be optimal at every n)
    - n-3: 8 chunks (more parallelism, more comm overhead)
    - n/2: max splitting, always exercises distribution
    Pass aer_blocking_qubits explicitly to sweep this parameter -- the
    default heuristic was set arbitrarily and may not be optimal.
    """
    import numpy as np
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
    from src.api.molecule_resolver import MoleculeResolver

    resolver = MoleculeResolver(max_qubits=30, allow_network=True,
                                cache_dir=".pubchem_cache")
    problem = resolver.resolve(molecule, freeze_core=True).to_chemistry_problem()
    problem.prepare()

    n_qubits = problem.num_qubits

    if aer_blocking_qubits is None:
        blocking_qubits = max(n_qubits - 2, n_qubits // 2)  # default heuristic
    else:
        # Guard: blocking_qubits must be < n_qubits (else no blocking) and
        # >= 1 (else meaningless). Clamp instead of erroring to keep sweeps
        # scriptable.
        blocking_qubits = max(1, min(aer_blocking_qubits, n_qubits - 1))
        if blocking_qubits != aer_blocking_qubits:
            print(f"[aer-mpi] --aer-blocking-qubits={aer_blocking_qubits} "
                  f"clamped to {blocking_qubits} (must be in [1, n_qubits-1])",
                  flush=True)

    # Safety guardrail: 2^(n_qubits - blocking_qubits) chunks means aggressive
    # splitting explodes fast. Empirically observed: blocking_qubits <= n-4 on
    # a 20q problem locked the terminal for 45+ min with no completion signal
    # (2026-09-03 session hang). Refuse to launch if the chunk count exceeds
    # a sane cap; user must lower splitting to proceed.
    n_chunks = 2 ** (n_qubits - blocking_qubits)
    MAX_CHUNKS = 64  # 6 splits max; anything more has never won in our measurements
    if n_chunks > MAX_CHUNKS:
        raise ValueError(
            f"[aer-mpi] refusing to launch: blocking_qubits={blocking_qubits} on "
            f"{n_qubits}q problem = {n_chunks} chunks (max {MAX_CHUNKS}). "
            f"Extreme splitting has hung the terminal in past runs with no signal. "
            f"Use blocking_qubits >= {n_qubits - 6} (i.e. <= {MAX_CHUNKS} chunks)."
        )
    device_label = "cpu-aer"
    aer_opts = {"method": "statevector",
                "blocking_enable": True,
                "blocking_qubits": blocking_qubits}
    try:
        sim = AerSimulator(device="GPU", **aer_opts)
        # Cheap probe: try to build a 1-qubit SV on this backend.
        from qiskit import QuantumCircuit
        c = QuantumCircuit(1); c.h(0); c.save_statevector()
        sim.run(c).result()
        device_label = "gpu-aer-blocking"
    except Exception as _e:
        print(f"[aer-mpi] Aer GPU unavailable "
              f"({type(_e).__name__}: {_e}) -- falling back to CPU AerSimulator",
              flush=True)
        sim = AerSimulator(**aer_opts)
        device_label = "cpu-aer-blocking"

    hamiltonian = SparsePauliOp.from_list(
        [(p, c) for p, c in problem.pauli_terms])

    ansatz = problem.ansatz_circuit
    n_params = problem.num_params
    sorted_params = sorted(ansatz.parameters, key=lambda x: x.name)

    def energy(theta):
        bound = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_params, theta)})
        bound.save_statevector()
        sv_data = sim.run(bound).result().get_statevector(bound)
        sv = Statevector(sv_data)
        return float(sv.expectation_value(hamiltonian).real)

    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.1, 0.1, n_params)
    c_spsa = 0.1
    a_spsa = 0.628 / np.sqrt(n_params / 8.0)
    A_spsa = max_iters * 0.1
    alpha, gamma = 0.602, 0.101
    history = []

    t0 = time.perf_counter()
    for k in range(1, max_iters + 1):
        ak = a_spsa / (k + A_spsa) ** alpha
        ck = c_spsa / k ** gamma
        delta = rng.choice([-1.0, 1.0], size=n_params)
        e_plus = energy(theta + ck * delta)
        e_minus = energy(theta - ck * delta)
        g = (e_plus - e_minus) / (2 * ck * delta)
        if np.all(np.isfinite(g)):
            theta = theta - ak * g
        history.append((e_plus + e_minus) / 2.0)
    wall = time.perf_counter() - t0

    return {
        "baseline_backend": "aer-mpi",
        "device": device_label,
        "blocking_qubits": blocking_qubits,
        "molecule": molecule,
        "num_qubits": n_qubits,
        "num_pauli": len(problem.pauli_terms),
        "num_params": n_params,
        "max_iters": max_iters,
        "iters_completed": len(history),
        "seed": seed,
        "wall_seconds": wall,
        "wall_per_iter": wall / max(len(history), 1),
        "energy_ha": history[-1] if history else None,
        "fci_ha": getattr(problem, "fci_energy", None),
    }


# ------------------------------------------------------------------ dispatch + I/O

_DISPATCH = {
    "hpchybrid": _run_hpchybrid,
    "lightning": _run_lightning,
    "aer-mpi": _run_aer_mpi,
}


def _is_rank_zero_or_unmpi(payload):
    # All three backends (hpchybrid via hpc_core, lightning via its own MPI
    # variant, aer-mpi via Aer's internal PMPI integration) coordinate MPI
    # themselves rather than through this script. Without this check,
    # EVERY rank launched by an external `mpirun -np N` independently
    # calls _save() with the same molecule/backend/near-identical
    # timestamp, and multiple ranks opening the same path in "w" mode
    # concurrently produces a truncated, doubly-written, invalid JSON
    # file -- confirmed for real (not hypothetical): a 2026-11-03
    # hpchybrid rerun at NP=2 hit this exact race and corrupted H2's
    # output.
    #
    # hpchybrid's _run_hpchybrid() stamps "_mpi_rank" into the payload,
    # captured from `stack.rank` BEFORE stack.finalize() runs -- this
    # function must use that captured value rather than re-derive the
    # rank itself via MPI.COMM_WORLD.Get_rank(), because by the time
    # main() calls _save(), finalize() (hpc_core.finalize_mpi()) has
    # already run for the hpchybrid path, and MPI_Comm_rank() after
    # MPI_FINALIZE is disallowed by the MPI standard -- confirmed for
    # real: an earlier version of this fix that checked rank live here
    # crashed every single run with "MPI_Comm_rank() called after
    # MPI_FINALIZE."
    if "_mpi_rank" in payload:
        return payload["_mpi_rank"] == 0
    # lightning/aer-mpi never call hpc_core.init_mpi() or finalize_mpi()
    # anywhere in this file (Aer/Lightning coordinate MPI internally, at
    # the C++ level, invisible to this Python-level check) -- so a live
    # mpi4py check here is safe for them: nothing finalizes what this
    # checks. MPI.Is_initialized() == False covers the genuine
    # single-process, no-mpirun invocation (this script's own first
    # usage example), where "rank 0" is simply the only process that
    # exists.
    try:
        from mpi4py import MPI
        if not MPI.Is_initialized():
            return True
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def _save(payload, out_dir, backend_slug, molecule):
    if not _is_rank_zero_or_unmpi(payload):
        return
    payload = {k: v for k, v in payload.items() if k != "_mpi_rank"}
    os.makedirs(os.path.join(out_dir, backend_slug), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, backend_slug, f"{molecule}_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] wrote {path}")


def main():
    args = _parse_args()

    if args.warmup and args.molecule != "H2":
        print("[baseline] running warmup on H2 before timed run...")
        try:
            warmup = _DISPATCH[args.backend]("H2", max_iters=5, seed=0,
                                             out_dir=os.path.join(args.out_dir, "_warmup"))
            print(f"[baseline] warmup done ({warmup['wall_seconds']:.2f}s)")
        except Exception as e:
            print(f"[baseline] warmup failed, continuing anyway: {e}")

    print(f"[baseline] backend={args.backend} molecule={args.molecule} "
          f"iters={args.max_iters} seed={args.seed}")
    if args.backend == "aer-mpi":
        result = _run_aer_mpi(args.molecule, args.max_iters, args.seed,
                              args.out_dir,
                              aer_blocking_qubits=args.aer_blocking_qubits)
    else:
        result = _DISPATCH[args.backend](args.molecule, args.max_iters,
                                         args.seed, args.out_dir)
    _save(result, args.out_dir, args.backend, args.molecule)
    print(f"[baseline] done: wall={result['wall_seconds']:.2f}s "
          f"({result['wall_per_iter']:.4f}s/iter) "
          f"device={result['device']} "
          f"E={result.get('energy_ha')}")


if __name__ == "__main__":
    main()
