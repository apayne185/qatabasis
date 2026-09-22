"""Serial Qiskit VQE reference — single-process, no MPI, no GPU.

This produces the T_serial number the paper uses for "distributed-vs-serial
wall-clock speedup." For that comparison to be meaningful, the ansatz on
both sides must be identical.

**Path parity note (fixed 2026-09-03)**: earlier versions of this file
re-implemented the ansatz-construction logic separately, with two
independently-derived variants of the adaptive-reps formula. Both
matched the main path (src/api/problems.py:build_ansatz) at LiH, BeH2,
H2O, NH3, N2 by coincidence, but diverged at H2 (the guard `if n_qubits
> 4 else reps` gave H2 reps=1 here while the main path gave reps=2) and
at LiH (this file used registry `reps=1` while the main path routed
through MoleculeResolver._recommended_reps() which returns 2 for 12q).

This file now routes through the SAME ChemistryProblem construction the
distributed stack uses (via MoleculeResolver + prepare()), so the
ansatz object is literally the same code path — no risk of drift
between the two comparison sides. What still differs is only the
execution model: serial CPU Statevector here, vs distributed
MPI + GPU cuStateVec in QatabasisStack.
"""
import os
import sys
import time
import json
import socket
from datetime import datetime

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

# Import path parity: route through the same problem-construction the
# distributed stack uses, instead of re-implementing it.
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
from src.api.molecule_resolver import MoleculeResolver
from src.api.log import init_log, close_log


DEFAULT_MOLECULES = ["H2", "LiH", "BeH2", "H2O", "NH3", "N2"]


def serial_vqe(mol_name, problem, seed=42):
    """Run one molecule's serial VQE using the same ChemistryProblem the
    distributed stack builds. Ansatz + Hamiltonian + FCI reference all
    come from problem.prepare() (already called by the caller). Only the
    execution path differs: pure-Python Qiskit Statevector, no MPI, no GPU.
    """
    np.random.seed(seed)
    ansatz = problem.ansatz_circuit
    pauli_op = SparsePauliOp.from_list(problem.pauli_terms)
    n_params = problem.num_params
    fci_ref = problem.fci_energy
    max_iterations = max(200, n_params * 8)
    theta = np.random.uniform(-0.1, 0.1, n_params)

    sorted_params = sorted(ansatz.parameters, key=lambda x: x.name)

    def energy(t):
        bound = ansatz.assign_parameters({p: v for p, v in zip(sorted_params, t)})
        sv = Statevector(bound)
        return float(sv.expectation_value(pauli_op).real)

    # SPSA hyperparams match QatabasisStack (fair comparison).
    # min_iters formula matched to interface.py:vqe_optimize() exactly
    # (fixed 2026-09-04) so the serial vs distributed comparison is not
    # confounded by different convergence-check timing. Old formula was
    # max(20, n_params // 2), which triggered earlier for large n_params
    # and produced an unfair "serial converged faster" artifact.
    c = 0.1
    a = 0.628 / np.sqrt(n_params / 8.0)
    A = max_iterations * 0.1
    alpha, gamma = 0.602, 0.101
    min_iters = max(10, min(n_params * 2, 50))

    history = []
    best_physical_energy = None
    best_physical_iter = 0
    t0 = time.perf_counter()

    for k in range(1, max_iterations + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = np.random.choice([-1, 1], size=n_params)
        e_plus = energy(theta + ck * delta)
        e_minus = energy(theta - ck * delta)
        current = (e_plus + e_minus) / 2.0
        history.append(current)
        gradient = (e_plus - e_minus) / (2 * ck * delta)
        theta -= ak * gradient

        # Track best energy in physical (above-FCI) sector
        if fci_ref is not None and current >= fci_ref:
            if best_physical_energy is None or current < best_physical_energy:
                best_physical_energy = current
                best_physical_iter = k

        if k >= min_iters and len(history) >= 10:
            spread = max(history[-10:]) - min(history[-10:])
            if spread < 1.6e-3:
                break

    t_total = time.perf_counter() - t0

    if fci_ref is not None and history[-1] < fci_ref and best_physical_energy is not None:
        report_energy = best_physical_energy
        report_iter = best_physical_iter
    else:
        report_energy = history[-1]
        report_iter = len(history)

    error = abs(report_energy - fci_ref) if fci_ref is not None else None
    err_str = f"Error={error:+.4f} Ha" if error is not None else "Error=N/A"
    print(f"[{mol_name}] Serial | E={report_energy:+.6f} Ha | {err_str} | "
          f"T={t_total:.3f}s | iters={len(history)} (best at {report_iter}) | "
          f"n_params={n_params}")

    return {
        "energy": report_energy,
        "fci": fci_ref,
        "error": error,
        "wall_time": t_total,
        "iterations": len(history),
        "best_iter": report_iter,
        "num_qubits": problem.num_qubits,
        "num_pauli": len(problem.pauli_terms),
        "num_params": n_params,
        "ansatz_tier": problem.ansatz_tier,
    }


if __name__ == "__main__":
    _env = os.environ.get("MOLECULES", "").strip()
    mols = _env.split() if _env else DEFAULT_MOLECULES

    hostname = socket.gethostname()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "results/cpu-only/serial-baseline"
    os.makedirs(out_dir, exist_ok=True)
    init_log(f"{out_dir}/serial_baseline_{ts}.log")

    print("----SERIAL QISKIT AER BASELINE--- ")
    print(f"[Config] Molecules: {mols}")
    print(f"[Config] Hostname: {hostname}")
    print(f"[Config] Path parity: via MoleculeResolver -> ChemistryProblem (same as distributed stack)")

    resolver = MoleculeResolver(max_qubits=30, allow_network=True,
                                cache_dir=".pubchem_cache")

    # Fixed path, reused for every incremental write below plus the final
    # save. Serial baselines are the slowest thing this stack runs (N2 was
    # excluded from the paper entirely for projecting >10h on one core) --
    # writing only once at the very end means an interrupted run (Ctrl+C,
    # crash, terminated cloud instance) loses every molecule that DID
    # finish, not just the one in flight. Same fix as local_test_run.py.
    out_path = f"{out_dir}/serial_baseline_{ts}.json"

    def _save(results_so_far, partial):
        with open(out_path, "w") as f:
            json.dump({
                "hostname": hostname,
                "timestamp": ts,
                "molecules": results_so_far,
                "partial": partial,
            }, f, indent=2)

    all_results = {}
    for name in mols:
        print(f"\n--- {name} (serial, no MPI) ---")
        try:
            problem = resolver.resolve(name, freeze_core=True).to_chemistry_problem()
            problem.prepare()
        except Exception as e:
            print(f"[{name}] resolution/prepare failed: {e}")
            continue
        all_results[name] = serial_vqe(name, problem)
        _save(all_results, partial=True)
        print(f"[Results] Incremental save ({len(all_results)}/{len(mols)} molecules) to {out_path}")

    print(f"\n{'Molecule':<10} {'Energy (Ha)':<16} {'Error (Ha)':<14} "
          f"{'Iters':<8} {'Time(s)':<10} {'n_params':<10}")
    for name, d in all_results.items():
        err = f"{d['error']:+.4f} Ha" if d['error'] is not None else 'N/A'
        print(f"{name:<10} {d['energy']:<16.6f} {err:<14} "
              f"{d['iterations']:<8} {d['wall_time']:<10.2f} {d['num_params']:<10}")

    _save(all_results, partial=False)
    print(f"\n[Results] Saved to {out_path}")
    close_log()
