"""Experiment 1, one clean run: AerSimulator-CPU serial baseline + MPI
distributed at P=1 and P=2, same engine on both sides, 5 seeds per
molecule (10 for LiH), all from one script, one git commit, one results
folder.

Why this script exists (full writeup: qatabasis-internal/docs/
RESULT_PROVENANCE.md, "Experiment 1 root cause"): the paper's distributed-
vs-serial speedup claim was built by hand-pairing files from THREE
different, non-interchangeable benchmark paths that happened to all look
like "P=1" or "baseline" numbers:

  1. `make scaling` / local_test_run.py:run_scaling_local() -- LiH ONLY,
     10 iterations, a quick scaling smoke test, not a convergence run.
  2. `make baseline` / serial_baseline.py -- full convergence, but (a)
     used qiskit.quantum_info.Statevector (pure Python) while the
     distributed side uses Aer's compiled engine, confounding "parallel
     speedup" with "compiled engine beats Python" -- the exact issue the
     paper's own "% RERUN INCOMING" comment flags; and (b) saved via its
     own ad hoc JSON writer (not src.api.results.save_results()), so the
     file carries no git_commit/gpu_name/hostname-via-hw field to
     cross-check it against anything.
  3. Full-convergence local_test_run.py sweeps at -np 1/2/4/8 with
     MOLECULES="H2 LiH BeH2 H2O NH3" -- the real distributed-scaling data
     (e.g. the 2026-09-07 sweep), unrelated to either of the above and a
     different experiment (strong scaling, Experiment 3) from Experiment
     1's actual question (does P=2 beat a true serial P=1, same engine).

This script closes both gaps: the serial leg here runs on
AerSimulator(method='statevector', device='CPU') -- the SAME compiled
engine the distributed path uses, just CPU-only and P=1 -- so a measured
speedup reflects MPI/Pauli-distribution, not Python-vs-compiled-C++.
Every leg (serial and distributed) stamps the same git_commit/hostname
via save_results() and writes into one results folder, so there is no
"which file goes with which" judgment call left for later.

Scope note (per Experiment 1's spec): this script only runs P=1 (serial,
no MPI) and P=2 (distributed, the minimum rank count that enables Pauli
term distribution) -- NOT P=4/P=8. The fuller strong-scaling curve is a
separate experiment (3); re-run that one separately if it needs a refresh.

Usage (inside the Docker container, from repo root):
    python3 benchmarks/exp1_clean_sweep.py --mode serial
    mpirun --allow-run-as-root -np 2 python3 benchmarks/exp1_clean_sweep.py --mode distributed

Env vars:
    MOLECULES   space-separated list. Default: "H2 LiH BeH2 H2O" (batch 1 --
                see scripts/run_exp1_sweep.sh for the batching rationale:
                NH3 and N2 are run as separate, later invocations so a
                multi-hour NH3 sweep can be safely interrupted/resumed
                without re-paying for the cheaper four molecules).
    SEEDS       space-separated seed list. Default: "42 43 44 45 46" (5
                seeds). LiH gets 5 more (47-51) automatically -- see
                LIH_EXTRA_SEEDS below, matching the paper's n=10 for LiH.
    N2_SEEDS    Default: "42" (single seed, per spec -- N2 has no serial
                comparison anyway; only its distributed P=2 number is used).
"""
import argparse
import os
import sys
import time
import json
import socket
import subprocess
from datetime import datetime

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

# --mode distributed must disable mpi4py's auto-init (mpi4py.rc.initialize
# defaults to True) BEFORE anything else imports mpi4py -- including
# HardwareProfile._detect_mpi()'s lazy `from mpi4py import MPI` below, which
# runs for BOTH modes via HardwareProfile.detect() further down this file.
# Getting this wrong double-inits MPI: mpi4py auto-inits on its own import,
# then QatabasisStack's hpc_core.init_mpi() (MPI_Init_thread) inits again --
# OpenMPI aborts with "has attempted to initialize MPI more than once."
# src/api/interface.py sets this same flag, but only takes effect if it is
# imported before any other mpi4py import -- which local_test_run.py gets
# right by importing interface.py at the top of its file, and this script
# got wrong by importing QatabasisStack lazily inside run_distributed().
# Setting it here, unconditionally, before HardwareProfile is even
# imported, fixes it for both modes without changing either call site.
if "distributed" in sys.argv or "--mode=distributed" in sys.argv:
    import mpi4py
    mpi4py.rc.initialize = False

from src.api.molecule_resolver import MoleculeResolver
from src.api.results import save_results
from src.api.log import init_log, close_log

DEFAULT_MOLECULES = ["H2", "LiH", "BeH2", "H2O"]   # NH3, N2 run as separate later batches
DEFAULT_SEEDS = [42, 43, 44, 45, 46]                # n=5
LIH_EXTRA_SEEDS = [47, 48, 49, 50, 51]              # LiH gets n=10 total, per spec


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _out_dir(hw_slug):
    ts_day = datetime.now().strftime("%Y-%m-%d")
    return os.path.join("results", f"{hw_slug}-exp1-clean-{ts_day}")


def _seeds_for(molecule, base_seeds):
    if molecule == "LiH":
        return base_seeds + LIH_EXTRA_SEEDS
    return base_seeds


def run_serial(molecules, base_seeds, out_dir):
    """Serial leg: AerSimulator CPU statevector (compiled engine, same as
    the distributed path) with NO MPI (P=1) -- isolates the parallelism
    effect from the Python-vs-compiled-engine effect. One sub-result per
    (molecule, seed); incremental save after every seed, not just every
    molecule, since a single molecule's 5-seed loop can itself run long.
    """
    import numpy as np
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator

    resolver = MoleculeResolver(max_qubits=30, allow_network=True, cache_dir=".pubchem_cache")
    sim = AerSimulator(method="statevector", device="CPU")
    out_path = os.path.join(out_dir, "serial_baseline_aercpu.json")

    # Load whatever a prior batch already wrote to this SAME file (e.g.
    # batch 1's H2/LiH/BeH2/H2O) so this batch's save_results() call
    # merges in, rather than overwriting it -- save_results() itself does
    # a plain json.dump of exactly what it's given, with no merge logic,
    # so the merge has to happen here before the first save of this batch.
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("molecules", {})

    for name in molecules:
        problem = resolver.resolve(name, freeze_core=True).to_chemistry_problem()
        problem.prepare()
        pauli_op = SparsePauliOp.from_list(problem.pauli_terms)
        ansatz = problem.ansatz_circuit   # read-only here: assign_parameters() below
        n_params = problem.num_params      # returns a NEW circuit, never mutates this template
        fci_ref = problem.fci_energy
        max_iterations = max(200, n_params * 8)
        qubit_range = list(range(problem.num_qubits))
        sorted_params = sorted(ansatz.parameters, key=lambda x: x.name)

        def energy(theta):
            # assign_parameters() returns a fresh circuit -- safe to mutate
            # with save_expectation_value (never touches problem.ansatz_circuit
            # itself; see CLAUDE.md Do-Not #5 / interface.py:_build_statevector's
            # comment on the same hazard).
            bound = ansatz.assign_parameters({p: v for p, v in zip(sorted_params, theta)})
            bound.save_expectation_value(pauli_op, qubit_range, label="ev")
            result = sim.run(bound).result()
            return float(result.data(0)["ev"].real)

        seeds = _seeds_for(name, base_seeds)
        per_seed = {}
        for seed in seeds:
            print(f"\n--- {name} seed={seed} (serial, AerSimulator-CPU, P=1) ---")
            np.random.seed(seed)
            theta = np.random.uniform(-0.1, 0.1, n_params)

            c, alpha, gamma = 0.1, 0.602, 0.101
            a = 0.628 / np.sqrt(n_params / 8.0)
            A = max_iterations * 0.1
            min_iters = max(10, min(n_params * 2, 50))

            history = []
            best_physical_energy, best_physical_iter = None, 0
            t0 = time.perf_counter()
            for k in range(1, max_iterations + 1):
                ak = a / (k + A) ** alpha
                ck = c / k ** gamma
                delta = np.random.choice([-1, 1], size=n_params)
                e_plus, e_minus = energy(theta + ck * delta), energy(theta - ck * delta)
                current = (e_plus + e_minus) / 2.0
                history.append(current)
                theta -= ak * (e_plus - e_minus) / (2 * ck * delta)
                if fci_ref is not None and current >= fci_ref:
                    if best_physical_energy is None or current < best_physical_energy:
                        best_physical_energy, best_physical_iter = current, k
                if k >= min_iters and len(history) >= 10:
                    if max(history[-10:]) - min(history[-10:]) < 1.6e-3:
                        break
            t_total = time.perf_counter() - t0

            if fci_ref is not None and history[-1] < fci_ref and best_physical_energy is not None:
                report_e, report_iter = best_physical_energy, best_physical_iter
            else:
                report_e, report_iter = history[-1], len(history)

            per_seed[str(seed)] = {
                "energy": report_e,
                "fci": fci_ref,
                "wall_time": t_total,
                "iterations": len(history),
                "best_iter": report_iter,
                "num_qubits": problem.num_qubits,
                "num_pauli": len(problem.pauli_terms),
                "num_params": n_params,
                "ansatz_tier": problem.ansatz_tier,
            }
            print(f"[{name}] seed={seed} E={report_e:+.6f} Ha | T={t_total:.3f}s | "
                  f"iters={len(history)} | n_params={n_params}")

            results[name] = per_seed
            save_results(
                {"mpi_ranks": 1, "gpu": False, "seed_list": seeds, "mode": "serial_aercpu",
                 "molecules": results, "partial": True},
                backend="serial_baseline", path=out_path,
            )

    save_results(
        {"mpi_ranks": 1, "gpu": False, "mode": "serial_aercpu",
         "molecules": results, "partial": False},
        backend="serial_baseline", path=out_path,
    )
    print(f"\n[Serial] Saved to {out_path}")


def run_distributed(molecules, base_seeds, out_dir):
    """One distributed leg (P ranks, whatever mpirun launched this with).
    Must be invoked under mpirun -- this file does not spawn MPI itself.
    5 seeds per molecule (10 for LiH), same as the serial leg, so every
    molecule has a matched seed set on both sides.
    """
    from src.api.interface import QatabasisStack

    resolver = MoleculeResolver(max_qubits=30, allow_network=True, cache_dir=".pubchem_cache")

    with QatabasisStack(use_gpu=True, backend="simulator") as stack:
        out_path = os.path.join(out_dir, f"distributed_P{stack.size}.json")

        # Same merge-not-overwrite fix as run_serial() above: load whatever
        # a prior batch already wrote to this exact file so this batch's
        # molecules are added, not substituted for, earlier ones. Every
        # rank takes this branch identically (same file, same content),
        # so there is no rank-divergence risk the way there would be for
        # a resolver-network call.
        results = {}
        if stack.rank == 0 and os.path.exists(out_path):
            with open(out_path) as f:
                results = json.load(f).get("molecules", {})
        results = stack.comm.bcast(results, root=0)

        for name in molecules:
            seeds = _seeds_for(name, base_seeds)
            per_seed = {}
            for seed in seeds:
                if stack.rank == 0:
                    print(f"\n--- {name} seed={seed} (distributed, P={stack.size}) ---")
                problem = resolver.resolve(name, freeze_core=True).to_chemistry_problem()
                problem.prepare()

                t0 = time.perf_counter()
                max_iters = max(200, problem.num_params * 8)
                theta, history = stack.vqe_optimize(
                    problem, max_iterations=max_iters, tolerance=1.6e-3,
                    checkpoint_dir=os.path.join("checkpoints", "exp1_clean", f"{problem.name}_s{seed}"),
                    seed=seed,
                )
                t_total = time.perf_counter() - t0

                if stack.rank == 0 and history:
                    fci = problem.fci_energy
                    final_e = history[-1]
                    best_phys = getattr(stack, "_best_physical_energy", None)
                    if fci is not None and final_e < fci - 1e-6 and best_phys is not None:
                        report_e = best_phys
                    else:
                        report_e = final_e

                    per_seed[str(seed)] = {
                        "energy": report_e,
                        "fci": fci,
                        "num_params": problem.num_params,
                        "num_qubits": problem.num_qubits,
                        "num_pauli": len(problem.pauli_terms),
                        "tier": getattr(problem, "ansatz_tier", "hwe"),
                        "iters": len(history),
                        "wall_time": t_total,
                    }
                    print(f"[{name}] seed={seed} P={stack.size} E={report_e:+.6f} Ha | "
                          f"T={t_total:.3f}s | iters={len(history)} | "
                          f"n_params={problem.num_params}")

                    results[name] = per_seed
                    save_results(
                        {"mpi_ranks": stack.size, "gpu": stack.use_gpu, "mode": "distributed",
                         "molecules": results, "partial": True},
                        backend="simulator", hw=stack.hw, stack=stack, path=out_path,
                    )

        if stack.rank == 0:
            save_results(
                {"mpi_ranks": stack.size, "gpu": stack.use_gpu, "mode": "distributed",
                 "molecules": results, "partial": False},
                backend="simulator", hw=stack.hw, stack=stack, path=out_path,
            )
            print(f"\n[Distributed P={stack.size}] Saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["serial", "distributed"], required=True)
    args = p.parse_args()

    _env_molecules = os.environ.get("MOLECULES", "").strip()
    molecules = _env_molecules.split() if _env_molecules else DEFAULT_MOLECULES
    _env_seeds = os.environ.get("SEEDS", "").strip()
    base_seeds = [int(s) for s in _env_seeds.split()] if _env_seeds else DEFAULT_SEEDS

    # Deliberately NOT HardwareProfile.detect().results_slug() -- that
    # returns "cpu-only" for the serial leg (which runs without --gpus all
    # by design, since the point is a true CPU baseline) and the real GPU
    # slug (e.g. "a100-sxm4-40gb") for the distributed legs, so the two
    # modes would silently write into TWO DIFFERENT folders despite being
    # part of the same sweep. This bit a real run (2026-10-29, NH3 batch):
    # the serial leg landed in results/cpu-only-exp1-clean-.../, orphaned
    # from the distributed legs' results/a100-sxm4-40gb-exp1-clean-.../,
    # and had to be manually merged after the fact. The slug below is
    # fixed per-experiment instead, independent of which leg is running.
    HW_SLUG_OVERRIDE = os.environ.get("EXP1_HW_SLUG", "a100-sxm4-40gb")
    out_dir = _out_dir(HW_SLUG_OVERRIDE)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "serial":
        init_log(os.path.join(out_dir, "serial_baseline_aercpu.log"))
        print(f"[Config] git_commit={_git_commit()} hostname={socket.gethostname()}")
        print(f"[Config] mode=serial(AerSimulator-CPU) molecules={molecules} base_seeds={base_seeds}")
        run_serial(molecules, base_seeds, out_dir)
        close_log()
    else:
        print(f"[Config] git_commit={_git_commit()} hostname={socket.gethostname()}")
        print(f"[Config] mode=distributed molecules={molecules} base_seeds={base_seeds}")
        run_distributed(molecules, base_seeds, out_dir)
