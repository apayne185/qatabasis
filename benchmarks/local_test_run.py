import sys
import os
import json
import numpy as np
import time
from datetime import datetime


# Add project root and C++ build dir to import path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

try:
    from src.api.interface import QatabasisStack
    from src.api.problems import ChemistryProblem, FinanceProblem, ANSATZ_TIERS
    from src.api.molecule_resolver import (MoleculeResolver, MoleculeTooBigError, ResolutionError)
    from src.api.results import save_results
    from src.api.log import init_log, log, close_log
    from src.api.hardware import HardwareProfile
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import hpc_core module: {e}")
    sys.exit(1)



USE_GPU = os.environ.get("USE_GPU", "yes").strip().lower() == "yes"
BACKEND = os.environ.get("BACKEND", "simulator")
SEED = int(os.environ.get("SEED", "42"))
MAX_ITERS_ENV = os.environ.get("MAX_ITERS", "").strip()   # optional cap for hardware-ceiling tests  # SPSA random seed (override for multi-seed runs)
_env_molecules = os.environ.get("MOLECULES", "").strip()
MOLECULES = _env_molecules.split() if _env_molecules else ["H2", "LiH", "BeH2", "H2O", "NH3", "N2"]

resolver = MoleculeResolver(max_qubits=30, allow_network=True, cache_dir=".pubchem_cache",)



def make_problem(molecule_input: str, force_tier: str|None= None, stack=None)-> ChemistryProblem | None:
    # Every rank calls this independently (no rank-0 guard, nothing broadcast
    # by the caller), which is normally fine for registry molecules --
    # PySCF/prepare() is deterministic and rank-local, no network or shared
    # state involved. But MoleculeResolver's non-registry paths (SMILES,
    # PubChem) DO touch the network and a shared on-disk cache file with no
    # locking -- if one rank hits a transient failure there (a race on the
    # cache write, a dropped connection) while others succeed, ranks return
    # divergent results with no collective to reconcile them. The caller
    # then either proceeds into vqe_optimize()'s comm.Bcast/Allreduce on
    # some ranks while others already returned early -- a silent,
    # permanent hang with no error message pointing at the cause.
    #
    # Fix: if a stack is passed, every rank still resolves independently
    # (cheap, no broadcast of the actual ChemistryProblem needed for the
    # common case), but an MPI Allreduce(LAND) forces every rank to agree
    # on whether ALL ranks succeeded. Any single-rank failure makes every
    # rank return None consistently, so the caller's "if problem is None:
    # return early" branch is taken uniformly -- no rank is left alone in
    # a state that leads it into a collective the others already skipped.
    ok = True
    problem = None
    try:
        result= resolver.resolve(molecule_input,freeze_core=True)
        problem = result.to_chemistry_problem(force_tier=force_tier)
        problem.prepare()
    except MoleculeTooBigError as e:
        print(f"\n[SKIPPED] {molecule_input}: {e}")
        ok = False
    except ResolutionError as e:
        print(f"\n[FAILED] {molecule_input}: {e}")
        ok = False
    except Exception as e:
        print(f"\n[ERROR] {molecule_input}: {type(e).__name__}: {e}")
        ok = False

    if stack is not None:
        from mpi4py import MPI as _MPI
        all_ok = stack.comm.allreduce(ok, op=_MPI.LAND)
        if not all_ok:
            if ok and stack.rank == 0:
                # This rank succeeded but at least one other rank did not --
                # discard this rank's own successful result so every rank
                # returns None uniformly, rather than diverging.
                print(f"[WARNING] {molecule_input}: at least one MPI rank "
                      f"failed to resolve this molecule; discarding on all "
                      f"ranks (including ones that succeeded) to avoid a "
                      f"rank-divergent state.")
            return None

    return problem




def run_chemistry_local(stack: QatabasisStack, molecule_input: str, force_tier: str | None= None):
    if stack.rank == 0: print(f"\n\n--- RUNNING CHEMISTRY TASK {molecule_input} ---")

    problem = make_problem(molecule_input, force_tier=force_tier, stack=stack)
    if problem is None:
        return None, None, None, None
    
    t0 = time.perf_counter()
    max_iters = int(MAX_ITERS_ENV) if MAX_ITERS_ENV else max(200, problem.num_params * 8)  # scale with parameter count
    ckpt_dir = os.path.join("checkpoints", problem.name)
    theta, history = stack.vqe_optimize(problem,
                                        max_iterations=max_iters,
                                        tolerance=1.6e-3,
                                        checkpoint_dir=ckpt_dir,
                                        seed=SEED)
    
    t_total = time.perf_counter() - t0
    
    if stack.rank ==0 and history:
        final_e = history[-1]
        fci = problem.fci_energy

        best_physical = getattr(stack, '_best_physical_energy', None)
        if fci is not None and final_e < fci - 1e-6 and best_physical is not None:
            report_e = best_physical
            report_iter = getattr(stack, '_best_physical_iter', len(history))
            report_label = "Best physical energy"
        else:
            report_e = final_e
            report_iter = len(history)
            report_label = "Final energy"

        error = abs(report_e - fci) if fci is not None else None
        error_str = f"{error:+.4f} Ha " if error is not None else "N/A"

        print(f"\n[{problem.name}] Ansatz tier:{problem.ansatz_tier}")
        print(f"[{problem.name}] Corr score: {problem.diagnostics.get('correlation_score', 'N/A'):.3f} ")
        print(f"[{problem.name}] {report_label}: {report_e:+.6f} Ha ")
        if fci is not None:
            print(f"[{problem.name}] Reference FCI : {fci:+.6f} Ha ")
            print(f"[{problem.name}] Absolute error: {error_str}")
            print(f"[{problem.name}] Iterations run: {len(history)} (best at iter {report_iter})")
            print(f"[{problem.name}] Wall time: {t_total:.2f}s (includes QPU queue + RTT) ")
            print(f"[{problem.name}] Time/iter: {t_total/ len(history):.2f} s ")

        if hasattr(problem, 'resolution_metadata'):
            meta = problem.resolution_metadata
            print(f"[{problem.name}] Source: {meta['source']} ")
            print(f"[{problem.name}] Active electrons: {meta['active_electrons']} ")
            print(f"[{problem.name}] Estimated qubits: {meta['estimated_qubits']} ")

            for w in meta.get('warnings',[]):    
                print(f"[{problem.name}] Note: {w}")

    return history, problem, t_total, theta   




def run_finance_local(stack:QatabasisStack):
    if stack.rank == 0: print("\n\n\n--- RUNNING FINANCE (4-Assest Portfolio QUBO) TASK ---")
    np.random.seed(42)
    n_assets = 4
    A = np.random.rand(n_assets, n_assets)
    cov = A @ A.T / n_assets  # synthetic positive-definite covariance
    mu  = np.random.uniform(0.02, 0.15, n_assets)

    problem = FinanceProblem(cov, expected_returns=mu, risk_factor=1.0)
    t0 = time.perf_counter()
    _, history = stack.vqe_optimize(problem,
                                        max_iterations=30,
                                        tolerance=1e-3,
                                        checkpoint_dir="checkpoints/finance",
    )
    t_total = time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        print(f"[Finance] Final objective : {history[-1]:+.6f} ")
        print(f"[Finance] Iterations run : {len(history)}")
        print(f"[Finance] Wall time : {t_total:.2f} s")
        result = {
            "final_objective": history[-1],
            "iterations": len(history),
            "wall_time": t_total,
            "n_assets": n_assets,
            "history": history,
        }
    return result



def run_scaling_local(stack: QatabasisStack):
    if stack.rank == 0: print(f"\n RUNNING SCALING (with P={stack.size} ranks) ")

    problem = make_problem("LiH", stack=stack)
    if problem is None:
        if stack.rank == 0: print("[Scaling] LiH  resolution failed, skipping.")
        return None
    t0 = time.perf_counter()

    _, history = stack.vqe_optimize(problem, max_iterations=10,
                                    checkpoint_dir="checkpoints/scaling")
    t_total= time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        result_line = (f"P={stack.size} T_total={t_total:.4f} final_E={history[-1]:+.6f} tier={getattr(problem, 'ansatz_tier', 'N/A')} qubits={problem.num_qubits} iters={len(history)}")
        print(f"[Scaling] {result_line}")

        scaling_dir = f"results/{stack.hw.results_slug()}/scaling"
        os.makedirs(scaling_dir, exist_ok=True)
        with open(f"{scaling_dir}/scaling_P{stack.size}.txt", "w") as f:
            f.write(result_line + "\n")

        result = {
            "ranks": stack.size,
            "wall_time": t_total,
            "final_energy": history[-1],
            "tier": getattr(problem, 'ansatz_tier', 'N/A'),
            "qubits": problem.num_qubits,
            "iterations": len(history),
        }
    return result


def run_weak_scaling(stack: QatabasisStack):
    # Molecule scales with P so per-rank Pauli term count stays relatively constant.
    # N2 (2951 terms) deliberately has no tier here -- it's close enough in size to
    # NH3 that it would want the same P=16 slot, and going further to P=32 to give
    # it its own slot pushes single-host MPI into a shared-memory-contention regime
    # deeper than anything else in this paper's scaling story tests (strong scaling
    # stops at P=8 for the same single-host reason -- see README.md's Known
    # Limitations). N2 stays excluded from weak scaling; disclose this in the paper
    # text the same way N2's strong-scaling exclusion is already disclosed.
    weak_scaling_map = {
        1: "H2",        # 15 terms / 1 rank  =15 terms/rank
        2: "LiH",        # 631 terms / 2 ranks = 316 terms/rank
        4: "BeH2",       # 666 terms / 4 ranks = 167 terms/rank
        8: "H2O",       # 1086 terms / 8 ranks = 136 terms/rank
        16: "NH3",      # 3057 terms / 16 ranks = 191 terms/rank
    }

    mol_name = weak_scaling_map.get(stack.size)
    if mol_name is None:
        if stack.rank == 0:
            print(f"[Weak Scaling] No molecule mapped for P={stack.size}, skipping.")
        return None

    if stack.rank == 0:
        print(f"\n RUNNING WEAK SCALING (P={stack.size}, molecule={mol_name})")

    problem = make_problem(mol_name, stack=stack)
    if problem is None:
        if stack.rank == 0:
            print(f"[Weak Scaling] {mol_name} resolution failed, skipping.")
        return None

    t0 = time.perf_counter()
    _, history = stack.vqe_optimize(problem, max_iterations=10,
                                    checkpoint_dir=f"checkpoints/weak_scaling")
    t_total = time.perf_counter() - t0
    t_per_iter = t_total / len(history) if history else 0

    result = None
    if stack.rank == 0 and history:
        n_terms = len(problem.pauli_terms)
        terms_per_rank = n_terms / stack.size
        result_line = (f"P={stack.size} mol={mol_name} terms={n_terms} "
                       f"terms/rank={terms_per_rank:.0f} T_total={t_total:.4f} "
                       f"T/iter={t_per_iter:.4f} final_E={history[-1]:+.6f}")
        print(f"[Weak Scaling] {result_line}")

        scaling_dir = f"results/{stack.hw.results_slug()}/scaling"
        os.makedirs(scaling_dir, exist_ok=True)
        with open(f"{scaling_dir}/weak_scaling_P{stack.size}.txt", "w") as f:
            f.write(result_line + "\n")

        result = {
            "ranks": stack.size,
            "molecule": mol_name,
            "pauli_terms": n_terms,
            "terms_per_rank": terms_per_rank,
            "wall_time": t_total,
            "time_per_iter": t_per_iter,
            "final_energy": history[-1],
            "iterations": len(history),
        }
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MOLECULES = sys.argv[1:]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Detected once here purely to route the log file; QatabasisStack below
    # re-detects for its own use (cheap, avoids threading hw through init_log).
    _hw_slug = HardwareProfile.detect().results_slug()
    os.makedirs(f"results/{_hw_slug}/simulator", exist_ok=True)
    init_log(f"results/{_hw_slug}/simulator/run_{ts}.log")

    # RESUME=1 opts into a seed-stable filename (no timestamp) so a fresh
    # process invocation (e.g. the next iteration of a `for seed in ...` loop
    # after a crash) finds the SAME file a prior attempt at this seed wrote,
    # instead of a timestamp making every restart invisible to the others.
    # Molecules already present in that file are skipped -- for a multi-hour
    # sweep (NH3/N2-scale problems), recomputing finished molecules after an
    # interrupt wastes real cloud-GPU money, not just time. Default stays
    # timestamped (old behavior) so this is opt-in, not a silent path change
    # for existing tooling/scripts that expect the timestamped filename.
    _resume = os.environ.get("RESUME", "").strip().lower() in {"1", "yes", "true"}
    if _resume:
        _results_path = f"results/{_hw_slug}/simulator/simulator_seed{SEED}.json"
    else:
        _results_path = f"results/{_hw_slug}/simulator/simulator_{ts}.json"

    results = {}
    if _resume and os.path.exists(_results_path):
        with open(_results_path) as f:
            _prior = json.load(f)
        _prior_molecules = _prior.get("molecules", {})
        _already_done = [m for m in MOLECULES if m in _prior_molecules]
        if _already_done:
            print(f"[Resume] Found prior results at {_results_path}, "
                  f"skipping already-completed molecules: {_already_done}")
            results.update(_prior_molecules)
            MOLECULES = [m for m in MOLECULES if m not in _prior_molecules]

    print(f"[Config] GPU={'requested' if USE_GPU else 'CPU mode'}")
    print(f"[Config] Molecules: {MOLECULES}")
    print(f"[Config] Resolver: max_qubits=30, cache=.pubchem_cache/")


    with QatabasisStack(use_gpu=USE_GPU, backend=BACKEND) as stack:
        for mol in MOLECULES:
            history, problem, t_total, theta = run_chemistry_local(stack, mol)

            if stack.rank == 0 and history:
                if problem is not None:
                    fci = getattr(problem, "fci_energy", None)
                    final_e = history[-1]
                    best_phys = getattr(stack, '_best_physical_energy', None)
                    best_phys_theta = getattr(stack, '_best_physical_theta', None)
                    if fci is not None and final_e < fci - 1e-6 and best_phys is not None:
                        report_e = best_phys
                        report_theta = best_phys_theta
                    else:
                        report_e = final_e
                        report_theta = theta

                    # "energy" (report_e above) is the SPSA-internal perturbed-average
                    # quantity -- (E(theta+ck*delta)+E(theta-ck*delta))/2 at whichever
                    # iteration was selected, NOT a separately-verified E(theta). It
                    # carries a small O(ck^2) bias that does not decay to zero over a
                    # realistic run (see evaluate_unperturbed_energy()'s docstring in
                    # src/api/interface.py). unperturbed_energy below is a genuine,
                    # single exact statevector evaluation of report_theta -- this is
                    # the reproducible, provenance-labeled number for accuracy claims.
                    # Cheap: one extra statevector build, done once per molecule here,
                    # not per-iteration.
                    unperturbed_energy = (
                        stack.evaluate_unperturbed_energy(problem, report_theta)
                        if report_theta is not None else None
                    )

                    results[mol] = {
                        "energy": report_e,
                        "unperturbed_energy": unperturbed_energy,
                        "tier":getattr(problem, "ansatz_tier", "hwe"),
                        "score":problem.diagnostics.get("correlation_score", 0.0),
                        "fci": fci,
                        "iters": len(history),
                        "wall_time": t_total,
                        "history": history,
                    }

                    # Incremental save: write results-so-far after every
                    # molecule, not just once at the end. A multi-hour sweep
                    # (e.g. NH3/N2 across several seeds) interrupted midway
                    # through molecule N still leaves molecules 1..N-1 on
                    # disk instead of losing the whole run.
                    save_results({
                        "mpi_ranks": stack.size,
                        "gpu": stack.use_gpu,
                        "seed": SEED,
                        "molecules": results,
                        "scaling": None,
                        "weak_scaling": None,
                        "finance": None,
                        "partial": True,
                    }, backend=BACKEND, hw=stack.hw, stack=stack, path=_results_path)

        # Finance problem - opt-in via RUN_FINANCE=1. Demonstrates the same
        # middleware running a non-chemistry workload (portfolio QUBO -> Ising).
        finance_result = None
        if os.environ.get("RUN_FINANCE", "").strip() in {"1", "yes", "true"}:
            finance_result = run_finance_local(stack)

        scaling_result = run_scaling_local(stack)
        weak_scaling_result = run_weak_scaling(stack)

        if stack.rank == 0:
            print("\n\n\n---ALL LOCAL BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")
            print(f"{'Molecule':<10} {'Energy (Ha)':<16} {'Error (Ha)':<14} {'Ansatz':<20} {'Corr':<6} {'Iters':<8} {'Time(s)':<10} {'T/iter(s)':<10}")

            for mol, data in results.items():
                fci = data.get("fci")
                energy  = data["energy"]
                err_str = (f"{abs(energy-fci):+.4f} Ha" if fci is not None else "N/A")
                tier_label = ANSATZ_TIERS.get(data['tier'],{}).get("label", data['tier'])
                iters  = data.get("iters", 0)
                t_wall = data.get("wall_time", 0.0)
                t_per_iter = t_wall / iters if iters > 0 else 0.0

                print(f"{mol:<10} {energy:<16.6f} {err_str:<14} {tier_label:<20} {data['score']:<6.3f} {iters:<8} {t_wall:<10.2f} {t_per_iter:<10.4f}")

            save_results({
                "mpi_ranks": stack.size,
                "gpu": stack.use_gpu,
                "seed": SEED,
                "molecules": results,
                "scaling": scaling_result,
                "weak_scaling": weak_scaling_result,
                "finance": finance_result,
            }, backend=BACKEND, hw=stack.hw, stack=stack, path=_results_path)

    close_log()



