import hpc_core     # compiled C++ pybind11 bridge (build/hpc_core.so)
from src.api.problems import QuantumProblem
from src.api.hardware import HardwareProfile

import glob as _glob
import os
import sys

# Warn at import time if this is a CPU-only build
if not hpc_core.cuda_build():
    print(
        "[VQE] WARNING: hpc_core was built without CUDA. "
        "GPU acceleration is not available in this installation.\n"
        "         All computation will run on CPU. "
        "To enable GPU support, rebuild with the CUDA toolkit installed.",
        file=sys.stderr
    )
import time as _time
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# nvtx ranges make nsys profiles readable (SPSA-iter, sv-build labels show
# up on the timeline). Fall back to a no-op decorator/contextmanager on
# machines without nvtx installed -- production hot path pays zero cost.
try:
    import nvtx as _nvtx
    _nvtx_range = _nvtx.annotate
except ImportError:
    from contextlib import contextmanager as _cm
    @_cm
    def _nvtx_range(message=None, color=None, domain=None):
        yield

_GPU_SV_AVAILABLE = False
_AerSimulator = None
try:
    from qiskit_aer import AerSimulator as _AerSim
    _AerSimulator = _AerSim
except ImportError as e:
    import sys
    print(f"[Stack] qiskit-aer not available ({e}) - GPU statevector disabled", file=sys.stderr)


class QatabasisStack:
    def __init__(self, use_gpu: bool | None = None, backend:str = 'simulator'):
        # Hardware auto-detection (GPU vendor/class, libs, MPI). Researchers
        # override via VQE_PRECISION / VQE_BACKEND / USE_GPU env vars.
        self.hw = HardwareProfile.detect()

        if use_gpu is None:
            use_gpu = self.hw.want_gpu()

        # If user requested GPU but this is a CPU-only build, warn and override
        if use_gpu and not hpc_core.cuda_build():
            print(
                "[VQE] WARNING: use_gpu=True requested but this build has no CUDA support. "
                "Falling back to CPU. Rebuild with CUDA toolkit to enable GPU.",
                file=sys.stderr
            )
            use_gpu = False

        self.use_gpu = use_gpu
        self.backend = backend
        self.precision = "auto"           # resolved per-problem in vqe_optimize
        self._ibm_session= None
        self._ibm_estimator = None
        self._ibm_transpiled = None
        self._ibm_observable = None

        # MPI init via C++ bridge ( precede any comm calls)
        self.provided_thread_level = hpc_core.init_mpi()
        if not MPI.Is_initialized():
            MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = hpc_core.get_rank()
        self.size = hpc_core.get_size()

        # Round-robin GPU assignment (1 GPU per rank)
        if self.use_gpu:
            try:
                hpc_core.set_cuda_device(self.rank)
            except Exception as e:
                if self.rank == 0:
                    print(f"Rank {self.rank}: GPU initialization failed, falling back to CPU.  Error: {e}")
                self.use_gpu = False

        # Test for cuStateVec GPU backend
        self._gpu_sv = False
        if self.use_gpu and _AerSimulator is not None:
            try:
                test_sim = _AerSimulator(method='statevector', device='GPU')
                self._gpu_sv = True
                self._aer_gpu = test_sim
                if self.rank == 0:
                    print("[Stack] GPU statevector (cuStateVec) available")
            except Exception as e:
                if self.rank == 0:
                    print(f"[Stack] GPU requested but cuStateVec unavailable: {e}")
                    print("[Stack] Falling back to CPU Statevector")

        if self.rank == 0:
            print(self.hw.describe())
            for n in self.hw.notes:
                print(f"[hw] note: {n}")
            print(f"[Stack]  Initialized {self.size} MPI rank(s), GPU={'enabled' if self.use_gpu else 'disabled'}, "
                  f"SV_backend={'GPU (cuStateVec)' if self._gpu_sv else 'CPU (Qiskit)'}, backend='{self.backend}'")


    def _build_statevector(self, bound_circuit):
        # Build statevector from bound circuit, uses GPU (cuStateVec) if available, else CPU.
        #
        # bound_circuit is mutated in place (save_statevector() appended directly,
        # no defensive .copy()) rather than copied -- safe because every call site
        # (_evaluate_distributed_statevector, _evaluate_ibm_estimator) passes a
        # circuit freshly produced by ansatz.assign_parameters() that is never
        # used again afterward. Do NOT pass problem.ansatz_circuit (the shared,
        # reused template) or any circuit the caller still needs post-call --
        # this would also corrupt the IBM hardware submission path, which reuses
        # the same shared ansatz template via a separate transpile() call.
        # aer_precision is set once per vqe_optimize() call (see there), not on
        # every evaluation -- it doesn't change mid-run.
        with _nvtx_range(message="sv-build", color="green", domain="vqe"):
            if self._gpu_sv:
                from qiskit.quantum_info import Statevector as _SV
                sim = self._aer_gpu
                bound_circuit.save_statevector()
                result = sim.run(bound_circuit).result()
                sv_data = result.get_statevector(bound_circuit)
                return _SV(sv_data)
            else:
                return Statevector(bound_circuit)


    def _expectation_on_gpu(self, bound_plus, bound_minus, local_terms):
        # Compute <psi(theta_plus)|H_local|psi(theta_plus)> and the same for
        # theta_minus, without ever pulling the full 2^n statevector to CPU.
        #
        # Uses Aer's save_expectation_value instruction: Aer builds the SV on
        # GPU (cuStateVec) and evaluates the Pauli expectation on the SAME
        # GPU state, returning only the two scalar reals.  The prior code
        # path built the SV, copied it CPU-side via get_statevector, then
        # called SparsePauliOp.expectation_value in numpy -- which is the
        # measured gap vs Lightning-GPU on H2O (1.79x slower).
        #
        # bound_plus and bound_minus are single-use circuits (freshly bound
        # via assign_parameters), safe to mutate in place. Same rule as
        # _build_statevector: do NOT pass a shared template here.
        with _nvtx_range(message="sv+expect", color="orange", domain="vqe"):
            local_op = SparsePauliOp.from_list(local_terms)
            qubit_range = list(range(bound_plus.num_qubits))
            bound_plus.save_expectation_value(local_op, qubit_range, label="ev")
            bound_minus.save_expectation_value(local_op, qubit_range, label="ev")

            sim = self._aer_gpu
            r_plus = sim.run(bound_plus).result()
            r_minus = sim.run(bound_minus).result()

            e_plus_local = float(r_plus.data(0)["ev"].real)
            e_minus_local = float(r_minus.data(0)["ev"].real)
            return e_plus_local, e_minus_local


    def vqe_optimize(self, problem: QuantumProblem, max_iterations:int=100, tolerance:float=1.6e-3, restart_from:str | None = None, checkpoint_dir: str = "checkpoints", start_iter: int = 0, seed: int | None = None) -> tuple[np.ndarray, list[float]]:
        self._below_fci_warned = False
        # Reset best-physical-energy trackers per molecule -- otherwise a
        # multi-molecule run leaks the previous molecule's cached value into
        # the current one when its trajectory drops below FCI.
        for _attr in ("_best_physical_energy", "_best_physical_theta", "_best_physical_iter"):
            if hasattr(self, _attr):
                delattr(self, _attr)
        if seed is not None:
            np.random.seed(seed)
        comm = self.comm
        problem.prepare()
        num_params = problem.num_params
        num_qubits  = problem.num_qubits

        if num_params == 0:
            raise ValueError(
                "problem.num_params= 0 after prepare(), check that ansatz was built correctly. "
            )

        # Resolve precision policy for this problem size (once per optimize call).
        self.precision = self.hw.recommend_precision(num_qubits)
        if self._gpu_sv:
            # Set once here, not on every _build_statevector() call -- precision
            # doesn't change mid-run, and this was previously re-set on every
            # single evaluation (hundreds of times per run, twice per iteration).
            aer_precision = "double" if self.precision == "fp64" else "single"
            try:
                self._aer_gpu.set_options(precision=aer_precision)
            except Exception:
                pass
        if self.rank == 0:
            # self.size (not self.hw.mpi_size) -- HardwareProfile.detect() runs before
            # MPI is initialized, so self.hw.mpi_size is stale/always 1. self.size is
            # the real post-init rank count, needed since multiple ranks can share one
            # physical GPU (round-robin device assignment) and each redundantly builds
            # its own full statevector -- see max_qubits_fit()'s docstring.
            max_fit = self.hw.max_qubits_fit(self.precision, mpi_size=self.size)
            extra = f" (GPU fits up to ~{max_fit} qubits at this precision)" if max_fit else ""
            print(f"[Stack] Precision: {self.precision} for {num_qubits}-qubit problem{extra}")
            self._memory_risk_pending = False
            if self._gpu_sv and max_fit and num_qubits > max_fit:
                ranks_per_gpu = -(-self.size // max(self.hw.gpu_count, 1))
                print(
                    f"[Stack] MEMORY WARNING: {num_qubits}-qubit problem exceeds the estimated "
                    f"~{max_fit}-qubit GPU capacity ({self.size} rank(s) across "
                    f"{self.hw.gpu_count} GPU(s), ~{ranks_per_gpu} rank(s) sharing each "
                    f"card). Whether this pages/thrashes (slow but survives) or hard-crashes "
                    f"with a CUDA out-of-memory error depends on this GPU/driver's memory "
                    f"oversubscription support, which is not guaranteed across vendors -- "
                    f"do not assume it will merely be slow. Consider fewer ranks, "
                    f"VQE_PRECISION=fp32, or a larger-memory GPU before proceeding."
                )
                self._memory_risk_pending = True

            # Compute-cost pre-flight: SPSA needs 2 statevector builds per iter, and each
            # rank evaluates its Pauli-term slice against that statevector. Warns about
            # the exact CO2 failure mode -- 16k Pauli terms * 240 params * default max_iters
            # would otherwise commit to a multi-hour run at $$$/hr. See gap H in
            # docs/KNOWN_GAPS.md: only Pauli evaluation is distributed, statevector
            # construction is redundant per rank.
            n_pauli = len(problem.pauli_terms)
            evals_per_iter = 2  # SPSA theta_plus + theta_minus
            total_pauli_evals = max_iterations * evals_per_iter * n_pauli
            # very rough cost proxy: each Pauli eval touches all 2^n amplitudes; per-iter
            # work scales with (2^n) * n_pauli. Warn at ~10^11 amplitude touches per iter.
            per_iter_work = (2 ** num_qubits) * n_pauli
            if per_iter_work > 1e11 or total_pauli_evals > 1e10:
                print(
                    f"[Stack] LARGE-COST WARNING: {num_qubits}q x {n_pauli} Pauli terms "
                    f"x {max_iterations} iters = ~{total_pauli_evals:.1e} Pauli evaluations "
                    f"(~{per_iter_work:.1e} amplitude touches per iteration). This will "
                    f"likely take many hours per iteration on GPU and cannot be interrupted "
                    f"safely mid-iter. Consider MAX_ITERS<=10 for ceiling tests, reps=1 to "
                    f"halve params, or wait for distributed statevector (docs/FUTURE_WORK.md #2)."
                )
                self._large_cost_pending = True
            else:
                self._large_cost_pending = False
        else:
            self._memory_risk_pending = False
            self._large_cost_pending = False

        # The warnings above used to only print and continue -- despite the code
        # comment claiming it "blocks the exact CO2 failure mode," nothing in
        # the control flow actually stopped execution. Gate on an explicit,
        # rank-uniform opt-in instead: rank 0's decision is broadcast so every
        # rank agrees on whether to proceed (matches this run's earlier
        # rank-agreement fix in make_problem() -- an abort/continue decision
        # that only some ranks take would itself cause a divergent-collective
        # hang, the same failure mode this is trying to prevent).
        _large_cost_flag = np.array([1 if getattr(self, "_large_cost_pending", False) else 0], dtype=np.int32)
        comm.Bcast(_large_cost_flag, root=0)
        if _large_cost_flag[0] and os.environ.get("VQE_ACCEPT_COST", "").strip() not in {"1", "yes", "true"}:
            if self.rank == 0:
                print(
                    "[Stack] Refusing to start: set VQE_ACCEPT_COST=1 to proceed "
                    "anyway once you've reviewed the LARGE-COST WARNING above."
                )
            raise RuntimeError(
                "vqe_optimize() aborted by the compute-cost pre-flight check "
                "(see the LARGE-COST WARNING printed on rank 0). Set "
                "VQE_ACCEPT_COST=1 to proceed anyway."
            )

        _memory_risk_flag = np.array([1 if getattr(self, "_memory_risk_pending", False) else 0], dtype=np.int32)
        comm.Bcast(_memory_risk_flag, root=0)
        if _memory_risk_flag[0] and os.environ.get("VQE_ACCEPT_MEMORY_RISK", "").strip() not in {"1", "yes", "true"}:
            if self.rank == 0:
                print(
                    "[Stack] Refusing to start: set VQE_ACCEPT_MEMORY_RISK=1 to proceed "
                    "anyway once you've reviewed the MEMORY WARNING above."
                )
            raise RuntimeError(
                "vqe_optimize() aborted by the GPU-memory pre-flight check "
                "(see the MEMORY WARNING printed on rank 0). Set "
                "VQE_ACCEPT_MEMORY_RISK=1 to proceed anyway."
            )

        os.makedirs(checkpoint_dir, exist_ok=True)
        theta = np.zeros(num_params, dtype=np.float64)

        if self.rank == 0:
            checkpoint_path = (restart_from if restart_from and os.path.exists(restart_from) else None)

            if checkpoint_path:
                print(f"[RESILIENCE] Loading θ from {checkpoint_path}...")
                theta = np.load(checkpoint_path).astype(np.float64)
                if theta.shape[0] != num_params:
                    raise ValueError(f"Checkpoint has {theta.shape[0]} params but problem has {num_params} params ")
                import re
                match = re.search(r'checkpoint_iter_(\d+)', checkpoint_path)
                if match and start_iter == 0:
                    start_iter= int(match.group(1))
                    print(f"[RESILIENCE] Resuming SPSA schedle from iteration {start_iter}  ")
            else:
                theta = np.random.uniform(-0.1, 0.1, num_params)      # near zero - stay close to HF reference
        else:
            theta = np.zeros(num_params)

        comm.Bcast(theta, root=0)

        # SPSA hyperparameters (must match serial_baseline.py - fair comparison)
        c = 0.1
        a = 0.628 / np.sqrt(num_params / 8.0)
        A = max_iterations * 0.1
        alpha, gamma = 0.602, 0.101
        min_iters_before_convergence = max(10, min(num_params * 2, 50))
        history: list[float] = []
        prev_energy = float('inf')
        stop_signal = np.array([0], dtype=np.int32)


        _run_t0 = _time.perf_counter()          # wall-clock start of the whole run
        _iter_wall_history: list[float] = []    # rolling per-iter durations for ETA

        for loop_k in range(1, max_iterations +1):
            k = loop_k + start_iter
            stop_signal[0] = 0
            # Declared on every rank (not just inside `if self.rank == 0:`
            # below) since the Bcast that reads/writes it after the loop
            # body must run on every rank -- see the comment at that Bcast.
            _ckpt_failed = np.array([0], dtype=np.int32)
            _iter_t0 = _time.perf_counter()

            # START-of-iter heartbeat -- for problems where a single iter
            # takes minutes (large molecules, 30q+), the end-of-iter print
            # is not enough to distinguish "loop running" from "hung".
            # Only heartbeat on rank 0 to keep the log clean under MPI.
            if self.rank == 0 and (num_qubits >= 20 or num_params >= 200):
                from datetime import datetime as _dt
                print(f"[iter {k:04d}] start {_dt.now().strftime('%H:%M:%S')} "
                      f"({num_qubits}q, {len(problem.pauli_terms)} Pauli terms, "
                      f"{num_params} params)", flush=True)

            # Iter boundary shows up on the nsys timeline as one region --
            # nested sv-build ranges appear inside it.
            _iter_range = _nvtx_range(message=f"spsa-iter-{k}", color="blue", domain="vqe")
            _iter_range.__enter__()

            combined_params = np.zeros(num_params*2, dtype=np.float64)
            ck = np.float64(0.0)

            if self.rank == 0:
                ak = a / (k + A)**alpha
                ck = np.float64(c / k**gamma)
                delta = np.random.choice([-1.0, 1.0], size=num_params)
                combined_params[:num_params] = theta + ck * delta
                combined_params[num_params:] = theta - ck * delta

            comm.Bcast(combined_params, root=0)
            ck_arr = np.array([ck], dtype=np.float64)
            comm.Bcast(ck_arr, root=0)
            ck = ck_arr[0]

            if self.backend == "simulator":
                e_plus, e_minus, masking_metric, used_path = self._evaluate_distributed_statevector(problem, combined_params)
            elif self.backend == "ibm_cloud":
                e_plus, e_minus, masking_metric, used_path = self._evaluate_ibm_estimator(problem, combined_params)
            else:         # C++ dispatcher fallback
                result = self._evaluate(problem, combined_params, num_qubits)
                e_plus = result.energy
                e_minus = result.e_minus
                used_path = result.used_path
                masking_metric = result.masking_metric

            if self.rank == 0:
                current_energy = (e_plus + e_minus) / 2.0

                # Track best energy in physical (above FCI sector)
                fci = getattr(problem, 'fci_energy', None)
                if fci is not None:
                    if current_energy >= fci - 1e-6:
                        if not hasattr(self, '_best_physical_energy') or current_energy < self._best_physical_energy:
                            self._best_physical_energy = current_energy
                            self._best_physical_theta = theta.copy()
                            self._best_physical_iter = k
                    elif not self._below_fci_warned:
                        print(f"NOTE: Energy {current_energy:.4f} crossed below FCI {fci:.4f} at iter {k} - HWE ansatz leaving physical sector (expected limitation)")
                        self._below_fci_warned = True

                gradient = (e_plus - e_minus) / (2*ck*delta)                 # SPSA finite difference gradient

                if not np.all(np.isfinite(gradient)):
                    print(f"WARNING: Non-finite gradient at iter {k}, skipping update")
                else:
                    theta = theta - ak * gradient

                delta_e = abs(current_energy - prev_energy)

                history.append(current_energy)
                prev_energy = current_energy

                _iter_wall = _time.perf_counter() - _iter_t0
                _iter_wall_history.append(_iter_wall)
                # Rolling median (last 5) is more stable than avg for ETA display.
                _recent = _iter_wall_history[-5:]
                _median = sorted(_recent)[len(_recent) // 2]
                _remaining = max_iterations - loop_k
                _eta_sec = int(_median * _remaining)
                _eta_str = f"{_eta_sec // 3600:d}h{(_eta_sec % 3600) // 60:02d}m" if _eta_sec >= 3600 else f"{_eta_sec // 60:d}m{_eta_sec % 60:02d}s"

                print(
                    f"Iter {k:04d}/{max_iterations} "
                    f"| E: {current_energy:.6f} | Δ: {delta_e:.2e} | M: {masking_metric:.4f} "
                    f"| iter={_iter_wall:6.2f}s | ETA: {_eta_str} "
                    f"| Path: {used_path}",
                    flush=True,
                )

                if loop_k >= min_iters_before_convergence and len(history) >= 10:
                    recent = history[-10:]
                    spread = max(recent) - min(recent)
                    if spread < tolerance:
                        print(f"Converged: energy spread over last 10 iters = {spread:.2e} < tol={tolerance}, at iteration {k}")
                        stop_signal[0] = 1

                # Adaptive checkpoint cadence: bound wall-clock time at risk
                # on interrupt, not iteration count. A fixed "every 5 iters"
                # is fine for small molecules (H2 at ~0.14s/iter loses under
                # a second) but on a much slower workload -- a bigger
                # molecule, a future non-chemistry application, UCCSD instead
                # of HWE -- 5 iterations could be many minutes, and losing
                # that on a crash is exactly the failure mode checkpointing
                # exists to prevent. Target: never checkpoint so rarely that
                # an interrupt risks more than ~_CKPT_MAX_SECONDS_AT_RISK of
                # recomputation, and never retain so little history that a
                # bad/corrupted latest checkpoint leaves no earlier fallback
                # within ~_CKPT_MIN_HISTORY_SECONDS.
                # Ceiling of 5 preserves the original fixed cadence as an
                # upper bound: on fast workloads (small molecules, short
                # smoke/diagnostic runs -- e.g. Experiment 6's 5-iteration
                # resilience test, which hard-asserts a checkpoint exists at
                # iteration 5) this never checkpoints LESS often than before,
                # only more often as iterations get slower. Without this cap,
                # a fast per-iter time would push the interval past
                # max_iterations entirely and a short run would never
                # checkpoint at all.
                _CKPT_MAX_SECONDS_AT_RISK = 120.0   # cap time lost on interrupt
                _CKPT_MIN_HISTORY_SECONDS = 600.0   # cap total rollback depth
                _CKPT_INTERVAL_CEILING = 5          # never checkpoint less often than this
                if _iter_wall_history:
                    _median_iter_s = sorted(_iter_wall_history[-5:])[len(_iter_wall_history[-5:]) // 2]
                    checkpoint_every = max(1, min(_CKPT_INTERVAL_CEILING,
                                                   round(_CKPT_MAX_SECONDS_AT_RISK / max(_median_iter_s, 1e-6))))
                    retain_n = max(5, min(50, round(_CKPT_MIN_HISTORY_SECONDS / (checkpoint_every * max(_median_iter_s, 1e-6)))))
                else:
                    checkpoint_every, retain_n = 5, 5   # no timing data yet (first iter) -- old default

                if k % checkpoint_every == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{k:04d}.npy")
                    try:
                        np.save(ckpt_path, theta)
                        print(f"[RESILIENCE] Iteration {k}: Global theta state checkpointed at path "
                              f"{ckpt_path} (every {checkpoint_every} iters, retaining last {retain_n}). ")
                        existing = sorted(_glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter_*.npy")))
                        for old in existing[:-retain_n]:
                            os.remove(old)
                    except OSError as e:
                        print(f"[Stack] rank 0: checkpoint write failed at iteration {k}: {e}",
                              file=sys.stderr, flush=True)
                        _ckpt_failed[0] = 1

            # A failed checkpoint write (disk full, permission error,
            # unwritable mount) must not let rank 0 die inside the
            # `if self.rank == 0:` block above while every other rank is
            # about to block in the Bcast calls below -- that's a silent,
            # indefinite hang with no timeout, the same divergent-collective
            # failure mode the cost-preflight Bcast earlier in this method
            # already guards against. This Bcast (and the check after it)
            # must run on EVERY rank, not just rank 0 -- that's why it's
            # outside the `if self.rank == 0:` block, mirroring theta/
            # stop_signal's Bcasts immediately below.
            comm.Bcast(_ckpt_failed, root=0)
            if _ckpt_failed[0]:
                raise RuntimeError(
                    f"Checkpoint write failed on rank 0 at iteration {k} "
                    f"(disk full, permission error, or unwritable "
                    f"checkpoint_dir='{checkpoint_dir}') -- aborting all "
                    f"ranks instead of hanging on the next collective call."
                )

            comm.Bcast(theta, root=0)
            comm.Bcast(stop_signal, root=0)

            _iter_range.__exit__(None, None, None)

            if stop_signal[0] == 1:
                break

        if self.rank == 0 and fci is not None and hasattr(self, '_best_physical_energy'):
            if history[-1] < fci - 1e-6:
                best_e = self._best_physical_energy
                best_iter = self._best_physical_iter
                best_err = abs(best_e - fci)
                print(f"[VQE] Best physical energy: {best_e:.6f} Ha (error: {best_err:.4f} Ha) at iter {best_iter}")
                print(f"[VQE] Final energy {history[-1]:.6f} is below FCI - reporting best physical result")

        return theta, history



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception on THIS rank means every other rank is likely
            # still waiting inside a collective call (comm.Bcast/Allreduce)
            # that this rank will now never reach -- finalize_mpi() /
            # MPI_Finalize() is itself collective, so calling it here would
            # just add a second, different hang on top of the first. Abort
            # the whole job instead of trying to finalize cleanly: a loud,
            # immediate failure on every rank beats a silent, permanent
            # deadlock with no error message (this was reachable from, e.g.,
            # an IBM API error, a disk-full checkpoint write, or a missing
            # IBM_QUANTUM_TOKEN raised only on rank 0 -- see
            # _init_ibm_session()).
            print(f"[Stack] rank {getattr(self, 'rank', '?')}: unhandled "
                  f"{exc_type.__name__}: {exc_val} -- aborting all MPI ranks "
                  f"rather than risk a silent hang in the other ranks' next "
                  f"collective call.", file=sys.stderr, flush=True)
            try:
                self.comm.Abort(1)
            except Exception:
                pass
            return False
        self.finalize()

    def finalize(self):
        if self._ibm_session is not None:
            try:
                self._ibm_session.close()
                if self.rank == 0:
                    print("[IBM] Session closed.")
            except Exception:
                pass
            self._ibm_session = None
            self._ibm_estimator = None
        hpc_core.finalize_mpi()




    def _evaluate(self, problem: QuantumProblem, combined_params: np.ndarray, num_qubits:int):
        # Dispatch to C++ dispatcher  (MPI broadcast+CUDA/CPU kernel) 
        workload = hpc_core.HybridWorkload()
        workload.parameters = combined_params.tolist()
        workload.num_qubits = num_qubits
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm
        workload.backend_target = self.backend

        local_pauli_terms = self.partition(problem.pauli_terms)
        workload.pauli_terms = [hpc_core.PauliTerm(op, coeff) for op, coeff in local_pauli_terms]

        return hpc_core.execute(workload)


    def evaluate(self, problem, combined_params, num_qubits):
        return self._evaluate(problem, combined_params, num_qubits)


    def partition(self, full_list:list) -> list:
        n= len(full_list)
        chunk = n // self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]




    def _evaluate_distributed_statevector(self, problem, combined_params):
        import time as _time
        comm = self.comm
        num_params = len(combined_params) // 2
        theta_plus = combined_params[:num_params]
        theta_minus = combined_params[num_params:]

        ansatz = problem.ansatz_circuit
        sorted_params = sorted(ansatz.parameters, key=lambda x: x.name)

        # T_accel: statevector build + Pauli expectation  ("acceleration" work)
        t_accel_start = _time.perf_counter()

        bound_plus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_params, theta_plus)})
        bound_minus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_params, theta_minus)})

        # Partition Pauli terms across MPI ranks (needed either path)
        local_terms = self.partition(problem.pauli_terms)

        # Two evaluation paths:
        # - GPU path: Aer's save_expectation_value instruction runs the
        #   expectation on GPU inside the same run() as the SV build. No 2^n
        #   GPU->CPU copy, no CPU-side numpy expectation. Beats legacy by 1.5x
        #   at NP=1 (validated 2026-09-01: H2O 22.99s -> 15.24s).
        # - Legacy path: build SV, copy to CPU, expectation_value in numpy.
        #   Slower at NP=1 but scales cleanly under MPI.
        #
        # DEFAULT POLICY (fixed 2026-09-04 after measured MPI regression):
        #   - NP=1: GPU-native path (proven faster)
        #   - NP>=2: legacy path (GPU-native has a 3.6x per-iter regression
        #     under MPI on H2O; root cause = per-rank Aer transpile cost of
        #     save_expectation_value amplifies when multiple ranks contend
        #     for the same GPU. See docs/GPU_EXPECTATION_FIX.md for the full
        #     measurement + a note on the tabled fix in FUTURE_WORK.md #7.2).
        #
        # Explicit override:
        #   VQE_LEGACY_EXPECT=1   -> force legacy path (was original opt-out)
        #   VQE_GPU_EXPECT_MPI=1  -> force GPU-native even under MPI
        #                            (for A/B measurement + future validation
        #                            of a fix for the transpile regression)
        _env = lambda k: os.environ.get(k, "").strip() in {"1", "yes", "true"}
        _force_legacy = _env("VQE_LEGACY_EXPECT")
        _force_gpu_under_mpi = _env("VQE_GPU_EXPECT_MPI")
        _mpi_safe_default = (self.size == 1) or _force_gpu_under_mpi

        _use_gpu_expect = (
            self._gpu_sv
            and len(local_terms) > 0
            and not _force_legacy
            and _mpi_safe_default
        )

        if _use_gpu_expect:
            e_plus_local, e_minus_local = self._expectation_on_gpu(
                bound_plus, bound_minus, local_terms)
        else:
            if len(local_terms) > 0:
                sv_plus = self._build_statevector(bound_plus)
                sv_minus = self._build_statevector(bound_minus)
                local_op = SparsePauliOp.from_list(local_terms)
                e_plus_local = float(sv_plus.expectation_value(local_op).real)
                e_minus_local = float(sv_minus.expectation_value(local_op).real)
            else:
                e_plus_local = 0.0
                e_minus_local = 0.0

        t_accel = _time.perf_counter() - t_accel_start

        # T_comm: MPI_Allreduce communication time
        t_comm_start = _time.perf_counter()
        e_plus_global = np.array([0.0], dtype=np.float64)
        e_minus_global = np.array([0.0], dtype=np.float64)
        comm.Allreduce(np.array([e_plus_local], dtype=np.float64), e_plus_global, op=MPI.SUM)
        comm.Allreduce(np.array([e_minus_local], dtype=np.float64), e_minus_global, op=MPI.SUM)
        t_comm = _time.perf_counter() - t_comm_start

        # M = T_accel / T_comm    -  measures how well compute masks communication
        masking_metric = (t_accel / t_comm) if t_comm > 1e-9 else 0.0

        if self._gpu_sv:
            sv_type = "GPU-cuStateVec+native-expect" if _use_gpu_expect else "GPU-cuStateVec+cpu-expect"
        else:
            sv_type = "CPU-Statevector"
        path = f"{sv_type} MPI-distributed ({self.size} ranks)"
        return float(e_plus_global[0]), float(e_minus_global[0]), masking_metric, path




    def _evaluate_statevector(self, problem, theta):
        # Single-rank exact statevector evaluation      (kept for standalone use)
        ansatz = problem.ansatz_circuit
        bound = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted(ansatz.parameters, key=lambda x: x.name), theta)})

        sv = self._build_statevector(bound)
        pauli_op = SparsePauliOp.from_list(problem.pauli_terms)
        energy = sv.expectation_value(pauli_op).real
        return float(energy)

    def evaluate_unperturbed_energy(self, problem, theta) -> float:
        """Compute the exact, unperturbed E(theta) -- rank 0 only, no MPI needed.

        Every per-iteration energy inside vqe_optimize() is
        (E(theta+ck*delta) + E(theta-ck*delta)) / 2, SPSA's gradient-estimate
        quantity, NOT E(theta) itself. That average has an expected bias of
        +(ck^2/2)*tr(Hessian) that does not decay to zero over a realistic
        run (ck ~ c/k^gamma with gamma=0.101 is still >50% of its initial
        value after hundreds of iterations) -- it is the correct quantity
        for driving the optimizer, but not a correct quantity to report as
        "the energy" for accuracy claims.

        Call this once, after vqe_optimize() returns, on the final theta (or
        on _best_physical_theta if the run's final trajectory point is below
        FCI) to get a real, reproducible, provenance-labeled energy for
        accuracy reporting. Only rank 0 needs to call this -- it does not
        touch MPI at all, so it's safe to call unconditionally on rank 0
        after a distributed run finishes, using whichever theta you want to
        re-evaluate.
        """
        return self._evaluate_statevector(problem, theta)


    def _init_ibm_session(self, problem):
        # Lazy-init - connect to IBM Quantum, transpile ansatz once, cache layout 
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        token = os.environ.get("IBM_QUANTUM_TOKEN", "")
        backend_name = os.environ.get("IBM_QUANTUM_BACKEND", "")
        region = os.environ.get("IBM_QUANTUM_REGION", "us-east")

        if not token:
            raise RuntimeError(
                "[IBM] IBM_QUANTUM_TOKEN is not set. "
                "Add it to your .env file (see .env.example)."
            )
        if not backend_name:
            raise RuntimeError(
                "[IBM] IBM_QUANTUM_BACKEND is not set. "
                "Set it to the backend you have access to (e.g. ibm_brisbane, ibm_kyoto). "
                "Check available backends at https://quantum.cloud.ibm.com"
            )

        print(f"[IBM] Connecting to {backend_name} ({region}) ...")
        QiskitRuntimeService.save_account(token=token, overwrite=True, set_as_default=True)  # required by v0.45+
        service = QiskitRuntimeService()
        self._ibm_backend = service.backend(backend_name)

        pm = generate_preset_pass_manager(backend=self._ibm_backend, optimization_level=1)
        ansatz = problem.ansatz_circuit
        self._ibm_transpiled = pm.run(ansatz)
        print(f"[IBM] Ansatz transpiled: {self._ibm_transpiled.num_qubits} physical qubits, depth {self._ibm_transpiled.depth()}")

        pauli_op = SparsePauliOp.from_list(problem.pauli_terms)
        self._ibm_observable = pauli_op.apply_layout(self._ibm_transpiled.layout)           # remap to physical qubits

        # mode=backend (not Session,  open plan has no session support)
        self._ibm_estimator = EstimatorV2(mode=self._ibm_backend)
        self._ibm_estimator.options.default_shots = 4096
        print(f"[IBM] EstimatorV2 ready (4096 shots)")




    def _evaluate_ibm_estimator(self, problem, combined_params):
        # Submit QPU job, compute statevector in parallel, collect QPU result
        comm = self.comm
        num_params = len(combined_params) // 2
        theta_plus = combined_params[:num_params]
        theta_minus = combined_params[num_params:]

        job = None
        t_qpu_start = _time.perf_counter()

        if self.rank == 0:
            if self._ibm_estimator is None:
                self._init_ibm_session(problem)

            isa_circuit = self._ibm_transpiled
            observable = self._ibm_observable
            sorted_params = sorted(isa_circuit.parameters, key=lambda x: x.name)
            binds_plus = {p: float(v) for p, v in zip(sorted_params, theta_plus)}
            binds_minus = {p: float(v) for p, v in zip(sorted_params, theta_minus)}

            pubs = [
                (isa_circuit, observable, binds_plus),
                (isa_circuit, observable, binds_minus),
            ]
            print(f"[IBM] Submitting job (2 PUBs, 4096 shots)...")
            job = self._ibm_estimator.run(pubs)
            print(f"[IBM] Job submitted: {job.job_id()}, classical work proceeding...")


        # All rank compute statevector concurrently with QPU (T_accel)
        t_accel_start = _time.perf_counter()

        ansatz = problem.ansatz_circuit
        sorted_ansatz_params = sorted(ansatz.parameters, key=lambda x: x.name)

        bound_plus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_ansatz_params, theta_plus)})
        bound_minus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_ansatz_params, theta_minus)})

        local_terms = self.partition(problem.pauli_terms)

        # Same GPU-native vs legacy dispatch as _evaluate_distributed_statevector,
        # including the MPI-safe default policy fixed 2026-09-04:
        # GPU-native at NP=1, legacy at NP>=2 (avoids the per-rank Aer transpile
        # regression that showed 3.6x slowdown on H2O at NP=2). See the
        # equivalent block in _evaluate_distributed_statevector for the full
        # rationale. VQE_LEGACY_EXPECT=1 forces legacy; VQE_GPU_EXPECT_MPI=1
        # forces GPU-native regardless of rank count.
        _env = lambda k: os.environ.get(k, "").strip() in {"1", "yes", "true"}
        _force_legacy = _env("VQE_LEGACY_EXPECT")
        _force_gpu_under_mpi = _env("VQE_GPU_EXPECT_MPI")
        _mpi_safe_default = (self.size == 1) or _force_gpu_under_mpi

        _use_gpu_expect = (
            self._gpu_sv
            and len(local_terms) > 0
            and not _force_legacy
            and _mpi_safe_default
        )

        if _use_gpu_expect:
            e_plus_sv_local, e_minus_sv_local = self._expectation_on_gpu(
                bound_plus, bound_minus, local_terms)
        else:
            sv_plus = self._build_statevector(bound_plus)
            sv_minus = self._build_statevector(bound_minus)
            if len(local_terms) > 0:
                local_op = SparsePauliOp.from_list(local_terms)
                e_plus_sv_local = float(sv_plus.expectation_value(local_op).real)
                e_minus_sv_local = float(sv_minus.expectation_value(local_op).real)
            else:
                e_plus_sv_local = 0.0
                e_minus_sv_local = 0.0

        e_plus_sv = np.array([0.0], dtype=np.float64)
        e_minus_sv = np.array([0.0], dtype=np.float64)
        comm.Allreduce(np.array([e_plus_sv_local], dtype=np.float64), e_plus_sv, op=MPI.SUM)
        comm.Allreduce(np.array([e_minus_sv_local], dtype=np.float64), e_minus_sv, op=MPI.SUM)

        t_accel = _time.perf_counter() - t_accel_start


        # Rank 0 blocks on QPU result for remaining T_quant
        result_buf = np.zeros(6, dtype=np.float64)              # [e+_qpu, e-_qpu, M, t_quant, e+_sv, e-_sv]

        if self.rank == 0:
            # job.result() has no built-in timeout in qiskit-ibm-runtime, and
            # every other rank is already past its last collective call
            # (comm.Allreduce above) by the time we get here -- an IBM job
            # stuck in QUEUED, or a dropped connection that never resolves,
            # would block this rank forever with the others waiting on
            # nothing. Same 1200s (20 min) budget as the C++ QPU client's
            # own timeout (qpu_client.cpp:202) for consistency. Run the
            # blocking call in a background thread so a timeout can
            # actually interrupt the wait; on timeout, raise so __exit__'s
            # comm.Abort() (see class docstring / __exit__) tears down
            # every rank instead of leaving them all hung.
            import concurrent.futures as _futures
            _IBM_JOB_TIMEOUT_S = 1200
            # Deliberately NOT a `with` block: ThreadPoolExecutor.__exit__
            # calls shutdown(wait=True) by default, which blocks until the
            # background thread finishes -- exactly what we're trying to
            # avoid if job.result() is genuinely stuck rather than just
            # slow. shutdown(wait=False) below returns immediately instead.
            # Note this can't forcibly kill the underlying network call --
            # Python threads aren't killable -- so on a real timeout the
            # HTTP request keeps running in the background until it
            # naturally resolves or the process exits; we just stop
            # blocking on it here rather than waiting for that to happen.
            _pool = _futures.ThreadPoolExecutor(max_workers=1)
            _future = _pool.submit(job.result)
            try:
                job_result = _future.result(timeout=_IBM_JOB_TIMEOUT_S)
            except _futures.TimeoutError:
                _pool.shutdown(wait=False)
                raise TimeoutError(
                    f"[IBM] job {job.job_id()} did not complete within "
                    f"{_IBM_JOB_TIMEOUT_S}s -- aborting rather than "
                    f"blocking indefinitely while other MPI ranks wait. "
                    f"Check the job status at quantum.ibm.com; it may "
                    f"still complete after this process exits."
                )
            _pool.shutdown(wait=False)
            t_qpu = _time.perf_counter() - t_qpu_start

            e_plus_qpu = float(job_result[0].data.evs)
            e_minus_qpu = float(job_result[1].data.evs)

            masking_metric = (t_accel / t_qpu) if t_qpu > 1e-9 else 0.0
            residual_wait = max(0.0, t_qpu - t_accel)

            print(f"[IBM] Job completed in {t_qpu:.1f}s "
                  f"(T_accel={t_accel:.2f}s, residual_wait={residual_wait:.1f}s, M={masking_metric:.4f})")
            print(f"[IBM] Classical SV estimate: E+={float(e_plus_sv[0]):.6f}, E-={float(e_minus_sv[0]):.6f}")
            print(f"[IBM] QPU measured: E+={e_plus_qpu:.6f}, E-={e_minus_qpu:.6f}")

            result_buf[0] = e_plus_qpu
            result_buf[1] = e_minus_qpu
            result_buf[2] = masking_metric
            result_buf[3] = t_qpu
            result_buf[4] = float(e_plus_sv[0])
            result_buf[5] = float(e_minus_sv[0])

        comm.Bcast(result_buf, root=0)
        e_plus = result_buf[0]
        e_minus = result_buf[1]
        masking_metric = result_buf[2]
        used_path = "IBM QPU + Classical SV (async)"

        return e_plus, e_minus, masking_metric, used_path


# Backward-compatibility alias: this class was renamed from HPCHybridStack
# to QatabasisStack in the Qatabasis rebrand. Legacy imports still work.
HPCHybridStack = QatabasisStack
