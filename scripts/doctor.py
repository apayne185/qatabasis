"""make doctor -- single-command readiness check for a new environment.

Runs inside the vqe-mpi-gpu container (via `make doctor`, which passes
--gpus all the same way `make trial`/`make run` do) so GPU visibility is
checked the same way the real workload sees it, not from the Docker host.

Checks, in order: GPU detection + database coverage, MPI, IBM credentials
(if present). Exits 0 only if nothing found is a hard blocker -- unmatched
GPU and missing IBM credentials are warnings (CPU-only / simulator-only use
is a legitimate way to run this stack), not failures.
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

WARNINGS: list[str] = []
FAILURES: list[str] = []


def _ok(msg: str) -> None:
    print(f"[Doctor]  OK   {msg}")


def _warn(msg: str) -> None:
    print(f"[Doctor]  WARN {msg}")
    WARNINGS.append(msg)


def _fail(msg: str) -> None:
    print(f"[Doctor]  FAIL {msg}")
    FAILURES.append(msg)


def check_gpu() -> None:
    from src.api.hardware import HardwareProfile, _GPU_DATABASE

    hw = HardwareProfile.detect()
    if not hw.has_cuda:
        _warn("No GPU detected (nvidia-smi unreachable or absent). "
              "CPU-only mode will be used -- fine for correctness, slower "
              "for large molecules. If you expected a GPU here, run "
              "`bash scripts/cloud_bootstrap.sh` first (fixes the Docker "
              "permission issue that makes a real GPU look absent).")
        return

    _ok(f"GPU detected: {hw.gpu_name} ({hw.gpu_memory_gb:.1f} GB)")

    if hw.gpu_class == "unknown":
        _warn(f"GPU '{hw.gpu_name}' is not in _GPU_DATABASE "
              f"(src/api/hardware.py). Falling back to fp64_ratio="
              f"{hw.fp64_ratio} (conservative default) -- this MAY be "
              f"wrong for your card. Add an entry to _GPU_DATABASE with "
              f"the correct (class, fp64_ratio) if you know it, or check "
              f"NVIDIA's spec sheet for the fp64:fp32 throughput ratio.")
    else:
        _ok(f"GPU classified as '{hw.gpu_class}', fp64_ratio={hw.fp64_ratio} "
            f"(known card, {len(_GPU_DATABASE)} entries in database)")

    if not hw.has_aer_gpu:
        _warn("Qiskit Aer GPU backend not available in this environment "
              "(AerSimulator(device='GPU') failed to instantiate). GPU is "
              "visible but Aer-GPU may not be correctly installed.")
    else:
        _ok("Qiskit Aer GPU backend available")

    if not hw.has_cuquantum:
        _warn("cuQuantum/cuStateVec not importable -- some acceleration "
              "paths may be unavailable.")
    else:
        _ok("cuQuantum/cuStateVec available")


def check_mpi() -> None:
    from src.api.hardware import HardwareProfile

    hw = HardwareProfile.detect()
    if not hw.has_mpi:
        _fail("mpi4py not importable or MPI runtime not functional. "
              "This stack requires MPI even for NP=1 runs.")
        return
    _ok(f"MPI functional (mpi4py, this process sees size={hw.mpi_size})")

    try:
        import hpc_core
        _ok("hpc_core (C++ MPI bridge) importable")
        if hpc_core.cuda_build():
            _ok("hpc_core compiled with CUDA support")
        else:
            _warn("hpc_core compiled WITHOUT CUDA support -- GPU paths "
                  "will be unavailable regardless of GPU detection above.")
    except ImportError as e:
        _fail(f"hpc_core not importable: {e}. Run `make build` first.")


def check_ibm() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "")
    backend = os.environ.get("IBM_QUANTUM_BACKEND", "")

    if not (token or instance or backend):
        _warn("No IBM_QUANTUM_* credentials set -- IBM QPU backend "
              "unavailable (simulator/CPU-only use is unaffected). "
              "Copy .env.example to .env and fill in credentials if you "
              "want QPU access.")
        return

    missing = [n for n, v in [("IBM_QUANTUM_TOKEN", token),
                               ("IBM_QUANTUM_INSTANCE", instance),
                               ("IBM_QUANTUM_BACKEND", backend)] if not v]
    if missing:
        _fail(f"IBM credentials partially set but missing: {', '.join(missing)}")
        return

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
        be = service.backend(backend)
        _ok(f"IBM backend '{backend}' reachable "
            f"({be.num_qubits} qubits, operational={be.operational})")
    except Exception as e:
        _fail(f"IBM backend '{backend}' not reachable: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print(" QatabasisStack readiness check")
    print("=" * 60)

    for name, fn in [("GPU", check_gpu), ("MPI", check_mpi), ("IBM QPU", check_ibm)]:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            _fail(f"{name} check crashed: {e}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f" NOT READY -- {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
        for f in FAILURES:
            print(f"   FAIL: {f}")
        sys.exit(1)
    elif WARNINGS:
        print(f" READY (with {len(WARNINGS)} warning(s) -- review above)")
        sys.exit(0)
    else:
        print(" READY -- all checks passed cleanly")
        sys.exit(0)
