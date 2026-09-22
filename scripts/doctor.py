"""make doctor -- single-command readiness check for a new environment.

Runs inside the vqe-mpi-gpu container (via `make doctor`, which passes
--gpus all the same way `make trial`/`make run` do) so GPU visibility is
checked the same way the real workload sees it, not from the Docker host.

Checks, in order: CPU architecture, GPU detection + database coverage, MPI,
IBM credentials (if present). Exits 0 only if nothing found is a hard
blocker -- unmatched GPU, non-x86_64 architecture, and missing IBM
credentials are warnings (CPU-only / simulator-only use is a legitimate way
to run this stack), not failures.
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


_CUDA_DRIVER_FLOOR = (560, 28)  # minimum host driver for the CUDA 12.6.3
                                 # base image this stack's Dockerfile uses
                                 # (NVIDIA's published minimum-driver
                                 # requirement for that CUDA toolkit version)


def _check_cuda_driver_floor() -> None:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # already reported as "no GPU" by the caller
    if out.returncode != 0 or not out.stdout.strip():
        return

    driver_str = out.stdout.strip().splitlines()[0]
    try:
        parts = tuple(int(p) for p in driver_str.split(".")[:2])
    except ValueError:
        _warn(f"Could not parse host driver version '{driver_str}' to check "
              f"against the CUDA 12.6.3 minimum-driver requirement.")
        return

    if parts < _CUDA_DRIVER_FLOOR:
        _warn(f"Host NVIDIA driver {driver_str} is older than "
              f"{'.'.join(map(str, _CUDA_DRIVER_FLOOR))}, the minimum for "
              f"CUDA 12.6.3 (this stack's Dockerfile base image). The "
              f"container will still BUILD fine (driver isn't checked at "
              f"build time) but CUDA context init may fail mid-run with a "
              f"cryptic 'CUDA driver version is insufficient for CUDA "
              f"runtime version' error. Update the host driver, or ask "
              f"your cloud provider for an image with a newer one.")
    else:
        _ok(f"Host NVIDIA driver {driver_str} meets the CUDA 12.6.3 floor "
            f"({'.'.join(map(str, _CUDA_DRIVER_FLOOR))}+)")


def check_arch() -> None:
    import platform

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        _ok(f"Architecture: {machine} (supported)")
        return

    # This stack's GPU acceleration path depends on x86_64/manylinux-only
    # wheels (cupy-cuda12x, the pinned nvidia-*-cu12 packages in
    # environment.yml, qiskit-aer's CUDA source build) -- none of these
    # publish aarch64 distributions. An ARM host (AWS Graviton, some
    # GCP/Azure ARM SKUs) would otherwise hit a bare pip resolver error
    # ("no matching distribution found") with no pointer to why. CPU-only
    # use is unaffected -- numpy/scipy/qiskit/pyscf all publish aarch64
    # wheels -- only the GPU acceleration path is x86_64-only today.
    _warn(f"Architecture: {machine} (NOT x86_64). This stack's GPU "
          f"acceleration path (cupy-cuda12x, qiskit-aer's CUDA build, "
          f"pinned nvidia-*-cu12 wheels) is x86_64-only -- none of these "
          f"publish {machine} distributions. CPU-only use should still "
          f"work (numpy/scipy/qiskit/pyscf all publish {machine} wheels), "
          f"but `make build`'s GPU-dependent pip installs will likely "
          f"fail with a raw 'no matching distribution found' error rather "
          f"than a clear message. Known limitation, not yet supported.")


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
    _check_cuda_driver_floor()

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

    _check_mpi_implementation()


def _check_mpi_implementation() -> None:
    # scripts/install_native.sh deliberately builds hpc_core against
    # conda's bundled mpich (not the host's system MPI, which may be
    # broken on some clusters -- see that script's own comments). The
    # scripts/slurm_*.sh scripts launch with `mpirun -bootstrap fork`,
    # an MPICH-specific flag -- OpenMPI's mpirun doesn't recognize it and
    # fails immediately with an "unrecognized argument" error. This
    # mismatch only bites the native/HPC-cluster path (Docker path always
    # uses its own bundled OpenMPI, matching the Dockerfile's `mpirun
    # --allow-run-as-root` invocations, which don't use -bootstrap), and
    # only if a cluster's default `mpirun` on PATH isn't the conda env's
    # mpich (e.g. the conda env isn't activated, or a `module load
    # openmpi` shadows it). Loud failure either way, not silent -- this
    # just gives an upfront pointer instead of a cryptic launcher error.
    import shutil
    import subprocess

    mpirun_path = shutil.which("mpirun")
    if not mpirun_path:
        return  # already reported as MPI-not-functional above

    try:
        out = subprocess.run([mpirun_path, "--version"], capture_output=True,
                              text=True, timeout=5)
        version_text = (out.stdout + out.stderr).lower()
    except (subprocess.TimeoutExpired, OSError):
        return

    if "mpich" in version_text:
        _ok(f"mpirun on PATH ({mpirun_path}) is MPICH -- compatible with "
            f"scripts/slurm_*.sh's `-bootstrap fork` flag")
    elif "open mpi" in version_text or "openrte" in version_text:
        _warn(f"mpirun on PATH ({mpirun_path}) is OpenMPI, not MPICH. This "
              f"is fine for the Docker path (make build/run/trial use "
              f"Docker's own bundled OpenMPI). If you're on the native/HPC "
              f"install path (scripts/install_native.sh, which links "
              f"hpc_core against conda's mpich): scripts/slurm_*.sh launch "
              f"with `mpirun -bootstrap fork`, an MPICH-only flag that "
              f"OpenMPI's mpirun will reject outright. Make sure the conda "
              f"env is activated (its mpich mpirun should shadow this one) "
              f"before running those scripts.")
    else:
        _warn(f"mpirun on PATH ({mpirun_path}) reports an unrecognized "
              f"implementation -- could not confirm MPICH vs OpenMPI "
              f"compatibility with scripts/slurm_*.sh's `-bootstrap fork` "
              f"flag.")


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

    for name, fn in [("Architecture", check_arch), ("GPU", check_gpu),
                      ("MPI", check_mpi), ("IBM QPU", check_ibm)]:
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
