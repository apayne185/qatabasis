"""Hardware auto-detection and policy layer.

Probes GPU vendor, class, memory, and MPI availability at stack init.
Derived decisions (precision, backend, qubit limits) are exposed via
helper methods on HardwareProfile so the rest of the stack doesn't need
to know about hardware details.

Researchers override auto-detection via env vars:
  VQE_PRECISION  = auto | fp32 | fp64 | mixed
  VQE_BACKEND    = auto | cuStateVec | aer_gpu | aer_cpu
  USE_GPU        = yes  | no    (existing; takes precedence over detection)
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from dataclasses import dataclass, field


# Known GPU families: (substring, class, est. fp64:fp32 ratio)
# Datacenter cards have full FP64 and do not benefit from mixed precision.
_GPU_DATABASE: list[tuple[str, str, float]] = [
    ("A100", "datacenter", 1 / 2),
    ("H100", "datacenter", 1 / 2),
    ("V100", "datacenter", 1 / 2),
    ("A10G", "workstation", 1 / 32),  # AWS g5.xlarge -- Ampere die, but fp64
                                       # is crippled like A40, not full like
                                       # A100/H100 despite being sold as
                                       # "datacenter" hardware by AWS.
    ("A40", "workstation", 1 / 32),
    ("RTX 6000 Ada", "workstation", 1 / 64),
    ("RTX A6000", "workstation", 1 / 32),
    ("RTX 4090", "consumer", 1 / 64),
    ("RTX 4080", "consumer", 1 / 64),
    ("RTX 3090", "consumer", 1 / 64),
    ("RTX 3080", "consumer", 1 / 64),
    ("GTX 1650", "consumer", 1 / 32),
    ("GTX 1660", "consumer", 1 / 32),
]


@dataclass
class HardwareProfile:
    has_cuda: bool = False
    gpu_name: str = ""
    gpu_class: str = "unknown"       # datacenter | workstation | consumer | unknown
    hostname: str = ""
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0                # number of physical GPUs on this node
    fp64_ratio: float = 1 / 32        # estimated FP64:FP32 throughput
    compute_capability: tuple[int, int] | None = None
    has_cuquantum: bool = False
    has_aer_gpu: bool = False
    has_mpi: bool = False
    mpi_size: int = 1

    # User overrides captured from env vars
    override_precision: str = "auto"
    override_backend: str = "auto"
    override_use_gpu: str = ""

    notes: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls) -> HardwareProfile:
        p = cls()
        p.override_precision = os.environ.get("VQE_PRECISION", "auto").strip().lower()
        p.override_backend = os.environ.get("VQE_BACKEND", "auto").strip().lower()
        p.override_use_gpu = os.environ.get("USE_GPU", "").strip().lower()
        p.hostname = socket.gethostname()
        p._detect_gpu()
        p._detect_libs()
        p._detect_mpi()
        return p

    # ---------- probes ----------

    def _detect_gpu(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.notes.append("nvidia-smi unavailable; assuming CPU-only")
            return

        if out.returncode != 0 or not out.stdout.strip():
            self.notes.append("nvidia-smi found no GPU")
            return

        lines = out.stdout.strip().splitlines()
        self.gpu_count = len(lines)
        first = lines[0].split(",")
        self.has_cuda = True
        self.gpu_name = first[0].strip()
        try:
            self.gpu_memory_gb = float(first[1].strip()) / 1024.0
        except (ValueError, IndexError):
            pass
        try:
            major, minor = first[2].strip().split(".")
            self.compute_capability = (int(major), int(minor))
        except (ValueError, IndexError):
            pass

        for sub, cls_, ratio in _GPU_DATABASE:
            if sub in self.gpu_name:
                self.gpu_class = cls_
                self.fp64_ratio = ratio
                return
        self.notes.append(f"GPU '{self.gpu_name}' not in database; "
                          "conservative defaults applied")

    def _detect_libs(self) -> None:
        try:
            from qiskit_aer import AerSimulator
            try:
                AerSimulator(method="statevector", device="GPU")
                self.has_aer_gpu = True
            except Exception:
                pass
        except ImportError:
            pass
        try:
            import cuquantum  # noqa: F401
            self.has_cuquantum = True
        except ImportError:
            pass

    def _detect_mpi(self) -> None:
        try:
            from mpi4py import MPI
            self.has_mpi = True
            # MPI may not be initialized yet (interface.py defers init to hpc_core.init_mpi).
            # Only query the communicator size if MPI is already up; otherwise the C-level
            # MPI_Comm_size call aborts before MPI_Init has run.
            if MPI.Is_initialized():
                self.mpi_size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            pass

    # ---------- policy decisions ----------

    def want_gpu(self) -> bool:
        """Whether the stack should attempt to use the GPU path."""
        if self.override_use_gpu == "no":
            return False
        if self.override_use_gpu == "yes":
            return self.has_cuda
        return self.has_cuda

    def recommend_precision(self, num_qubits: int) -> str:
        """Pick fp32 or fp64 based on hardware class and problem size.

        Rules:
          - user override always wins (VQE_PRECISION=fp32 | fp64)
          - no GPU → fp64 (CPU Statevector has no reduced-precision path)
          - consumer/workstation GPU AND < 20 qubits → fp32 (these cards have
            fp64:fp32 throughput as low as 1/64 -- see fp64_ratio in
            _GPU_DATABASE -- fp32 is a large, mostly-free throughput win at
            small qubit counts)
          - everything else (datacenter GPU, or ≥20 qubits) → fp64. Datacenter
            cards (A100/H100/V100) have fp64_ratio=1/2, so there's little
            throughput to gain; large qubit counts want the extra numerical
            margin regardless of hardware class.

        NOTE: this was previously unconditional ("return fp64" always,
        despite this docstring) -- no run in this repo's history has ever
        exercised the fp32 path. Before trusting fp32 results, validate the
        accuracy delta empirically against a known fp64 run (same molecule,
        same seed) -- fp32's ~1e-7 relative precision, accumulated across a
        2^n-term statevector reduction, is not obviously negligible against
        the 1.6 mHa chemical-accuracy threshold and hasn't been measured.
        """
        if self.override_precision in {"fp32", "fp64"}:
            return self.override_precision
        if not self.has_cuda:
            return "fp64"
        if self.gpu_class in {"consumer", "workstation"} and num_qubits < 20:
            return "fp32"
        return "fp64"

    def recommend_backend(self) -> str:
        """Pick a simulator backend implementation."""
        if self.override_backend in {"custatevec", "aer_gpu", "aer_cpu"}:
            return self.override_backend
        if self.has_cuda and self.has_cuquantum and self.has_aer_gpu:
            return "custatevec"
        if self.has_cuda and self.has_aer_gpu:
            return "aer_gpu"
        return "aer_cpu"

    def max_qubits_fit(self, precision: str = "fp64", mpi_size: int = 1) -> int:
        """Upper bound on qubit count that fits in GPU memory.

        mpi_size matters: CUDA device assignment is round-robin per rank
        (rank % gpu_count, see bridge.cpp:set_cuda_device), so multiple ranks
        can land on the *same* physical GPU and each build their own full
        redundant statevector concurrently (this stack does not distribute
        statevector construction itself, only the Pauli-term summation --
        see CLAUDE.md gap H). A single-simulation budget silently ignores
        that and can look like it "fits" when it doesn't once a second rank
        is sharing the card. Pass the real QatabasisStack.size here, not
        self.mpi_size (populated at HardwareProfile.detect() time, which
        happens *before* MPI is initialized in QatabasisStack.__init__, so
        it's unreliable -- almost always stale at 1).
        """
        # fp64 statevector: 16 bytes/amplitude (complex128)
        # fp32 statevector:  8 bytes/amplitude (complex64)
        bytes_per_amp = 16 if precision == "fp64" else 8
        if self.gpu_memory_gb <= 0:
            return 0
        ranks_per_gpu = -(-max(mpi_size, 1) // max(self.gpu_count, 1))  # ceil division
        # Reserve ~25% for gates / Qiskit overhead, then split what's left
        # across however many ranks are sharing this physical GPU.
        usable = self.gpu_memory_gb * 0.75 * (1024 ** 3) / ranks_per_gpu
        n = 0
        while (2 ** (n + 1)) * bytes_per_amp <= usable:
            n += 1
        return n

    def results_slug(self) -> str:
        """Filesystem-safe hardware identifier used to route results into
        results/<slug>/... so runs from different GPUs (or CPU-only) never
        land in the same directory without being distinguishable.
        """
        if not self.has_cuda or not self.gpu_name:
            return "cpu-only"
        name = re.sub(r"(?i)^nvidia\s+", "", self.gpu_name)
        name = re.sub(r"(?i)\bmobile\b", "", name)
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return slug or "gpu-unknown"

    def describe(self) -> str:
        gpu = (
            f"{self.gpu_name} ({self.gpu_class}, {self.gpu_memory_gb:.1f} GB, "
            f"fp64:fp32≈{self.fp64_ratio:.3f})"
            if self.has_cuda else "no GPU (CPU only)"
        )
        libs = []
        if self.has_aer_gpu: libs.append("aer-gpu")
        if self.has_cuquantum: libs.append("cuquantum")
        libs_str = ", ".join(libs) if libs else "none"
        return (
            f"[hw] {gpu} | libs: {libs_str} | "
            f"MPI={self.mpi_size if self.has_mpi else 'off'} | "
            f"backend={self.recommend_backend()}"
        )