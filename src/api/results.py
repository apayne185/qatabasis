# Structured JSON results persistence for  VQE runs 

import json
import os
import subprocess
from datetime import datetime


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _hw_attr(hw, name, default):
    if hw is None:
        return default
    v = getattr(hw, name, None)
    return v if v is not None else default


def _hpc_cuda_build():
    # Read the compile-time CUDA flag from the C++ bridge. Safe on Python-only
    # builds -- returns None if the module can't be imported.
    try:
        import hpc_core
        return bool(hpc_core.cuda_build())
    except Exception:
        return None


def save_results(data: dict, backend: str, results_dir: str = "results", hw=None, stack=None, path: str | None = None) -> str:
    # Save run results as JSON, returns the file path.
    # Files are organized results/<hardware-slug>/<backend-subdir>/ so runs
    # from different GPUs (or CPU-only) are never mixed in the same directory.
    #
    # Pass `path` to write/overwrite a specific file instead of minting a new
    # timestamped one -- used for incremental saves during a long sweep, so a
    # script interrupted mid-run (Ctrl+C, crash, terminated instance) still
    # leaves the molecules that did finish on disk. See local_test_run.py's
    # call sites: same path is reused across incremental saves, then the
    # final save is just another call with that same path.

    if path is None:
        # Map backend to subdirectory
        subdir_map = {
            "simulator": "simulator",
            "ibm_cloud": "ibm",
            "serial_baseline": "baseline",
        }
        subdir = subdir_map.get(backend, backend)
        slug = hw.results_slug() if hw is not None else "unsorted"
        out_dir = os.path.join(results_dir, slug, subdir)
        os.makedirs(out_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{backend}_{ts}.json"
        path = os.path.join(out_dir, filename)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend,
        "git_commit": _git_commit(),
        "gpu_name": hw.gpu_name if hw is not None else "",
        "gpu_class": hw.gpu_class if hw is not None else "",
        "hostname": hw.hostname if hw is not None else "",
        "precision": getattr(stack, "precision", None) or _hw_attr(hw, "override_precision", "auto"),
        "cuda_build": _hpc_cuda_build(),
        "mpi_thread_level": getattr(stack, "provided_thread_level", None),
        "gpu_sv_available": getattr(stack, "_gpu_sv", None),
        **data,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[Results] Saved to {path}")
    return path
