# Results layout

Results are organized **by hardware first, then by run/date**, so runs from
different GPUs (or CPU-only) never get mixed together in one directory:

```
results/
  <hardware-slug>/
    simulator/    run_<ts>.log + simulator_<ts>.json  (per-run VQE data)
    scaling/      scaling_P<N>.txt, weak_scaling_P<N>.txt
    ibm/          ibm_run_<ts>.log + ibm_cloud_<ts>.json
    baseline/     serial_baseline_<ts>.json (serial-baseline backend only)
    trial/        7-layer diagnostic logs (make trial / make slurm-trial)
    slurm/        raw Slurm job logs, if run via sbatch
    session-archives/  termination snapshots (code + git log) from a cloud
                       rental, kept for provenance
  baseline_comparison/           external-baseline sweep (hpchybrid vs
                                  Lightning-GPU vs Aer-MPI), one subfolder
                                  per backend: hpchybrid/, lightning/, aer-mpi/
  baseline_comparison_gpuexpect/ same sweep, re-run after the GPU-native
                                  expectation fix ("Fix A") + Lightning
                                  ansatz-parity fix landed — this is the
                                  current, citable version; see
                                  `paper_table.md` inside it
  baseline_comparison_legacy_check/  A/B check of `VQE_LEGACY_EXPECT=1`
                                       against the gpuexpect sweep
  blocking_sweep/                 Aer-MPI `blocking_qubits` tuning sweep
                                  (N2), organized as np<N>/aer-mpi/
  profiling/                      Nsight Systems `.nsys-rep` captures
  plots/          aggregate figures generated from the above (hardware-agnostic)
```

Hardware folders may carry a `-YYYY-MM-DD` date suffix (e.g.
`a100-sxm4-40gb-2026-09-07/`) when a second, distinct session on the same
returning hardware class needs to stay separable from the first session's
folder — for example when re-running a sweep after a regression fix, so the
old and new data can both be inspected without one silently overwriting or
merging into the other. Treat a dated sibling folder as a separate,
self-contained session; check its own README/notes (if present) before
assuming it supersedes the undated folder.

## Current hardware folders

- **`rtx-6000-ada-generation/`** — NVIDIA RTX 6000 Ada (48 GB), university IE
  cluster (`haskell`), full P∈{1,2,4,8} strong+weak scaling sweep, April 2026.
  No seed sweep (predates that feature), base 4-molecule benchmark only.
  Also has `slurm/` (raw Slurm job logs) and `trial/` (7-layer diagnostic logs)
  since this ran via `make slurm-trial`/`make slurm-gpu`, not locally.
  This name is exactly what `results_slug()` derives from `nvidia-smi`, so a
  future rerun on the same cluster hardware auto-appends here.

- **`a100-sxm4-40gb/`** — NVIDIA A100-SXM4-40GB, Lambda Cloud. First session
  July 2026: seeds 42-46, adds NH3 + N2 to the benchmark set (CO2 attempted,
  cancelled after running >1hr — no result), only P=2. Confirmed 2026-08-06
  that a fresh Lambda instance reports the identical `nvidia-smi` name
  (`NVIDIA A100-SXM4-40GB` -> `results_slug()` -> `a100-sxm4-40gb`), so this
  folder was renamed from its original hand-picked name
  (`a100-lambda-jul2026-archive`) to match exactly -- new P=1/4/8 strong
  scaling + weak scaling runs auto-append here with no manual merge needed.
  `session-archives/` holds a termination snapshot (exact code + git log)
  from the first Lambda session, kept for provenance.
  If a *future* Lambda rental reports a different A100 variant (SXM4 vs
  PCIe, 40GB vs 80GB), it'll create a new sibling folder instead of landing
  here -- don't assume every A100 rental matches this one.

- **`cpu-only/`** — anything with no GPU: the `serial-baseline/` single-core
  reference (run on Anna's laptop, i7-1065G7 — see `hostname` field in each
  JSON to confirm which machine, since re-running this benchmark on a
  different CPU produces non-comparable wall-clock numbers), plus early
  CPU-only distributed-MPI dev runs and IBM QPU runs from March 2026
  (`distributed-mpi/`, `ibm/`).

- **`gtx1650-reference/`** — placeholder. The raw GTX 1650 result files were
  never committed to this repo and are presumed lost. The only surviving
  numbers are the ones published in the thesis PDF
  (`sections/methodology.tex` §4.4.6, `sections/results.tex` §4.6.4/§4.7) —
  treat those as a fixed reference, not reproducible raw data. This GPU is
  kept out of the published GPU-vs-GPU comparison going forward; it was the
  preliminary/exploratory hardware only.

## How new runs get routed here automatically

`HardwareProfile.results_slug()` (`src/api/hardware.py`) derives the slug
from `nvidia-smi`'s reported GPU name (e.g. `NVIDIA A100-SXM4-40GB` ->
`a100-sxm4-40gb`), or `cpu-only` if no GPU is detected. `template.py` and
`benchmarks/{local_test_run,ibm_test_run}.py` use this to build their output
paths, and `src/api/results.py:save_results()` also stamps `gpu_name`,
`gpu_class`, and `hostname` into every JSON payload — so even if a file gets
copied out of its folder later, it's still self-identifying.

`serial_baseline.py` is the one exception: it never touches the GPU by
design, so it always writes to `results/cpu-only/serial-baseline/`
regardless of what hardware it's run on, but still stamps `hostname` since
different host CPUs aren't comparable to each other either.

When aggregating across runs (`benchmarks/aggregate_scaling.py`,
`benchmarks/aggregate_seeds.py`), pass `--hw <slug>` to target one hardware
folder — both scripts refuse to silently average/compare wall-clock times
across different hardware slugs.
