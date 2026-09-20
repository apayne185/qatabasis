"""Aggregate multi-seed VQE result JSONs into median ± IQR statistics.

Scans results/simulator/ for JSON files that have a "seed" field, groups by
molecule, and reports median/min/max across seeds for energy + wall time.

Run:
    python benchmarks/aggregate_seeds.py
    python benchmarks/aggregate_seeds.py --backend simulator --since 2026-06-23
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
from glob import glob

# The buggy GPU-native-expectation-under-MPI code path only exists between
# these two commits -- it did not exist before 411486b (so anything earlier,
# e.g. the July 2026 seed sweep, never had the bug to trigger) and was fixed
# by 90008c9. Any run with mpi_ranks >= 2 timestamped strictly inside this
# window used the buggy path (measured 3.6x slower per-iteration on H2O),
# producing misleading scaling/timing numbers -- not an energy correctness
# bug by itself, but it silently mixes a broken configuration into a table
# of otherwise-fixed-path data if not excluded.
#   411486b "interface: GPU-native Pauli expectation via Aer
#            save_expectation_value" -- 2026-09-01 13:04:45+02:00 (bug introduced)
#   90008c9 "interface: MPI-safe default for GPU-native expectation
#            (fixes MPI regression)" -- 2026-09-04 12:44:33+02:00 (bug fixed)
# Compared as digits-only strings against each JSON's own timestamp field
# (handles both "2026-09-04T10:03:23" and "20260904_100323" shapes), since
# git_commit is "unknown" in every JSON produced before that field was made
# reliable -- timestamp is the only signal actually available. See
# docs/RESULT_PROVENANCE.md.
_MPI_REGRESSION_BUG_INTRODUCED = "20260901130445"  # digits only
_MPI_REGRESSION_BUG_FIXED = "20260904124433"       # digits only

# 2026-09-05: commit eabcaef (PR #37) reset _best_physical_energy per
# molecule. Before this, a multi-molecule process leaked the previous
# converged molecule's cached energy into any later molecule whose SPSA
# trajectory dropped below FCI (NH3 inheriting H2O's value being the
# concrete, repeated case). Rather than hardcode a fix date (multi-molecule
# runs from well before this date can still be affected if they were never
# re-run), detect the actual signature directly: two different molecules
# in the same run reporting bit-identical energy is not physically
# plausible and is the exact fingerprint of this bug.


def _is_mpi_regression_window(d: dict) -> bool:
    ts = d.get("timestamp", "")
    ranks = d.get("mpi_ranks")
    if ranks is None or ranks < 2:
        return False
    digits = "".join(c for c in ts if c.isdigit())
    if len(digits) < 14:
        return False
    key = digits[:14]
    return _MPI_REGRESSION_BUG_INTRODUCED <= key < _MPI_REGRESSION_BUG_FIXED


def _has_cross_molecule_contamination(d: dict) -> str | None:
    """Return an offending (mol_a, mol_b) description if two molecules in
    this run report an identical energy -- the _best_physical_energy leak
    signature -- else None."""
    mols = d.get("molecules", {})
    seen: dict[float, str] = {}
    for mol, data in mols.items():
        e = data.get("energy")
        if e is None:
            continue
        if e in seen and seen[e] != mol:
            return f"{mol} energy ({e}) is identical to {seen[e]}'s"
        seen[e] = mol
    return None


def load_seeded_results(backend: str, since: str | None,
                        ranks: int | None, hw: str | None = None) -> list[dict]:
    """Load seeded JSON result files; optionally filter by mpi_ranks.

    Deduplicates by (seed, mpi_ranks), keeping the most recent — guards against
    accidental contamination when scaling sweeps reuse SEED=42 default and
    produce JSONs at multiple P values.

    Also rejects two known-bad file signatures rather than silently
    including them -- see docs/RESULT_PROVENANCE.md for the incidents that
    motivated this:
      1. Runs at mpi_ranks>=2 timestamped inside the MPI-regression window
         (before commit 90008c9 landed).
      2. Runs where two molecules report a bit-identical energy (the
         _best_physical_energy cross-molecule leak signature).
    """
    pattern = os.path.join("results", hw or "*", backend, f"{backend}_*.json")
    files = sorted(glob(pattern))
    candidates = []
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            print(f"[skip] {path}: invalid JSON", file=sys.stderr)
            continue
        if "seed" not in d:
            continue
        if since and d.get("timestamp", "") < since:
            continue
        if ranks is not None and d.get("mpi_ranks") != ranks:
            continue
        if _is_mpi_regression_window(d):
            print(f"[skip] {path}: mpi_ranks={d.get('mpi_ranks')} run timestamped "
                  f"before the MPI-regression fix (90008c9, 2026-09-04 12:44:33) "
                  f"-- known 3.6x per-iter slowdown under this config, excluded "
                  f"from aggregation. See docs/RESULT_PROVENANCE.md.",
                  file=sys.stderr)
            continue
        contamination = _has_cross_molecule_contamination(d)
        if contamination:
            print(f"[skip] {path}: cross-molecule energy contamination detected "
                  f"({contamination}) -- signature of the pre-eabcaef "
                  f"_best_physical_energy leak bug, excluded from aggregation. "
                  f"See docs/RESULT_PROVENANCE.md.", file=sys.stderr)
            continue
        d["_path"] = path
        d["_hw_slug"] = path.split(os.sep)[1]
        candidates.append(d)

    # Group by (seed, mpi_ranks) and MERGE molecule coverage across every
    # file sharing that key, rather than picking one winning file. A "most
    # complete single file wins" rule silently drops legitimate data: a
    # standalone NH3-only or N2-only probe at seed=42 (run separately because
    # no full n=5 sweep exists for those molecules yet) used to be invisible
    # whenever a bigger H2/LiH/BeH2/H2O seed=42 sweep also existed for the
    # same (seed, ranks) key, even though the two files cover disjoint
    # molecules and both are legitimate. When the SAME molecule appears in
    # more than one file for a key, keep the more recently-timestamped copy
    # (guards against a stale single-molecule rerun silently overriding a
    # newer sweep's number for that molecule, or vice versa).
    grouped: dict[tuple, list[dict]] = {}
    for d in candidates:
        key = (d["seed"], d.get("mpi_ranks"))
        grouped.setdefault(key, []).append(d)

    merged_runs = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda d: d.get("timestamp", ""))
        merged_molecules: dict[str, dict] = {}
        mol_source: dict[str, str] = {}  # for the printed file listing below
        for d in group_sorted:
            for mol, data in d.get("molecules", {}).items():
                merged_molecules[mol] = data
                mol_source[mol] = d["_path"]
        # Base the merged record on the most recent file (for fields like
        # gpu_name/hostname/scaling that aren't per-molecule), then overlay
        # the merged molecule set.
        base = dict(group_sorted[-1])
        base["molecules"] = merged_molecules
        base["_merged_from"] = sorted({d["_path"] for d in group})
        base["_mol_source"] = mol_source
        merged_runs.append(base)
    return merged_runs


def aggregate(runs: list[dict]) -> dict[str, dict]:
    """Group by molecule, collect (seed, energy, wall_time, iters) tuples.

    Supports two JSON shapes:
      - simulator runs: top-level "molecules" dict keyed by name
      - ibm runs: top-level "chemistry" with a "molecule" field naming the species
    """
    by_mol: dict[str, list[dict]] = {}
    for run in runs:
        seed = run["seed"]
        # Simulator shape
        for mol, data in run.get("molecules", {}).items():
            by_mol.setdefault(mol, []).append({
                "seed": seed,
                "energy": data["energy"],
                # unperturbed_energy is None for JSONs predating this field
                # (the SPSA perturbed-average bias fix) -- callers that need
                # an accuracy-grade number should prefer this over "energy"
                # when present, and fall back to "energy" with a caveat
                # otherwise. See docs/API.md's "energy vs unperturbed_energy"
                # note for why the two differ.
                "unperturbed_energy": data.get("unperturbed_energy"),
                "fci": data.get("fci"),
                "wall_time": data.get("wall_time"),
                "iters": data.get("iters"),
            })
        # IBM shape — single chemistry record per run
        chem = run.get("chemistry")
        if chem and isinstance(chem, dict) and "energy" in chem:
            mol = chem.get("molecule", "H2")
            by_mol.setdefault(mol, []).append({
                "seed": seed,
                "energy": chem["energy"],
                "unperturbed_energy": chem.get("unperturbed_energy"),
                "fci": chem.get("fci"),
                "wall_time": chem.get("wall_time"),
                "iters": chem.get("iterations"),
            })
    return by_mol


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) — IQR is min/max for small n."""
    s = sorted(values)
    return (statistics.median(s), s[0], s[-1])


def report(by_mol: dict[str, list[dict]]) -> None:
    print(f"\n{'Molecule':<8} {'n':<3} {'Seeds':<20} "
          f"{'Median E (Ha)':<16} {'E range':<22} "
          f"{'Median |err| (Ha)':<18} "
          f"{'Median T (s)':<14}")
    print("-" * 110)

    best_rows = []  # collected for the best-of-N summary printed after the median table

    for mol, runs in sorted(by_mol.items()):
        seeds = sorted({r["seed"] for r in runs})
        # Prefer unperturbed_energy (a real, separate E(theta) evaluation) for
        # accuracy reporting when present; "energy" is SPSA's own internal
        # perturbed-average quantity and carries a small, non-vanishing O(ck^2)
        # bias -- see docs/API.md's "energy vs unperturbed_energy" note. Falls
        # back to "energy" for older JSONs that predate this field, but flags
        # that fallback explicitly rather than silently mixing the two.
        has_unperturbed = [r.get("unperturbed_energy") is not None for r in runs]
        use_unperturbed = all(has_unperturbed) if runs else False
        mixed_sources = any(has_unperturbed) and not use_unperturbed
        energies = [
            (r["unperturbed_energy"] if use_unperturbed else r["energy"])
            for r in runs
        ]
        fci = next((r["fci"] for r in runs if r["fci"] is not None), None)
        wts = [r["wall_time"] for r in runs if r["wall_time"]]

        med_e, min_e, max_e = median_iqr(energies)
        e_range = f"[{min_e:.4f}, {max_e:.4f}]"

        if fci is not None:
            errs = [abs(e - fci) for e in energies]
            med_err = statistics.median(errs)
            err_str = f"{med_err:.4f}"
            best_idx = min(range(len(runs)), key=lambda i: errs[i])
            best_rows.append({
                "mol": mol, "seed": runs[best_idx]["seed"], "energy": energies[best_idx],
                "fci": fci, "err": errs[best_idx], "n": len(runs),
                "source": "unperturbed" if use_unperturbed else "SPSA-perturbed (no unperturbed_energy in JSON)",
            })
        else:
            err_str = "N/A"

        med_t = statistics.median(wts) if wts else 0.0
        source_flag = "" if use_unperturbed else (
            " [MIXED old+new data]" if mixed_sources else " [SPSA-perturbed]"
        )

        print(f"{mol:<8} {len(runs):<3} {str(seeds):<20} "
              f"{med_e:<16.6f} {e_range:<22} "
              f"{err_str:<18} {med_t:<14.2f}{source_flag}")

    if best_rows:
        print(f"\nBest-of-N (min |error| vs FCI among the seeds above; report this per-molecule "
              f"as \"best of n=<seed count> independent SPSA trajectories\" (see n column above), "
              f"not as typical performance):")
        print(f"{'Molecule':<8} {'Best seed':<10} {'Energy (Ha)':<16} "
              f"{'FCI (Ha)':<14} {'|err| (Ha)':<12} {'Chem. acc.?':<12} {'Source'}")
        print("-" * 76)
        for r in best_rows:
            chem_acc = "YES" if r["err"] < 1.6e-3 else ("near" if r["err"] < 0.01 else "no")
            print(f"{r['mol']:<8} {r['seed']:<10} {r['energy']:<16.6f} "
                  f"{r['fci']:<14.4f} {r['err']:<12.4f} {chem_acc:<12} {r['source']}")

    print("\nNotes:")
    print(" - Median is taken across SPSA random seeds at fixed hyperparameters.")
    print(" - E range is [min, max] across seeds; reportable as median ± half-range.")
    print(" - For paper, this corresponds to 'n=N independent SPSA trajectories'.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="simulator",
                   help="results subdir: simulator | ibm | baseline (default: simulator)")
    p.add_argument("--since", default=None,
                   help="ISO timestamp prefix; ignore runs older than this (e.g. 2026-06-23)")
    p.add_argument("--ranks", type=int, default=2,
                   help="filter by mpi_ranks; default 2 (canonical config). "
                        "Use 0 to include all (e.g. for ibm backend).")
    p.add_argument("--hw", default=None,
                   help="restrict to one results/<hw-slug>/ folder (e.g. a100-sxm4-40gb). "
                        "Required if runs from more than one hardware slug are found "
                        "(wall-clock medians across GPUs are meaningless).")
    args = p.parse_args()

    ranks_filter = args.ranks if args.ranks > 0 else None
    runs = load_seeded_results(args.backend, args.since, ranks_filter, args.hw)
    if not runs:
        print(f"No seeded {args.backend} runs found "
              f"(looking for JSON files in results/*/{args.backend}/ with 'seed' field).")
        if args.since:
            print(f"Filter: timestamp >= {args.since}")
        if ranks_filter:
            print(f"Filter: mpi_ranks == {ranks_filter}")
        sys.exit(1)

    hw_slugs = {r["_hw_slug"] for r in runs}
    if len(hw_slugs) > 1:
        print(f"ERROR: runs span multiple hardware folders {sorted(hw_slugs)} -- "
              f"wall-clock medians mixing GPUs are meaningless. Re-run with --hw <slug>.")
        sys.exit(1)

    rank_label = f"P={ranks_filter}" if ranks_filter else "any P"
    print(f"Found {len(runs)} unique (seed, rank) group(s) at {rank_label}, "
          f"hw={hw_slugs.pop() if hw_slugs else 'n/a'} (molecule coverage merged across files sharing a seed):")
    for r in runs:
        sources = r.get("_merged_from", [r.get("_path", "?")])
        mols = sorted(r.get("molecules", {}).keys())
        if len(sources) == 1:
            print(f"  seed={r['seed']:<4} {r.get('timestamp', '?')[:19]}  {sources[0]}  [{', '.join(mols)}]")
        else:
            print(f"  seed={r['seed']:<4} merged from {len(sources)} files -> [{', '.join(mols)}]")
            for src in sources:
                src_mols = sorted(m for m, p in r.get("_mol_source", {}).items() if p == src)
                print(f"      {src}  [{', '.join(src_mols)}]")

    by_mol = aggregate(runs)
    report(by_mol)


if __name__ == "__main__":
    main()
