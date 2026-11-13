# N2 controlled rerun — 2026-11-13 — RESOLVED (environmental, not code)

## What this answers

Experiment 2's N2 result got ~3x *slower* between the 2026-09-01 and 2026-11-03
`hpchybrid` runs (wall_seconds 25.92s -> 79.30s; both P=1, GPU-native path,
seed=42, 50 iters), while every other molecule in that comparison got *faster*
after the Nov 3 fixes landed. An earlier session traced all ~13-16 intervening
commits to `interface.py`/`baseline_comparison.py` individually and ruled each
one out as touching N2's P=1/GPU-native code path, concluding this was "very
likely environmental/run-to-run variance... but NOT yet resolved to a specific
root cause," with a controlled rerun explicitly deferred to later.

This is that rerun.

## Method

Lambda A100-SXM4-40GB instance (`132.145.128.148`), fresh. Two git worktrees
checked out side by side: `cfa4473` (the exact commit backing the 2026-09-01
result file) and `2d66360` (the exact commit backing the 2026-11-03 result
file — the sparse-matvec fix commit itself). Each built as its own Docker
image **on this same instance, same session**, then ran the identical
invocation used for the original runs:

```
python3 -m benchmarks.baseline_comparison --backend hpchybrid --molecule N2 --max-iters 50
```

**Known limitation, accepted deliberately before running:** `cfa4473`'s
Dockerfile has fully unpinned pip installs (no version pins at all — the
pins were added later, in the same work that produced `2d66360`). No
pip-freeze record exists anywhere for the actual original 2026-09-01 run, so
its real historical dependency versions are unrecoverable. Building `cfa4473`
today necessarily resolves today's latest available package versions, not
September's. This was a known, explicit tradeoff (see chat record) — the
rerun tests "does this code, run right now, on this hardware, reproduce the
old slow number," not "does this code with its original exact dependencies."
A `pip freeze` was captured inside both containers this time specifically so
this gap doesn't recur for future investigations.

## Result

| | wall_seconds | wall_per_iter | energy_ha |
|---|---|---|---|
| Sep1 **original** run (Sep 1 instance) | 25.924 | 0.5185 | -110.503723 |
| Nov3 **original** run (Nov 3 instance) | **79.301** | **1.5860** | -110.468157 |
| Sep1 **code**, rerun today (this instance) | 21.910 | 0.4382 | -109.910407 |
| Nov3 **code**, rerun today (this instance) | 22.754 | 0.4551 | -109.822731 |

**Both commits, run back-to-back on the identical instance today, land
within ~4% of each other (21.9s vs 22.8s) — both close to the original Sep 1
time, and neither anywhere near the original Nov 3 run's 79.3s.**

Dependency versions differed between the two legs as expected given the
limitation above (`pip freeze` diff: qiskit 2.6.2 vs 2.5.2, qiskit-aer 0.18.2
vs 0.17.2, pyscf 2.15.0 vs 2.14.0, etc. — Sep1's unpinned build resolved
newer versions than Nov3's pinned ones). Despite that real difference, timing
still converged to the same ~22s ballpark on both legs. This makes the
conclusion below *stronger*, not weaker: even with an actual dependency
difference present, neither leg reproduced anything resembling the 79s
outlier.

Energies differ from the originals and from each other (-109.82 to -110.50
Ha range) as expected for 50-iteration, non-converged SPSA runs with
stochastic perturbations — this is diagnostic/timing data, not a convergence
study, consistent with how both this comparison and Experiment 4's QPU runs
are already framed in the paper.

## Conclusion

**The ~3x N2 slowdown between the original Sep-1 and Nov-3 runs was
environmental/run-to-run variance (different physical instance, GPU
thermal/clock state, host-level contention, or similar) — not a code
regression.** This directly confirms, with controlled evidence rather than
only a process-of-elimination commit trace, what the earlier session's
investigation already pointed toward. The Nov-3 commit (the sparse-matvec
fix) does not, in fact, cause N2's P=1 path to slow down when actually run —
it never touched that code path in the first place, exactly as the commit
trace concluded, and this rerun now shows it empirically rather than only
by elimination.

This also answers the standing question of whether the other 5 molecules'
results need re-verification: no evidence here suggests they do. Both
commits' N2 timings were internally consistent with each other today: the
only outlier was the original Nov 3 *run*, isolated to N2 specifically, with
no sign of the same effect in any of this session's reruns or in any other
molecule's historical data.

## Files

```
results/n2-controlled-rerun-2026-11-13/
  00_manifest.txt              <- commit SHAs, GPU, host, timestamps
  sep1_build.log, nov3_build.log
  sep1_pip_freeze.txt, nov3_pip_freeze.txt   <- full dependency snapshot, both legs
  sep1_run.log, nov3_run.log   <- full console output, both legs
  sep1_result/hpchybrid/N2_20261113_151221.json
  nov3_result/hpchybrid/N2_20261113_152510.json
```

The Lambda instance used for this rerun has not yet been terminated as of
this writing — confirm with the user before assuming it's been killed.
