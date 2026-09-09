# Finding and Verifying the Correct Result

An agent (or a person) trying to cite a number from this repo's `results/`
tree has to answer three separate questions before trusting any file:
**does it exist, is it on the branch you're looking from, and is it still
valid (not superseded by a later bug fix)?** All three have failed at
least once in this repo's history — this doc exists because of concrete
incidents, not hypothetical caution. Follow it in order; do not skip to
"just read the JSON."

**Standing rule: this stack has changed considerably since last semester's
work (March–June 2026 era) — the ansatz-construction path, the GPU
expectation path, and the per-molecule state handling have all had
correctness fixes land since then (see Step 3's table). Prefer the most
recent, fixed-code-path result over an older one whenever both exist for
the same claim, even if an older file is what a paper draft currently
cites — "already cited" is not the same as "still correct," and the
default assumption for any file dated before Sept 2026 should be that it
predates at least one of the fixes in Step 3 until checked otherwise.**

---

## Step 1 — Confirm the file exists where you're looking

```bash
git branch --show-current                      # know which branch you're on
find results -iname "*<molecule>*<backend>*"    # candidate files
```

If nothing turns up, **do not assume the data was never produced.** Check
other branches before concluding a gap exists:

```bash
git log --all --oneline -- "results/**/*<pattern>*"
git branch -a --contains <commit-that-touched-it>
```

**Incident this guards against**: the reconstructed BeH2/H2O/NH3
serial-baseline JSON existed and was verified-correct, but sat on
`feature/gpu-expectation-fix` (commit `05b809b`) for days while `main` only
had H2/LiH. An agent reading only `main` reported "BeH2/H2O are genuinely
absent" — wrong; they were one `git cherry-pick` away. Always check
`git log --all`, not just the current branch's history, before writing
"this data doesn't exist" into a paper or a report.

## Step 2 — Check for a same-directory invalidation marker

Before trusting any file, `ls` its directory and its parent for:

```bash
find <the-directory-and-its-parents> -iname "README*" -iname "*INVALID*"
```

Known example: `results/baseline_comparison_gpuexpect/lightning/README-INVALIDATED.md`
marks every JSON in that one subdirectory as invalid (ansatz-parity bug —
Lightning ran a ~25% smaller problem than the side it was being compared
against) while the sibling `hpchybrid/` and `aer-mpi/` directories in the
same sweep remain valid. **Invalidation is scoped to the directory it's
written in — do not assume a whole sweep is bad because one backend's
subfolder is, and do not assume a backend is fine elsewhere just because
one sweep's copy of it was.**

If you are the one discovering a fairness/correctness bug in a committed
result, leave a `README-INVALIDATED.md` in the exact directory affected,
stating: what was wrong, which numbers it changed, which fix (commit hash)
resolves it, and what supersedes it. This is the established convention —
follow it rather than editing/deleting the bad JSON in place.

## Step 3 — Check whether a fix postdates the file (silent staleness)

A file having no invalidation marker does **not** mean it's safe — markers
only exist where someone already caught the problem. Cross-check the file's
timestamp against known correctness-affecting fixes:

| Fix (commit) | Landed | Any run BEFORE this is suspect for... |
|---|---|---|
| `_best_physical_energy` per-molecule reset (PR #37, `eabcaef`) | 2026-09-05 | Any multi-molecule run reporting a below-FCI molecule (e.g. NH3) — it silently inherits the previous molecule's cached energy. Check: does the reported energy for that molecule appear earlier in the same log/JSON for a *different* molecule? If yes, it's contaminated. |
| MPI-safe GPU-expectation default (`90008c9`) | 2026-09-04 12:44:33 | Any NP≥2 GPU run started before this exact timestamp — 3.6x per-iter slowdown misread as "no scaling benefit" rather than a since-fixed bug. Check the run's own start timestamp against the commit timestamp, not just the date. |
| Lightning ansatz-parity fix (2026-09-03) | 2026-09-03 | Any `benchmarks/baseline_comparison.py --backend lightning` output — see Step 2's marker for the concrete case, but the general rule is: a fix landing doesn't retroactively fix already-written JSON. |
| Serial-baseline path-parity fix (`dcb8994`) | pre-Sept 2026 | Any serial baseline JSON from before this commit used a separately-reimplemented ansatz that could silently diverge in `n_params` from the distributed path (confirmed divergence: H2 16 vs 24 params, LiH 72 vs 96). Check `n_params` in the JSON against `MOLECULE_REGISTRY` + `build_ansatz()`'s current logic, not just that the file exists. |

**The general check, not just this table**: `git log --oneline -- <the source file that produced this result>` and look for fix commits dated *after* the result's own timestamp. A result is only as trustworthy as "no relevant fix landed after it was produced" — verify this explicitly, don't assume recency of the result implies correctness.

## Step 4 — Confirm the numbers are self-consistent, not just present

A file existing, unmarked, and post-dating all known fixes can *still* be
wrong. Cheap sanity checks that have caught real bugs in this repo:

- **`n_params` matches the current ansatz construction.** Cross-check
  against `src/api/problems.py`'s `MOLECULE_REGISTRY` + `build_ansatz()`
  for that molecule's `reps`/tier — don't trust a hardcoded number in a
  `.tex` file or an old doc.
- **Provenance fields are present and sane**: `gpu_name`, `gpu_class`,
  `hostname`, `git_commit` (added by `src/api/results.py:save_results()`).
  If a JSON predates this, it has none of these — treat it as historical
  and don't compare its wall-clock numbers against hardware-stamped data.
- **For anything from a multi-molecule run**: does this molecule's number
  match a *different* molecule's number earlier in the same file/log? That
  exact pattern is the signature of the `_best_physical_energy` leak (Step
  3's first row) — grep the log for the suspicious value string across all
  molecules in that run before trusting it.
- **For a reconstructed-from-log JSON** (filename usually has
  `_reconstructed` or a note field saying so): open the underlying `.log`
  and grep for the exact line the number came from — don't trust a
  regex-reconstruction's output without re-deriving it from the raw source
  at least once. See `results/cpu-only/serial-baseline/serial_baseline_20260903_225112_reconstructed.json`'s `_note` field for the pattern this repo uses to flag a reconstructed (vs directly-written) result.

## Step 5 — When multiple candidate files exist for the same claim, pick by this order

1. A file in the **hardware-slug-first canonical location**
   (`results/<hardware-slug>/<category>/...`) on **`main`**, with no
   invalidation marker, postdating all relevant fixes (Steps 2–3) — this
   is authoritative.
2. A file in a **dated sibling folder** (`results/<hardware-slug>-YYYY-MM-DD/`)
   — a separate, self-contained session; check its own `README-YYYY-MM-DD.md`
   before assuming it supersedes or is superseded by the undated folder.
   Per `results/README.md`, these exist specifically to avoid one session's
   data silently overwriting another's.
3. A file that only exists on an **unmerged feature branch** — usable if
   verified (Step 4), but flag explicitly that it needs to land on `main`
   before anyone else relying on `main` can find it. Don't silently cite
   branch-only data as if it were on `main`.
4. **Never** prefer a file merely because it's newer by timestamp alone —
   newer does not mean correct (see the Sept 4 MPI-regression sweep, which
   is newer than and worse than the July 27 data it was meant to replace).

---

## Worked example: "what's the LiH strong-scaling P=2 wall-clock?"

1. `find results -iname "scaling_P2*"` → multiple hits across
   `results/a100-sxm4-40gb/scaling/`, `results/a100-sxm4-40gb-2026-09-07/.../scaling/`,
   `results/rtx-6000-ada-generation/scaling/`.
2. Hardware matters first — A100 is the paper's primary hardware
   (see `README.md`), so start there, not RTX 6000.
3. Two A100 candidates: the canonical `results/a100-sxm4-40gb/scaling/scaling_P2.txt`
   and the dated `results/a100-sxm4-40gb-2026-09-07/.../scaling/scaling_P2.txt`.
   These are **different sweeps, not duplicates** — the undated one is the
   restored July 27 baseline (`git log --follow` shows commit `52a96d2`
   explicitly restored it after an accidental overwrite); the dated one is
   the Sept 7 legacy-path re-sweep. Check which one the paper section you're
   writing actually needs (the July 27 baseline for Experiment 2's original
   table, or the Sept 7 legacy sweep for its NH3-inclusive successor) —
   they are not interchangeable and citing the wrong one silently swaps
   which experiment's methodology backs the number.
4. Read the value directly from the file — do not average or merge across
   these two candidates.

---

---

## Experiment 1 (Distributed Statevector VQE vs. serial baseline) — current answer, worked out

This is the specific case that motivated this document, worked all the way
through so it doesn't need re-deriving. Experiment 1 needs, per molecule:
a serial-baseline number and a distributed number, both from the
**path-parity-fixed** code (commit `dcb8994`+, which routes both paths
through the same `MoleculeResolver` → `ChemistryProblem` → `prepare()` —
see `docs/API.md`).

**Serial baseline — use these, not the March files:**

| Molecule | File | `n_params` | Status |
|---|---|---:|---|
| H2 | `results/cpu-only/serial-baseline/serial_baseline_20260903_134751.json` | 24 | Path-parity fixed, verified against distributed path |
| LiH | same file | 96 | Same run, same file |
| BeH2 | `results/cpu-only/serial-baseline/serial_baseline_20260903_225112_reconstructed.json` | 112 | Reconstructed from log after a mid-run crash (N2 killed it before its own JSON wrote) — verified against the raw log line-for-line before being trusted; see Step 4 above |
| H2O | same reconstructed file | 112 | Same reconstruction |
| NH3 | same reconstructed file | 128 | Same reconstruction |
| N2 | **none — deliberately excluded** | — | Projected single-CPU wall-clock exceeded 10 hours; this is a disclosed design decision, not a gap. Report N2's distributed number alone, with no serial comparison, and say why. |

**Do not use** `results/cpu-only/serial-baseline/serial_baseline_20260319_001432.json`
(and its siblings `20260319_214156.json`, `20260326_163724.json`) for
Experiment 1's numbers — these predate the path-parity fix. Concretely,
this file's H2 energy is `-1.1257578...`; the fixed-path file's H2 energy
is `-1.1348957...` — different code paths, different numbers, and only
the second is consistent with the ansatz the distributed side actually
uses (`n_params=24`, not whatever the pre-fix reimplementation produced).

**Known unresolved issue as of this writing**: the paper's own
`4_results.tex` Table 4 currently cites the March 19 (pre-fix) file, while
`3_experiments.tex`'s prose describes the current (post-fix) code — a
live mismatch between what the text claims about methodology and which
numbers back it. Fixing this is a paper-editing decision (re-run and
retable with the Sept numbers above, vs. keep March numbers with a
disclosure footnote), not something to silently resolve by picking one —
flag it to the user if you're asked to write or check Experiment 1's
results table.

**Distributed side**: `results/cpu-only/distributed-mpi/simulator_20260318_112038.json`
covers H2/LiH/BeH2/H2O and is confirmed consistent (exact match against
`4_results.tex` Table 3 per the 2026-09-07 fact-check audit in
`qatabasis-internal/paper-tracker/paper-repo-NOTES.md`). No known issue
with this file. N2's distributed-only number should come from the A100
sweep (`results/a100-sxm4-40gb/` or the dated `-2026-09-07` sibling,
picked per Step 5's ranking above) — not from this March CPU-only file,
which never ran N2.

---

## If you are an agent given a task that depends on a result file

State explicitly, in your output, which file you used and why (Step 5's
ranking) — not just the number. If Step 3 or Step 4 raised any doubt you
couldn't fully resolve, say so rather than presenting the number as clean.
"I used X because Y, and could not verify Z" is a correct and useful
answer; a silently-wrong number that looks confident is the actual failure
mode this document exists to prevent.
