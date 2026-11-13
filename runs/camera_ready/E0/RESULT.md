# E0 — Housekeeping: Phase 5 report bug + definitions audit

## Provenance
- Git: baseline snapshot `fb7e43f`; fix committed in the E0 commit (see `git log -- experiments/phase5_sae_reproducibility.py`).
- Command: `python -c "import pandas as pd, experiments.phase5_sae_reproducibility as p; p.write_md(pd.read_csv('runs/phase5_sae_repro/AGG_phase5.csv'), 'runs/phase5_sae_repro')"`
- No recomputation. Input: `runs/phase5_sae_repro/AGG_phase5.csv` (unchanged). Data: ETTh1/ETTh2 HTTP mirror (`c6d1709fd0693b1a` / `c022db5521d23684`). Seeds: forecaster seed 0, SAE seeds 10/20/30. CPU. Wall-clock < 1 s.

## E0.1 — Fix
`write_md` hard-coded "0.5 SD" in **five** places (lines 290, 297, 320, 328, 336), not two. All now read `agg["headline_threshold"]`. The sentence at line 290 also justified 0.5 SD as "the loosest still-meaningful cutoff — a stricter one only lowers the match rate"; that is false for 0.25 and was removed.

`PHASE5_RESULT.diff` (old → new) changes only threshold text: 3 hunks (methods paragraph, headline heading, reading paragraph). No number changes. Old file kept as `PHASE5_RESULT.before_fix.md`.

### Numbers (threshold actually used: **0.25 SD**, pooled-SD units in (P_j, U_j, R_j))

| dataset | comparison | MVDF match | pool base rate | "shuffle floor" | match @0.1 | @0.5 | @1.0 |
|---|---|---|---|---|---|---|---|
| ETTh1 | run0 vs run1 | 0.250 | 0.425 | 0.250 | 0.167 | 0.250 | 0.250 |
| ETTh1 | run0 vs run2 | 0.333 | 0.350 | 0.333 | 0.083 | 0.333 | 0.333 |
| ETTh2 | run0 vs run1 | 0.333 | 0.283 | 0.333 | 0.000 | 0.333 | 0.500 |
| ETTh2 | run0 vs run2 | 0.083 | 0.317 | 0.083 | 0.083 | 0.333 | 0.417 |

- Per-comparison: MVDF match **0.08–0.33 (mean 0.25)**, pool base rate **0.28–0.43**. This is the "last check" and it is correct.
- The paper's "0.21–0.29 vs base 0.30–0.39" are **per-dataset means of the two comparisons** (ETTh1 0.29 vs 0.39; ETTh2 0.21 vs 0.30), also at 0.25 SD. Both are arithmetically right; the per-comparison range is the honest one to quote because it shows the spread.
- MVDF match ≤ pool base rate in 3/4 comparisons.

### Anomaly: the "signature-shuffle floor" is not a floor
`signature_shuffle_rate == mvdf_match_rate` in all 4 rows. Cause (`phase5_sae_reproducibility.py:148-151`): the null permutes the **rows** of run-r's signature matrix, but mutual nearest neighbours between two point clouds do not depend on row order, so the "shuffled" matching is the same matching. The quantity carries no information. **The paper must not cite it as a floor.** The pool base rate is the only valid reference. Not fixed (a meaningful null needs a new design).

## E0.2 — Definitions
See `DEFINITIONS.md`. Six code-vs-text discrepancies are listed in its §6. Stride = 1 (E3 lag logic uses window index = origin hour).

## For the paper
| | MVDF cross-seed match | pool base rate |
|---|---|---|
| ETTh1 (2 comparisons) | 0.25, 0.33 | 0.43, 0.35 |
| ETTh2 (2 comparisons) | 0.33, 0.08 | 0.28, 0.32 |

> "At a 0.25-SD signature threshold, 8–33% of MVDF features (mean 25%) have a mutual-nearest-neighbour counterpart in an independently seeded SAE, versus 28–43% for arbitrary pool features."
