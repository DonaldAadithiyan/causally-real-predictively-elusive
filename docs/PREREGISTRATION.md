# Pre-registration index

Every camera-ready experiment's pass/fail criteria were written to a `PREREG.md` and committed
**before** the experiment's first run. The commit times below are the evidence; the commit that
adds each PREREG contains no results for that experiment.

## Commit hashes: working repository → public repository

The working repository was put under version control on 2026-11-14, immediately before the
camera-ready experiments (baseline snapshot). Before publication its history was rewritten twice with `git filter-branch`: (1) to replace
absolute home-directory paths in logs and reports by `<repo>` / `<home>`, and (2) to remove
tool-attribution trailers from commit messages. Both rewrites changed every commit hash; commit
times, file contents (apart from the scrubbed paths) and the order of commits are unchanged. The
mapping for the pre-registration commits:

| experiment | PREREG file (public layout) | working-repo hash | public-repo hash | committed (UTC+05:30) |
|---|---|---|---|---|
| baseline snapshot (pre-camera-ready state) | — | `fb7e43f` | `6fb1864` | 2026-11-14 04:49:45 |
| E2 more seeds | `results/camera_ready/E2/PREREG.md` | `aaec82f` | `3242fca` | 2026-11-14 04:52:43 |
| E3 ex-ante baselines + E1 electricity/traffic | `results/camera_ready/E3/PREREG.md`, `E1/PREREG.md` | `cabf8c9` | `927a883` | 2026-11-14 04:56:04 |
| E4 R_j robustness + E5 dose-response / natural range | `results/camera_ready/E4/PREREG.md`, `E5/PREREG.md` | `62f7e35` | `93b3200` | 2026-11-14 05:08:04 |
| E6 TimeMixer | `results/camera_ready/E6/PREREG.md` | `763d853` | `cebc944` | 2026-11-14 05:15:34 |
| F follow-up (regime gating, pooled enrichment, …) | `results/camera_ready/F/PREREG.md` | `f5dbe9a` | `bc9e856` | 2026-11-16 14:00:10 |
| G final round (stopped; see `G/STATUS.md`) | `results/camera_ready/G/PREREG.md` | `7fc583f` | `1293235` | 2026-11-16 14:09:35 |

Full old → new mapping of all 17 pre-publication commits:
`fb7e43f→6fb1864, 89315c7→0ba71d0, aaec82f→3242fca, cabf8c9→927a883, 03d5a5b→c2420e7,
62f7e35→93b3200, c2b9f33→70b3e5b, 763d853→cebc944, 8c8c140→96109e6, 4a75d1f→423b221,
93fbb0c→9bfe65c, 845e3a6→2da9099, f5dbe9a→bc9e856, 67a588b→ad632c8, 7fc583f→1293235,
eaa368a→4e03d42, 2a3a6c7→e944bfe`.

## Criteria that FAILED (reported as such, not re-thresholded)

| ID | criterion | outcome |
|---|---|---|
| C3.1 | ex-ante predictable (one set predictive on ≥ 80% of seeds) | FAIL on all four datasets (ETTh2 best: input statistics 6/10) |
| C3.2 | dissociation (requires C3.1) | FAIL (MVDF not predictive on 30/30) |
| C3.4 | recoverable opportunity (GRR ≥ 0.20 at valid coverage) | FAIL on all four datasets |
| C4.2 | enrichment robust to the residualiser | FAIL (binned mean: 0/20 runs with CI > 1) |
| C5b | natural-range enrichment on the same cells | FAIL on one boundary cell (CI lower bound exactly 1.000) |
| C5c | interchange sign agreement ≥ 70% on ≥ 80% of seeds | FAIL on ETTh1 (7/10), pass on ETTh2 (9/10) |
| C2.1 | every "3/3" claim holds on ≥ 80% of seeds | PARTLY: per-run significance of the scorer-vs-naive gap fails on ETTh1 (4/10, 0/10) and electricity |

Passed: C1.1, C1.2, C3.3, C3.5 (ETTh2 only), C4.1, C4.3, C5a, C2.2, E6 C1.1, F1 (0/20 regime
gating passes, which is the pre-registered negative), and the health rule (0/30 unhealthy seeds).
Full table: `results/camera_ready/CAMERA_READY_SUMMARY.md` §1.

## Changes made after pre-registration (all disclosed)

- **Enrichment-CI order dependence** (bug in an added helper, `experiments/camera_ready/common.py::enrichment_stats`):
  the bootstrap draws depended on the order in which selected features were listed. Fixed during
  the run, before the final tables. This changed the isotonic CI>1 count 4 → 5/20, and C5b from
  PASS to FAIL. The post-fix verdict is the one reported.
- **Spline residualiser** (E4 definition 3): fitted on z-scored P, U because scipy's GCV was
  ill-conditioned at P ~ 1e-5. Conditioning only; λ is still chosen by GCV.
- **H7-A on 10 seeds** (F1): the original `h7a_gate.py` could not be re-run unchanged (its
  cell-B feature IDs come from a different SAE; see `docs/KNOWN_ISSUES.md`); the corrected
  construction was pre-registered in `F/PREREG.md` before computation.

## Pre-registration before the camera-ready

Earlier project stages (v1 MVE, v2 phases 0–5, H7-A, SLCP, baselines; September 2026) also fixed
their stop rules in advance. These rules are recorded in the scripts' docstrings (e.g.
`experiments/v2/h7a_gate.py`, `experiments/v2/slcp_gate.py`) and in `docs/reports/`. That work was
not under version control, so commit timestamps cannot attest to it.
