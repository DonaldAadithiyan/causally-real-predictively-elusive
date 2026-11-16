# E6 — L1 battery on TimeMixer (stretch)

## Provenance

- Git: `4a75d1f`
- Command: `python -m camera_ready.e6_timemixer --datasets ETTh1 ETTh2 --seeds 0 1 2; CR_ARR_DIR=runs/camera_ready/E6/arrays python -m camera_ready.phase34 ...; CR_ARR_DIR=... python -m camera_ready.e3_exante --sets F0 F1 F1p F2 F6 --out runs/camera_ready/E6; python -m camera_ready.report_e3 --prefix E6`
- Seeds: {'ETTh1': [0, 1, 2], 'ETTh2': [0, 1, 2]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684)
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: regeneration 3 × ~10 min per dataset (CPU, 4 threads, contended); analysis ~25 min
- Generated: 2026-11-14 06:40:06 +0530

## Pre-registered criteria (PREREG.md)
- C1.1 (TimeMixer): best held-out R² of F0 has CI including 0 or below 0 on ≥ 2/3 seeds per dataset, both batteries.
- C3.1–C3.3 as in E3, restricted to F0, F1, F1p, F2, F6 (3/3 seeds rule). C3.3 needs F7, which is not in the E6 subset, so it is not evaluated here.

## Reproduction gate
All 6 TimeMixer cells were regenerated from scratch and reproduce the stored run **exactly**: forecaster val MSE and SAE FVU identical to `runs/arch_gen_timemixer/AGG_arch_gen.csv` (|diff| < 1e-15), and `select_mvdf_v2` returns the stored MVDF set on 6/6. Phase-3 best R² recomputed from the saved arrays matches the stored TimeMixer values within 0.0005 on 5 cells and 0.0073 on ETTh2 s2 (`REPRO_phase3_vs_archgen.csv`; float32 storage feeding GBM/MLP).

## C1.1 — Phase 3 on TimeMixer (original definitions; block-bootstrap CI, block 96)

| dataset | seed | marginal best R² [CI] | active best R² [CI] |
|---|---|---|---|
| ETTh1 | 0 | −0.519 [−1.000, −0.149] | −0.608 [−0.965, −0.246] |
| ETTh1 | 1 | −0.520 [−0.988, −0.131] | −0.534 [−0.986, −0.083] |
| ETTh1 | 2 | −0.494 [−0.935, −0.126] | −0.712 [−0.865, −0.367] |
| ETTh2 | 0 | −1.165 [−2.521, −0.507] | −1.191 [−2.614, −0.524] |
| ETTh2 | 1 | −1.036 [−2.403, −0.371] | −1.056 [−2.453, −0.392] |
| ETTh2 | 2 | −1.152 [−2.365, −0.461] | −1.132 [−2.305, −0.470] |

**C1.1 (TimeMixer): PASS** — every CI lies entirely below 0 (3/3 per dataset, both batteries). Unchanged with block 192 (upper bounds −0.08 to −0.66).

Oracle gap on TimeMixer (Phase 4 construction): ID 25.8–26.6% (ETTh1), 29.6–30.0% (ETTh2); shift 24.6–25.6% / 23.6–24.2%; every block-bootstrap CI lower bound ≥ 14.5%.

## C3.1–C3.2 — ex-ante sets on TimeMixer
### CV-selected estimator, E_eval_ID: R² (vs fit mean) [block CI] / Spearman ρ [CI]; ✓ = predictive (C3.1 rule)

**ETTh1** (n_fit = 1179, n_eval = 787, N = 3 seeds)

| set | s0 | s1 | s2 | median R² [min,max] | pooled 2-level CI | k/N predictive |
| --- | --- | --- | --- | --- | --- | --- |
| F0 MVDF | +0.004 [-0.01,+0.01] / +0.07 [-0.13,+0.19] · (ridge) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.004 [-0.00,+0.00] / +0.01 [-0.03,+0.17] · (ridge) | +0.000 [-0.004,+0.004] | [-0.003,+0.004] | 0/3 |
| F1 lagged S (full) | +0.033 [-0.21,+0.41] / -0.28 [-0.59,+0.22] · (iso_col) | -0.024 [-0.24,+0.39] / -0.25 [-0.57,+0.30] · (iso_col) | -0.041 [-0.24,+0.35] / -0.25 [-0.58,+0.30] · (iso_col) | -0.024 [-0.041,+0.033] | [-0.144,+0.202] | 0/3 |
| F1p lagged S (partial) | +0.017 [-0.01,+0.03] / -0.20 [-0.45,-0.01] · (iso_col) | +0.011 [-0.00,+0.02] / -0.16 [-0.44,+0.03] · (iso_col) | +0.015 [-0.01,+0.03] / -0.22 [-0.47,-0.01] · (iso_col) | +0.015 [+0.011,+0.017] | [+0.000,+0.020] | 0/3 |
| F2 input stats | +0.278 [-0.27,+0.37] / -0.04 [-0.51,+0.20] · (gbm) | +0.204 [-0.20,+0.30] / -0.03 [-0.50,+0.21] · (gbm) | +0.272 [-0.30,+0.38] / -0.08 [-0.53,+0.12] · (gbm) | +0.272 [+0.204,+0.278] | [-0.038,+0.305] | 0/3 |
| F6 all ex-ante | +0.033 [-0.21,+0.41] / -0.28 [-0.59,+0.22] · (iso_col) | -0.024 [-0.24,+0.39] / -0.25 [-0.57,+0.30] · (iso_col) | +0.175 [+0.15,+0.31] / +0.06 [-0.24,+0.37] · (gbm) | +0.033 [-0.024,+0.175] | [-0.095,+0.269] | 0/3 |

**ETTh2** (n_fit = 1179, n_eval = 787, N = 3 seeds)

| set | s0 | s1 | s2 | median R² [min,max] | pooled 2-level CI | k/N predictive |
| --- | --- | --- | --- | --- | --- | --- |
| F0 MVDF | -0.019 [-0.05,+0.01] / -0.01 [-0.22,+0.27] · (ridge) | +0.003 [-0.00,+0.02] / +0.05 [-0.03,+0.17] · (iso_col) | -0.001 [-0.05,+0.05] / +0.15 [+0.05,+0.27] · (rf) | -0.001 [-0.019,+0.003] | [-0.026,+0.018] | 0/3 |
| F1 lagged S (full) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.001 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.000,+0.001] | [+0.000,+0.001] | 0/3 |
| F1p lagged S (partial) | +0.080 [+0.02,+0.13] / +0.07 [-0.06,+0.16] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.096 [-0.20,-0.05] / +0.30 [+0.11,+0.43] · (ridge) | -0.000 [-0.096,+0.080] | [-0.105,+0.068] | 0/3 |
| F2 input stats | +0.057 [-0.06,+0.14] / +0.32 [-0.18,+0.63] · (ridge) | +0.066 [-0.05,+0.14] / +0.32 [-0.16,+0.63] · (ridge) | +0.053 [-0.07,+0.13] / +0.31 [-0.19,+0.62] · (ridge) | +0.057 [+0.053,+0.066] | [-0.035,+0.096] | 0/3 |
| F6 all ex-ante | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.001 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.000,+0.001] | [+0.000,+0.001] | 0/3 |


- **C3.1: FAIL on both datasets** — no ex-ante set is predictive (R² CI > 0 and ρ CI > 0) on any seed (0/3 for F1, F1p, F2, F6). Same at block 192.
- **C3.2: FAIL** (requires C3.1). F0 is not predictive on 3/3 seeds on both datasets.
- Near-miss, reported as is: on ETTh2, F2 (input-window statistics) has ρ = +0.31 to +0.32 on all three seeds and R² +0.05 to +0.07, but every CI includes 0 (ρ CI ≈ [−0.18, +0.63]); E_eval of 787 windows is ≈ 8 blocks of 96, so the block bootstrap is wide.
- F6 / F1 on ETTh2 return exactly 0: CV selects the existing isotonic rule, which is increasing-only and collapses to the fit mean when the most-correlated column is negatively correlated. CV choosing a constant means no estimator beat the mean in time-series CV inside the fit split.

## Plain-English summary
The forecaster-architecture check holds for the main null: on TimeMixer, the natural activations of the causally selected MVDF features predict difficulty no better than a constant (best held-out R² −0.49 to −1.16, every CI below 0), and an oracle could still narrow intervals by 24–30%. Cheap forecast-time information does not clear the pre-registered bar on TimeMixer either; on ETTh2 the input-window statistics show a consistent but not significant rank signal (ρ ≈ 0.32).

## Anomalies
- None in reproduction. The `REPRO_phase34.csv` written by `phase34` compares against the iTransformer reference and was deleted as not applicable; the TimeMixer comparison is in `REPRO_phase3_vs_archgen.csv`.

## For the paper

| TimeMixer | MVDF best R² (marg.) | MVDF best R² (active) | oracle gap ID | ex-ante predictive seeds |
|---|---|---|---|---|
| ETTh1 (3 seeds) | −0.52 to −0.49, all CI < 0 | −0.71 to −0.53, all CI < 0 | 25.8–26.6% | 0/3 |
| ETTh2 (3 seeds) | −1.16 to −1.04, all CI < 0 | −1.19 to −1.06, all CI < 0 | 29.6–30.0% | 0/3 |

> "On TimeMixer, natural MVDF activation again fails to predict difficulty (best held-out R² −0.49 to −1.16 across six runs, every block-bootstrap CI below zero), while the oracle gap (24–30%) is the same size as for iTransformer."
