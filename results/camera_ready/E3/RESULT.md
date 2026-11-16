# E3 — Ex-ante difficulty baselines + gap recovery (ETTh1, ETTh2)

## Provenance

- Git: `93fbb0c` (+uncommitted changes)
- Command: `python -m camera_ready.e3_exante --datasets ETTh1 ETTh2 --seeds <0-9> --out runs/camera_ready/E3; python -m camera_ready.report_e3 --datasets ETTh1 ETTh2 --prefix E3`
- Seeds: {'ETTh1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ETTh2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684)
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: 130 min summed over 20 cells (CPU, 4 torch threads, RF n_jobs 4)
- Generated: 2026-11-14 07:11:40 +0530

## Pre-registered criteria
Verbatim in `PREREG.md` (committed `cabf8c9` before the first run). Seed threshold: k ≥ ceil(0.8 N) = 8/10 on ETTh1 and ETTh2.

## Verdicts (ETTh1, ETTh2; N = 10 seeds each; block 96 primary)

| criterion | ETTh1 | ETTh2 |
|---|---|---|
| **C3.1** ex-ante predictable (one set on ≥ 8/10 seeds) | **FAIL**: best set F6 1/10 | **FAIL**: best set F2 6/10 (F1p 2/10) |
| **C3.2** dissociation (C3.1 and F0 not predictive) | **FAIL** (needs C3.1); F0 not predictive on 10/10 | **FAIL** (needs C3.1); F0 not predictive on 10/10 |
| **C3.3** MVDF adds nothing (ΔR² CI ∋ 0 or < 0) | **PASS** 10/10 (ΔR² median −0.00001, range −0.065 to +0.007) | **PASS** 10/10 (ΔR² = 0.000 on all seeds) |
| **C3.4** recoverable opportunity (GRR ≥ 0.20, valid coverage) | **FAIL** 0/10 | **FAIL** 0/10 |
| **C3.5** representation contains it (F4 or F5) | **FAIL**: F4 0/10, F5 0/10 | **PASS**: F4 **10/10**, F5 7/10 |

Lenient reading of C3.1 (any ex-ante set predictive on a given seed): ETTh1 1/10 seeds, ETTh2 6/10. Still below 8/10.

**Block-192 sensitivity:** no ETTh verdict changes (predictive counts identical except ETTh2 F3 0→1/10). See E1 for the one dataset (electricity) where a verdict does change.

### Key numbers: CV-selected estimator, E_eval_ID (n_fit 1179, n_eval 787 per seed)

| set | ETTh1 R² median [min, max] | ETTh1 ρ median | pred. | ETTh2 R² median [min, max] | ETTh2 ρ median [min, max] | pred. |
|---|---|---|---|---|---|---|
| F0 MVDF | −0.000 [−0.02, +0.01] | −0.01 | 0/10 | +0.002 [−0.03, +0.01] | +0.00 | 0/10 |
| F1 lagged S (full) | −0.037 [−0.28, +0.06] | −0.18 | 0/10 | +0.000 | +0.00 | 0/10 |
| F1p lagged S (partial) | +0.022 [+0.01, +0.03] | −0.01 | 0/10 | +0.000 [−0.05, +0.08] | +0.00 [−0.30, +0.25] | 2/10 |
| F2 input stats | +0.151 [−0.47, +0.23] | +0.00 | 0/10 | +0.082 [+0.07, +0.10] | +0.32 [+0.27, +0.35] | 6/10 |
| F3 forecast stats | +0.001 | −0.04 | 0/10 | +0.012 [−0.00, +0.03] | +0.17 [−0.05, +0.33] | 0/10 |
| F4 hidden PCA-12 | −0.008 | −0.01 | 0/10 | +0.048 [+0.04, +0.06] | +0.38 [+0.32, +0.40] | **10/10** |
| F5 SAE PCA-12 | +0.005 | −0.02 | 0/10 | +0.030 [−0.00, +0.05] | +0.26 [−0.07, +0.43] | 7/10 |
| F6 all ex-ante | +0.167 [+0.00, +0.37] | −0.02 | 1/10 | +0.000 | +0.00 | 0/10 |

### Gap recovery (C3.4)
- The ID test block (last 20% of E_eval_ID, 394 windows ≈ 4 blocks of 96) is unusable for GRR. naive_matched (static, calibrated on the first 80%) covers only 0.62 (ETTh1) / 0.56 (ETTh2) there, so the LOO oracle (≈ 0.90 by construction) is wider than naive and the GRR denominator is negative on 20/20 cells. This is the chronological-calibration-transfer problem documented in round 2, now affecting the static path too.
- On the shift block, naive_matched covers 0.78 (ETTh1) / 0.91 (ETTh2). Wherever the best ex-ante Mondrian band reaches valid coverage it is wider than naive: GRR −0.47 to −2.41. No cell recovers any of the oracle gap.

## Plain-English summary
Using information genuinely available when the forecast is made, nothing predicts realised difficulty on ETTh1: lagged realised errors, partial errors of in-flight windows, input-window statistics, forecast statistics, the dense hidden state and the full SAE code all fail, under six estimator classes, with model choice made by time-series CV inside the fit split. Where an R² is positive it comes with a near-zero or negative rank correlation, i.e. it tracks the mean level of difficulty between blocks, not which windows are hard. ETTh2 is different and is the important nuance: there is a real, small, rank-level signal (ρ ≈ 0.3–0.4, R² ≈ 0.05–0.08). It sits in the **dense hidden state** (10/10 seeds) and the SAE code as a whole (7/10), and partly in input-window statistics (6/10), **but not in the causally selected MVDF features (0/10)**. MVDF adds nothing on top of the ex-ante set on any seed. None of the ex-ante predictors produces a conformal band narrower than the constant band at valid coverage, so the oracle gap stays an upper bound on heterogeneity, none of it recovered.

## Anomalies
- **Constant predictions.** On several cells the CV-selected estimator is the existing isotonic rule (increasing-only, applied to the most-|corr| column), which predicts a constant whenever that correlation is negative. This yields R² ≈ 0 and ρ = 0 exactly (e.g. F6 on ETTh2). Per the pre-registration this is the CV verdict ("nothing beats the mean in CV"), not tuned away. Best-on-eval values are reported separately as descriptive (`_tables_E3.md`) and are higher (e.g. ETTh1 F1 iso-on-OLS up to +0.33). They are optimistic by construction.
- **The R² reference is the fit-split mean,** as the spec states; Phase 3's code uses the eval mean (E0 DEFINITIONS §6). Both are stored (`R2`, `R2_evalmean`).
- **Wide block CIs.** The eval set is 787 windows ≈ 8 blocks of 96. ETTh2 F2's ρ ≈ +0.32 on every seed but its CI includes 0 on 4/10, which is why C3.1 falls short.

## For the paper

| set (CV-selected estimator) | ETTh1 ρ / predictive seeds | ETTh2 ρ / predictive seeds |
|---|---|---|
| MVDF activations (F0) | −0.01 / 0/10 | +0.00 / 0/10 |
| all ex-ante: lagged errors + input + forecast stats (F6) | −0.02 / 1/10 | input stats alone (F2): +0.32 / 6/10 |
| dense hidden state, PCA-12 (F4) | −0.01 / 0/10 | +0.38 / 10/10 |
| full SAE code, PCA-12 (F5) | −0.02 / 0/10 | +0.26 / 7/10 |
| GRR of best ex-ante band (valid coverage) | none ≥ 0 | none ≥ 0 |

> "Information available at forecast time does not predict realised difficulty on ETTh1 (0–1 of 10 seeds for every feature set, six estimators, CV-selected). On ETTh2 a weak rank signal exists (Spearman ρ ≈ 0.38) and is carried by the dense hidden state on 10/10 seeds, but not by the causally validated MVDF features (0/10), which add nothing on top of ex-ante features (ΔR² CI ∋ 0 on 20/20 seeds). No ex-ante Mondrian band narrowed intervals at valid coverage, so the 24–30% oracle gap is an upper bound on heterogeneity, none of which was recovered by any ex-ante predictor we tested."

---

## Full tables (auto-generated)

### CV-selected estimator, E_eval_ID: R² (vs fit mean) [block CI] / Spearman ρ [CI]; ✓ = predictive (C3.1 rule)

**ETTh1** (n_fit = 1179, n_eval = 787, N = 10 seeds)

| set | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median R² [min,max] | pooled 2-level CI | k/N predictive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 MVDF | -0.013 [-0.03,-0.00] / +0.11 [+0.02,+0.22] · (rf) | +0.001 [-0.00,+0.01] / -0.08 [-0.19,+0.01] · (iso_col) | +0.001 [-0.01,+0.02] / -0.07 [-0.30,+0.10] · (ridge) | -0.025 [-0.03,-0.00] / -0.12 [-0.22,+0.06] · (rf) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.001 [-0.01,+0.01] / -0.07 [-0.23,+0.09] · (ridge) | +0.006 [-0.00,+0.02] / -0.02 [-0.12,+0.12] · (rf) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.006 [-0.01,-0.00] / +0.09 [-0.01,+0.17] · (rf) | -0.000 [-0.01,+0.02] / +0.08 [+0.00,+0.22] · (ridge) | -0.000 [-0.025,+0.006] | [-0.008,+0.003] | 0/10 |
| F1 lagged S (full) | +0.021 [-0.16,+0.38] / -0.34 [-0.63,+0.14] · (iso_col) | -0.026 [-0.28,+0.41] / -0.11 [-0.49,+0.48] · (rf) | -0.047 [-0.26,+0.28] / -0.09 [-0.38,+0.38] · (rf) | -0.178 [-0.40,+0.09] / -0.16 [-0.50,+0.29] · (gbm) | -0.064 [-0.28,+0.38] / -0.20 [-0.58,+0.29] · (rf) | +0.060 [-0.09,+0.31] / -0.29 [-0.55,+0.11] · (rf) | -0.090 [-0.28,+0.33] / -0.28 [-0.59,+0.26] · (gbm) | -0.016 [-0.34,+0.31] / -0.08 [-0.45,+0.23] · (rf) | +0.028 [-0.10,+0.27] / -0.12 [-0.41,+0.24] · (rf) | -0.277 [-0.57,+0.24] / -0.21 [-0.46,+0.26] · (rf) | -0.037 [-0.277,+0.060] | [-0.144,+0.081] | 0/10 |
| F1p lagged S (partial) | +0.024 [-0.01,+0.04] / +0.06 [-0.11,+0.13] · (iso_col) | +0.006 [+0.00,+0.01] / -0.18 [-0.43,+0.06] · (iso_col) | +0.022 [-0.02,+0.05] / -0.05 [-0.20,+0.04] · (iso_col) | +0.018 [-0.01,+0.04] / -0.04 [-0.15,+0.05] · (iso_col) | +0.022 [-0.00,+0.05] / +0.04 [-0.09,+0.13] · (iso_col) | +0.022 [+0.00,+0.04] / +0.04 [-0.04,+0.10] · (iso_col) | +0.023 [+0.00,+0.04] / +0.05 [-0.08,+0.14] · (iso_col) | +0.035 [+0.00,+0.05] / +0.03 [-0.09,+0.06] · (iso_col) | +0.015 [+0.00,+0.03] / -0.09 [-0.38,+0.11] · (iso_col) | +0.018 [+0.00,+0.03] / -0.08 [-0.34,+0.10] · (iso_col) | +0.022 [+0.006,+0.035] | [+0.009,+0.024] | 0/10 |
| F2 input stats | -0.472 [-0.86,-0.28] / +0.07 [-0.27,+0.59] · (ridge) | +0.193 [-0.07,+0.30] / +0.10 [-0.31,+0.32] · (gbm) | +0.178 [-0.05,+0.25] / -0.04 [-0.37,+0.11] · (gbm) | +0.163 [+0.04,+0.29] / +0.01 [-0.26,+0.26] · (gbm) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.140 [+0.05,+0.25] / +0.04 [-0.24,+0.23] · (gbm) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.229 [-0.14,+0.31] / -0.05 [-0.47,+0.12] · (gbm) | +0.201 [-0.11,+0.33] / -0.01 [-0.37,+0.24] · (gbm) | +0.151 [-0.472,+0.229] | [-0.113,+0.148] | 0/10 |
| F3 forecast stats | +0.024 [-0.00,+0.03] / -0.04 [-0.44,+0.26] · (iso_col) | +0.002 [+0.00,+0.00] / -0.09 [-0.53,+0.18] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [-0.01,+0.00] / -0.16 [-0.27,-0.01] · (iso_col) | +0.014 [+0.00,+0.02] / -0.05 [-0.45,+0.19] · (iso_col) | +0.010 [-0.00,+0.02] / -0.08 [-0.25,+0.19] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.037 [+0.01,+0.04] / -0.04 [-0.47,+0.28] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.001 [-0.000,+0.037] | [+0.001,+0.011] | 0/10 |
| F4 hidden PCA-12 | +0.002 [+0.00,+0.00] / -0.05 [-0.16,+0.06] · (iso_col) | +0.002 [-0.00,+0.01] / -0.05 [-0.21,+0.25] · (iso_col) | -0.008 [-0.02,+0.03] / +0.02 [-0.27,+0.37] · (ridge) | -0.008 [-0.02,+0.02] / +0.02 [-0.20,+0.32] · (ridge) | -0.007 [-0.02,+0.02] / -0.00 [-0.31,+0.33] · (ridge) | -0.014 [-0.03,+0.02] / -0.02 [-0.41,+0.40] · (ridge) | -0.012 [-0.03,+0.04] / -0.01 [-0.33,+0.39] · (ridge) | -0.011 [-0.02,+0.02] / -0.05 [-0.42,+0.34] · (ridge) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.013 [-0.03,+0.03] / +0.02 [-0.34,+0.42] · (ridge) | -0.008 [-0.014,+0.002] | [-0.010,+0.004] | 0/10 |
| F5 SAE PCA-12 | +0.017 [-0.02,+0.04] / -0.00 [-0.17,+0.09] · (iso_col) | -0.011 [-0.02,+0.02] / -0.04 [-0.30,+0.26] · (ridge) | +0.000 [-0.01,+0.05] / -0.17 [-0.41,+0.12] · (ridge) | +0.003 [-0.01,+0.04] / -0.07 [-0.30,+0.20] · (ridge) | +0.012 [+0.00,+0.04] / +0.02 [-0.07,+0.16] · (iso_col) | +0.001 [-0.00,+0.00] / +0.01 [-0.04,+0.03] · (iso_col) | +0.013 [-0.00,+0.04] / +0.00 [-0.11,+0.09] · (iso_col) | +0.006 [+0.00,+0.01] / +0.04 [-0.13,+0.12] · (iso_col) | -0.011 [-0.03,+0.03] / -0.10 [-0.38,+0.14] · (ridge) | +0.023 [-0.00,+0.05] / -0.18 [-0.29,-0.07] · (iso_col) | +0.005 [-0.011,+0.023] | [-0.002,+0.016] | 0/10 |
| F6 all ex-ante | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.165 [-0.03,+0.46] / -0.16 [-0.40,+0.28] · (gbm) | +0.336 [+0.17,+0.43] / +0.11 [-0.08,+0.29] · (gbm) | +0.195 [+0.08,+0.49] / -0.11 [-0.36,+0.37] · (gbm) | +0.115 [+0.07,+0.32] / +0.31 [+0.09,+0.57] ✓ (rf) | +0.210 [-0.10,+0.46] / -0.15 [-0.45,+0.29] · (gbm) | +0.131 [-0.02,+0.46] / -0.03 [-0.38,+0.56] · (rf) | +0.371 [-0.18,+0.52] / -0.00 [-0.31,+0.28] · (gbm) | +0.168 [+0.05,+0.38] / -0.26 [-0.46,+0.10] · (gbm) | +0.167 [+0.000,+0.371] | [+0.079,+0.260] | 1/10 |
| F7 all + MVDF | +0.150 [+0.02,+0.36] / -0.11 [-0.45,+0.23] · (gbm) | +0.164 [+0.06,+0.41] / -0.12 [-0.35,+0.29] · (gbm) | +0.164 [+0.02,+0.47] / -0.09 [-0.34,+0.40] · (gbm) | +0.327 [+0.20,+0.45] / +0.13 [-0.02,+0.29] · (gbm) | +0.203 [+0.03,+0.47] / -0.15 [-0.41,+0.29] · (gbm) | +0.115 [+0.07,+0.32] / +0.31 [+0.09,+0.57] ✓ (rf) | +0.145 [-0.17,+0.43] / -0.17 [-0.48,+0.30] · (gbm) | +0.131 [-0.02,+0.46] / -0.03 [-0.38,+0.56] · (rf) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.134 [+0.06,+0.37] / -0.20 [-0.40,+0.20] · (gbm) | +0.147 [-0.000,+0.327] | [+0.098,+0.250] | 1/10 |

**ETTh2** (n_fit = 1179, n_eval = 787, N = 10 seeds)

| set | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median R² [min,max] | pooled 2-level CI | k/N predictive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 MVDF | +0.005 [-0.01,+0.03] / +0.00 [-0.08,+0.10] · (rf) | -0.011 [-0.04,+0.03] / +0.06 [-0.10,+0.29] · (iso_col) | +0.003 [-0.04,+0.04] / +0.11 [-0.01,+0.27] · (rf) | +0.012 [-0.01,+0.05] / +0.11 [-0.02,+0.26] · (rf) | -0.033 [-0.05,+0.05] / +0.07 [-0.02,+0.24] · (rf) | +0.003 [-0.02,+0.02] / +0.00 [-0.09,+0.11] · (iso_col) | +0.012 [-0.06,+0.07] / -0.00 [-0.14,+0.18] · (rf) | -0.019 [-0.03,-0.01] / -0.04 [-0.18,+0.13] · (ridge) | +0.001 [-0.01,+0.01] / -0.00 [-0.07,+0.07] · (rf) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.002 [-0.033,+0.012] | [-0.009,+0.017] | 0/10 |
| F1 lagged S (full) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.012 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.010 [+0.01,+0.01] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.011 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [-0.000,+0.012] | [-0.000,+0.007] | 0/10 |
| F1p lagged S (partial) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.077 [+0.01,+0.11] / +0.22 [+0.08,+0.27] ✓ (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.003 [-0.00,+0.00] / -0.30 [-0.57,-0.14] · (iso_col) | +0.065 [+0.01,+0.09] / +0.19 [+0.07,+0.22] ✓ (iso_col) | -0.045 [-0.10,-0.02] / +0.25 [+0.08,+0.40] · (ridge) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.071 [-0.00,+0.09] / +0.11 [-0.03,+0.15] · (iso_col) | +0.000 [-0.045,+0.077] | [-0.010,+0.031] | 2/10 |
| F2 input stats | +0.070 [-0.02,+0.13] / +0.32 [-0.04,+0.60] · (ridge) | +0.092 [+0.02,+0.15] / +0.33 [+0.02,+0.59] ✓ (ridge) | +0.065 [-0.03,+0.13] / +0.27 [-0.11,+0.55] · (ridge) | +0.085 [+0.01,+0.15] / +0.34 [+0.01,+0.60] ✓ (ridge) | +0.077 [+0.00,+0.14] / +0.31 [-0.05,+0.59] · (ridge) | +0.090 [+0.02,+0.15] / +0.34 [+0.04,+0.60] ✓ (ridge) | +0.080 [-0.02,+0.14] / +0.27 [-0.08,+0.54] · (ridge) | +0.082 [+0.00,+0.14] / +0.31 [+0.04,+0.57] ✓ (ridge) | +0.098 [+0.03,+0.15] / +0.35 [+0.08,+0.60] ✓ (ridge) | +0.082 [+0.00,+0.14] / +0.34 [+0.02,+0.59] ✓ (ridge) | +0.082 [+0.065,+0.098] | [+0.044,+0.096] | 6/10 |
| F3 forecast stats | +0.009 [-0.02,+0.02] / +0.27 [+0.04,+0.44] · (ridge) | +0.032 [-0.01,+0.04] / +0.17 [-0.05,+0.28] · (ridge) | -0.000 [-0.04,+0.01] / +0.33 [+0.09,+0.49] · (ridge) | +0.015 [-0.01,+0.02] / +0.16 [-0.07,+0.29] · (ridge) | +0.030 [-0.00,+0.04] / -0.05 [-0.26,+0.02] · (ridge) | +0.034 [-0.02,+0.05] / +0.13 [-0.08,+0.27] · (ridge) | +0.002 [-0.03,+0.01] / +0.24 [+0.01,+0.36] · (ridge) | +0.003 [-0.03,+0.01] / +0.27 [-0.00,+0.42] · (ridge) | +0.015 [-0.01,+0.02] / +0.15 [-0.07,+0.28] · (ridge) | +0.004 [-0.04,+0.03] / +0.18 [-0.05,+0.32] · (ridge) | +0.012 [-0.000,+0.034] | [-0.007,+0.014] | 0/10 |
| F4 hidden PCA-12 | +0.049 [+0.04,+0.11] / +0.40 [+0.29,+0.63] ✓ (ridge) | +0.044 [+0.03,+0.12] / +0.36 [+0.26,+0.61] ✓ (ridge) | +0.041 [+0.02,+0.11] / +0.33 [+0.24,+0.57] ✓ (ridge) | +0.047 [+0.03,+0.12] / +0.40 [+0.26,+0.65] ✓ (ridge) | +0.053 [+0.04,+0.11] / +0.39 [+0.24,+0.63] ✓ (ridge) | +0.059 [+0.04,+0.12] / +0.39 [+0.28,+0.63] ✓ (ridge) | +0.048 [+0.02,+0.11] / +0.32 [+0.21,+0.57] ✓ (ridge) | +0.047 [+0.03,+0.12] / +0.37 [+0.27,+0.63] ✓ (ridge) | +0.054 [+0.04,+0.10] / +0.38 [+0.27,+0.59] ✓ (ridge) | +0.046 [+0.03,+0.10] / +0.38 [+0.27,+0.62] ✓ (ridge) | +0.048 [+0.041,+0.059] | [+0.059,+0.086] | 10/10 |
| F5 SAE PCA-12 | +0.036 [+0.02,+0.08] / +0.30 [+0.13,+0.52] ✓ (ridge) | +0.015 [-0.06,+0.14] / +0.16 [+0.06,+0.40] · (rf) | +0.038 [+0.02,+0.09] / +0.24 [+0.08,+0.49] ✓ (ridge) | +0.026 [+0.00,+0.11] / +0.23 [+0.11,+0.44] ✓ (ridge) | +0.031 [+0.02,+0.08] / +0.29 [+0.16,+0.54] ✓ (ridge) | +0.047 [+0.02,+0.14] / +0.43 [+0.35,+0.67] ✓ (ridge) | +0.029 [+0.00,+0.09] / +0.26 [+0.14,+0.53] ✓ (ridge) | -0.004 [-0.01,+0.00] / -0.07 [-0.13,+0.10] · (iso_col) | +0.019 [-0.00,+0.08] / +0.27 [+0.19,+0.55] · (ridge) | +0.033 [+0.02,+0.09] / +0.30 [+0.22,+0.55] ✓ (ridge) | +0.030 [-0.004,+0.047] | [+0.028,+0.068] | 7/10 |
| F6 all ex-ante | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.012 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.010 [+0.01,+0.01] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.011 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [-0.000,+0.012] | [-0.000,+0.007] | 0/10 |
| F7 all + MVDF | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.012 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.010 [+0.01,+0.01] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [+0.00,+0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.011 [+0.01,+0.02] / +0.00 [+0.00,+0.00] · (iso_col) | -0.000 [-0.00,-0.00] / +0.00 [+0.00,+0.00] · (iso_col) | +0.000 [-0.000,+0.012] | [-0.000,+0.007] | 0/10 |

### Descriptive only: best-on-eval R² per set (max over the 7 estimators; selection on eval is optimistic), median over seeds [min, max]

| dataset | split | F0 | F1 | F1p | F2 | F3 | F4 | F5 | F6 | F7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | ID | +0.00 [-0.01,+0.02] | +0.23 [+0.07,+0.30] | +0.02 [+0.01,+0.03] | +0.20 [+0.14,+0.23] | +0.08 [+0.03,+0.15] | +0.06 [-0.01,+0.08] | +0.02 [-0.00,+0.05] | +0.25 [+0.16,+0.37] | +0.24 [+0.13,+0.37] |
| ETTh1 | shift | +0.00 [-0.00,+0.01] | -0.03 [-0.12,+0.15] | +0.03 [+0.01,+0.04] | +0.06 [+0.02,+0.16] | +0.03 [+0.00,+0.09] | +0.05 [+0.01,+0.09] | +0.01 [-0.01,+0.04] | +0.01 [-0.03,+0.15] | +0.00 [-0.02,+0.15] |
| ETTh2 | ID | +0.01 [-0.01,+0.03] | +0.24 [-0.00,+0.50] | +0.00 [-0.00,+0.08] | +0.25 [+0.22,+0.27] | +0.07 [+0.06,+0.10] | +0.07 [+0.05,+0.13] | +0.04 [+0.03,+0.07] | +0.26 [-0.00,+0.32] | +0.26 [-0.00,+0.33] |
| ETTh2 | shift | -0.00 [-0.02,+0.01] | +0.00 [-0.00,+0.01] | +0.00 [-0.00,+0.11] | +0.04 [+0.03,+0.05] | +0.05 [+0.03,+0.07] | -0.05 [-0.08,-0.00] | -0.00 [-0.10,+0.02] | +0.00 [-0.00,+0.01] | +0.00 [-0.00,+0.01] |

### Full estimator × feature-set grid, E_eval_ID, median R² over seeds


**ETTh1**

| feature_set | gbm | iso_col | iso_ols | mlp | ols | rf | ridge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | -0.005 | -0.002 | -0.004 | -0.096 | -0.004 | -0.005 | -0.001 |
| F1 | -0.089 | 0.055 | 0.208 | -0.584 | -0.091 | -0.015 | 0.072 |
| F1p | -0.026 | 0.022 | -0.008 | -0.055 | -0.005 | -0.037 | 0.011 |
| F2 | 0.197 | 0.000 | -0.542 | -0.509 | -1.422 | -0.068 | -0.445 |
| F3 | 0.078 | 0.001 | -0.078 | -0.092 | -0.104 | 0.065 | -0.086 |
| F4 | 0.039 | -0.002 | -0.031 | -0.005 | -0.024 | 0.043 | -0.012 |
| F5 | 0.007 | 0.004 | -0.017 | -0.032 | -0.015 | 0.008 | -0.006 |
| F6 | 0.203 | 0.022 | -0.162 | -0.989 | -0.722 | 0.123 | -0.277 |
| F7 | 0.183 | 0.022 | -0.203 | -0.999 | -0.716 | 0.123 | -0.277 |

**ETTh2**

| feature_set | gbm | iso_col | iso_ols | mlp | ols | rf | ridge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | -0.011 | -0.000 | -0.003 | -0.101 | -0.003 | 0.002 | 0.002 |
| F1 | -0.121 | 0.000 | -0.593 | -0.592 | -0.581 | 0.244 | -0.221 |
| F1p | -0.128 | 0.001 | -0.091 | -0.109 | -0.085 | -0.114 | -0.077 |
| F2 | -0.145 | 0.089 | 0.237 | 0.033 | 0.246 | -0.087 | 0.082 |
| F3 | -0.118 | 0.074 | -0.108 | -0.169 | -0.056 | -0.103 | 0.012 |
| F4 | 0.030 | -0.013 | 0.066 | -0.088 | 0.070 | -0.002 | 0.048 |
| F5 | 0.018 | -0.000 | 0.023 | -0.118 | 0.040 | 0.021 | 0.032 |
| F6 | -0.198 | 0.000 | -0.987 | -0.363 | -1.300 | 0.262 | -0.138 |
| F7 | -0.209 | 0.000 | -0.917 | -0.519 | -1.178 | 0.264 | -0.100 |

### Same grid, median Spearman ρ


**ETTh1**

| feature_set | gbm | iso_col | iso_ols | mlp | ols | rf | ridge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | -0.025 | -0.008 | -0.028 | -0.020 | -0.040 | -0.036 | -0.041 |
| F1 | -0.200 | -0.336 | -0.160 | -0.127 | -0.107 | -0.117 | -0.183 |
| F1p | 0.169 | -0.007 | 0.112 | 0.015 | 0.103 | 0.220 | 0.091 |
| F2 | 0.003 | 0.000 | -0.257 | -0.023 | -0.243 | 0.012 | 0.072 |
| F3 | -0.134 | -0.039 | 0.036 | 0.258 | 0.047 | -0.117 | 0.187 |
| F4 | -0.145 | -0.043 | -0.000 | 0.009 | 0.006 | -0.099 | 0.006 |
| F5 | -0.069 | 0.006 | -0.044 | -0.035 | -0.047 | -0.069 | -0.047 |
| F6 | -0.128 | -0.168 | -0.184 | -0.042 | -0.139 | 0.021 | -0.056 |
| F7 | -0.118 | -0.168 | -0.180 | -0.049 | -0.130 | 0.021 | -0.059 |

**ETTh2**

| feature_set | gbm | iso_col | iso_ols | mlp | ols | rf | ridge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 0.033 | 0.000 | 0.055 | 0.046 | 0.060 | 0.028 | 0.063 |
| F1 | 0.370 | 0.000 | 0.024 | 0.112 | 0.041 | 0.378 | 0.176 |
| F1p | 0.265 | 0.000 | 0.320 | 0.350 | 0.325 | 0.300 | 0.274 |
| F2 | -0.232 | 0.268 | -0.014 | 0.120 | -0.024 | 0.043 | 0.325 |
| F3 | 0.221 | -0.048 | 0.133 | 0.410 | 0.143 | 0.302 | 0.174 |
| F4 | 0.263 | 0.053 | 0.366 | 0.126 | 0.377 | 0.219 | 0.377 |
| F5 | 0.189 | 0.000 | 0.264 | 0.155 | 0.265 | 0.196 | 0.265 |
| F6 | 0.472 | 0.000 | 0.012 | 0.131 | 0.012 | 0.481 | 0.210 |
| F7 | 0.459 | 0.000 | 0.009 | 0.142 | 0.006 | 0.483 | 0.198 |

### CV-selected estimator, E_eval_shift (models fit on the ID fit split)

| dataset | set | R² median [min,max] | ρ median [min,max] | k/N predictive |
| --- | --- | --- | --- | --- |
| ETTh1 | F0 MVDF | -0.000 [-0.024,+0.004] | -0.02 [-0.08,+0.02] | 0/10 |
| ETTh1 | F1 lagged S (full) | -0.159 [-0.253,-0.110] | -0.15 [-0.25,-0.03] | 0/10 |
| ETTh1 | F1p lagged S (partial) | +0.025 [+0.014,+0.034] | +0.14 [+0.10,+0.18] | 7/10 |
| ETTh1 | F2 input stats | +0.031 [-0.616,+0.164] | +0.11 [-0.45,+0.30] | 3/10 |
| ETTh1 | F3 forecast stats | +0.003 [-0.000,+0.017] | +0.04 [+0.00,+0.16] | 0/10 |
| ETTh1 | F4 hidden PCA-12 | -0.000 [-0.006,+0.003] | +0.01 [-0.03,+0.08] | 0/10 |
| ETTh1 | F5 SAE PCA-12 | +0.001 [-0.019,+0.015] | -0.01 [-0.09,+0.08] | 1/10 |
| ETTh1 | F6 all ex-ante | -0.024 [-0.153,+0.055] | -0.08 [-0.25,+0.05] | 0/10 |
| ETTh1 | F7 all + MVDF | -0.050 [-0.152,+0.021] | -0.13 [-0.28,+0.00] | 0/10 |
| ETTh2 | F0 MVDF | -0.028 [-0.089,-0.000] | -0.03 [-0.11,+0.00] | 0/10 |
| ETTh2 | F1 lagged S (full) | +0.000 [-0.000,+0.012] | +0.00 [+0.00,+0.00] | 0/10 |
| ETTh2 | F1p lagged S (partial) | +0.000 [-0.094,+0.082] | +0.00 [-0.07,+0.26] | 0/10 |
| ETTh2 | F2 input stats | +0.037 [+0.026,+0.048] | -0.21 [-0.23,-0.18] | 0/10 |
| ETTh2 | F3 forecast stats | -0.029 [-0.093,+0.008] | -0.12 [-0.15,+0.03] | 0/10 |
| ETTh2 | F4 hidden PCA-12 | -0.067 [-0.076,-0.036] | -0.12 [-0.13,-0.07] | 0/10 |
| ETTh2 | F5 SAE PCA-12 | -0.102 [-0.325,-0.002] | -0.16 [-0.21,+0.02] | 0/10 |
| ETTh2 | F6 all ex-ante | +0.000 [-0.000,+0.012] | +0.00 [+0.00,+0.00] | 0/10 |
| ETTh2 | F7 all + MVDF | +0.000 [-0.000,+0.012] | +0.00 [+0.00,+0.00] | 0/10 |

### C3.3 ΔR² = R²(F7) − R²(F6), F6's CV-selected estimator, E_eval_ID, block 96

| dataset | seed | estimator | R2_F6 | R2_F7 | dR2 | CI |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh1 | 1 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh1 | 2 | gbm | 0.165 | 0.164 | -0.001 | [-0.024,+0.075] |
| ETTh1 | 3 | gbm | 0.336 | 0.327 | -0.009 | [-0.017,+0.046] |
| ETTh1 | 4 | gbm | 0.195 | 0.203 | 0.007 | [-0.084,+0.048] |
| ETTh1 | 5 | rf | 0.115 | 0.115 | 0.000 | [-0.000,+0.001] |
| ETTh1 | 6 | gbm | 0.210 | 0.145 | -0.065 | [-0.084,-0.026] |
| ETTh1 | 7 | rf | 0.131 | 0.131 | -0.000 | [-0.000,+0.000] |
| ETTh1 | 8 | gbm | 0.371 | 0.374 | 0.003 | [-0.032,+0.195] |
| ETTh1 | 9 | gbm | 0.168 | 0.134 | -0.034 | [-0.064,+0.074] |
| ETTh2 | 0 | iso_col | -0.000 | -0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 1 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 2 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 3 | iso_col | -0.000 | -0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 4 | iso_col | 0.012 | 0.012 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 5 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 6 | iso_col | 0.010 | 0.010 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 7 | iso_col | 0.000 | 0.000 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 8 | iso_col | 0.011 | 0.011 | 0.000 | [+0.000,+0.000] |
| ETTh2 | 9 | iso_col | -0.000 | -0.000 | 0.000 | [+0.000,+0.000] |

### Gap recovery (static split conformal; 50/30/20 of E_eval_ID; same test windows for all)


**ETTh1 — ID** (n_test = 394)

| seed | feature_set | estimator | coverage | width | W_naive | W_oracle | GRR_str | best_exante |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | F0 | rf | 0.703 | 1.753 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F0 | ridge | 0.683 | 1.711 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F0 | ridge | 0.683 | 1.741 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F0 | rf | 0.698 | 1.744 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F0 | iso_col | 0.711 | 1.747 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F0 | rf | 0.701 | 1.743 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F0 | rf | 0.698 | 1.719 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F0 | iso_col | 0.693 | 1.729 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F0 | rf | 0.690 | 1.735 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F0 | iso_col | 0.703 | 1.746 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F1 | mlp | 0.665 | 1.559 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) | ★ |
| 1 | F1 | iso_ols | 0.647 | 1.523 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) | ★ |
| 2 | F1 | gbm | 0.660 | 1.594 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) | ★ |
| 3 | F1 | ridge | 0.652 | 1.608 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) | ★ |
| 4 | F1 | gbm | 0.845 | 2.019 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) | ★ |
| 5 | F1 | ridge | 0.619 | 1.531 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) | ★ |
| 6 | F1 | gbm | 0.650 | 1.724 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) | ★ |
| 7 | F1 | rf | 0.650 | 1.617 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F1 | mlp | 0.734 | 2.023 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F1 | rf | 0.683 | 1.651 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) | ★ |
| 0 | F1p | iso_col | 0.706 | 1.793 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F1p | iso_col | 0.685 | 1.686 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F1p | iso_col | 0.685 | 1.717 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F1p | iso_col | 0.701 | 1.720 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F1p | iso_col | 0.695 | 1.745 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F1p | iso_col | 0.718 | 1.787 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F1p | iso_col | 0.685 | 1.741 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F1p | iso_col | 0.695 | 1.720 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F1p | iso_col | 0.690 | 1.725 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) | ★ |
| 9 | F1p | iso_col | 0.701 | 1.751 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F2 | gbm | 0.774 | 2.149 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F2 | mlp | 0.624 | 1.561 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F2 | rf | 0.701 | 1.881 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F2 | rf | 0.708 | 2.203 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F2 | rf | 0.642 | 1.694 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F2 | rf | 0.632 | 1.830 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F2 | gbm | 0.698 | 2.097 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F2 | rf | 0.640 | 1.814 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F2 | gbm | 0.774 | 2.210 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F2 | iso_col | 0.701 | 1.741 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F3 | iso_col | 0.695 | 1.745 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F3 | iso_col | 0.685 | 1.686 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F3 | ols | 0.734 | 1.865 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F3 | ols | 0.744 | 1.924 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F3 | ridge | 0.751 | 1.945 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F3 | ols | 0.746 | 1.927 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F3 | iso_col | 0.693 | 1.730 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F3 | iso_ols | 0.739 | 1.940 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) | ★ |
| 8 | F3 | iso_col | 0.701 | 1.742 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F3 | iso_col | 0.701 | 1.741 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F4 | ridge | 0.711 | 1.776 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F4 | ridge | 0.698 | 1.757 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F4 | ridge | 0.670 | 1.741 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F4 | ridge | 0.716 | 1.773 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F4 | ridge | 0.746 | 1.765 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F4 | ridge | 0.721 | 1.857 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F4 | iso_col | 0.693 | 1.732 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F4 | ridge | 0.695 | 1.782 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F4 | ridge | 0.713 | 1.801 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F4 | ridge | 0.701 | 1.734 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F5 | ridge | 0.723 | 1.758 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F5 | ridge | 0.736 | 1.771 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F5 | ridge | 0.713 | 1.756 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F5 | ridge | 0.721 | 1.795 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F5 | iso_col | 0.749 | 1.798 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F5 | ridge | 0.703 | 1.781 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F5 | iso_col | 0.711 | 1.790 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F5 | ridge | 0.701 | 1.801 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F5 | iso_col | 0.713 | 1.765 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F5 | rf | 0.721 | 1.793 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F6 | iso_col | 0.690 | 1.726 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F6 | iso_col | 0.685 | 1.686 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F6 | rf | 0.690 | 1.764 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F6 | ridge | 0.721 | 1.757 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F6 | rf | 0.673 | 1.769 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F6 | rf | 0.708 | 1.759 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F6 | iso_col | 0.693 | 1.730 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F6 | rf | 0.617 | 1.605 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F6 | gbm | 0.731 | 2.005 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F6 | gbm | 0.766 | 1.889 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F7 | iso_col | 0.690 | 1.726 | 1.527 | 1.603 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F7 | iso_col | 0.685 | 1.686 | 1.541 | 1.593 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F7 | rf | 0.690 | 1.762 | 1.524 | 1.550 | undefined (W_naive ≤ W_oracle) |  |
| 3 | F7 | ridge | 0.706 | 1.706 | 1.535 | 1.565 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F7 | rf | 0.668 | 1.782 | 1.523 | 1.557 | undefined (W_naive ≤ W_oracle) |  |
| 5 | F7 | rf | 0.754 | 1.812 | 1.522 | 1.560 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F7 | iso_col | 0.693 | 1.730 | 1.520 | 1.576 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F7 | rf | 0.617 | 1.605 | 1.519 | 1.568 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F7 | gbm | 0.718 | 1.972 | 1.527 | 1.581 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F7 | gbm | 0.827 | 1.870 | 1.531 | 1.582 | undefined (W_naive ≤ W_oracle) |  |
| 0 | naive_matched | - | 0.632 | 1.527 | 1.527 | 1.603 | — |  |
| 1 | naive_matched | - | 0.609 | 1.541 | 1.541 | 1.593 | — |  |
| 2 | naive_matched | - | 0.629 | 1.524 | 1.524 | 1.550 | — |  |
| 3 | naive_matched | - | 0.612 | 1.535 | 1.535 | 1.565 | — |  |
| 4 | naive_matched | - | 0.619 | 1.523 | 1.523 | 1.557 | — |  |
| 5 | naive_matched | - | 0.617 | 1.522 | 1.522 | 1.560 | — |  |
| 6 | naive_matched | - | 0.607 | 1.520 | 1.520 | 1.576 | — |  |
| 7 | naive_matched | - | 0.584 | 1.519 | 1.519 | 1.568 | — |  |
| 8 | naive_matched | - | 0.617 | 1.527 | 1.527 | 1.581 | — |  |
| 9 | naive_matched | - | 0.622 | 1.531 | 1.531 | 1.582 | — |  |

**ETTh1 — shift** (n_test = 1871)

| seed | feature_set | estimator | coverage | width | W_naive | W_oracle | GRR_str | best_exante |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | F0 | rf | 0.866 | 1.689 | 1.527 | 1.349 | coverage-invalid (-0.92) |  |
| 1 | F0 | ridge | 0.875 | 1.707 | 1.541 | 1.338 | coverage-invalid (-0.82) |  |
| 2 | F0 | ridge | 0.895 | 1.766 | 1.524 | 1.324 | -1.21 |  |
| 3 | F0 | rf | 0.882 | 1.729 | 1.535 | 1.336 | -0.97 |  |
| 4 | F0 | iso_col | 0.898 | 1.747 | 1.523 | 1.339 | -1.21 |  |
| 5 | F0 | rf | 0.887 | 1.726 | 1.522 | 1.332 | -1.08 |  |
| 6 | F0 | rf | 0.890 | 1.728 | 1.520 | 1.346 | -1.20 |  |
| 7 | F0 | iso_col | 0.900 | 1.729 | 1.519 | 1.327 | -1.09 |  |
| 8 | F0 | rf | 0.894 | 1.738 | 1.527 | 1.344 | -1.15 |  |
| 9 | F0 | iso_col | 0.892 | 1.747 | 1.531 | 1.337 | -1.11 |  |
| 0 | F1 | mlp | 0.751 | 1.618 | 1.527 | 1.349 | coverage-invalid (-0.51) | ★ |
| 1 | F1 | iso_ols | 0.697 | 1.481 | 1.541 | 1.338 | coverage-invalid (+0.30) | ★ |
| 2 | F1 | gbm | 0.826 | 1.642 | 1.524 | 1.324 | coverage-invalid (-0.59) | ★ |
| 3 | F1 | ridge | 0.812 | 1.644 | 1.535 | 1.336 | coverage-invalid (-0.55) | ★ |
| 4 | F1 | gbm | 0.873 | 1.925 | 1.523 | 1.339 | coverage-invalid (-2.18) | ★ |
| 5 | F1 | ridge | 0.789 | 1.580 | 1.522 | 1.332 | coverage-invalid (-0.31) | ★ |
| 6 | F1 | gbm | 0.843 | 1.750 | 1.520 | 1.346 | coverage-invalid (-1.33) | ★ |
| 7 | F1 | rf | 0.781 | 1.676 | 1.519 | 1.327 | coverage-invalid (-0.81) |  |
| 8 | F1 | mlp | 0.820 | 1.794 | 1.527 | 1.344 | coverage-invalid (-1.45) |  |
| 9 | F1 | rf | 0.732 | 1.555 | 1.531 | 1.337 | coverage-invalid (-0.12) | ★ |
| 0 | F1p | iso_col | 0.896 | 1.802 | 1.527 | 1.349 | -1.55 |  |
| 1 | F1p | iso_col | 0.872 | 1.686 | 1.541 | 1.338 | coverage-invalid (-0.72) |  |
| 2 | F1p | iso_col | 0.877 | 1.694 | 1.524 | 1.324 | coverage-invalid (-0.85) |  |
| 3 | F1p | iso_col | 0.887 | 1.720 | 1.535 | 1.336 | -0.93 |  |
| 4 | F1p | iso_col | 0.873 | 1.709 | 1.523 | 1.339 | coverage-invalid (-1.01) |  |
| 5 | F1p | iso_col | 0.895 | 1.780 | 1.522 | 1.332 | -1.36 |  |
| 6 | F1p | iso_col | 0.872 | 1.714 | 1.520 | 1.346 | coverage-invalid (-1.12) |  |
| 7 | F1p | iso_col | 0.882 | 1.706 | 1.519 | 1.327 | -0.97 |  |
| 8 | F1p | iso_col | 0.889 | 1.725 | 1.527 | 1.344 | -1.08 | ★ |
| 9 | F1p | iso_col | 0.865 | 1.721 | 1.531 | 1.337 | coverage-invalid (-0.98) |  |
| 0 | F2 | gbm | 0.848 | 1.857 | 1.527 | 1.349 | coverage-invalid (-1.86) |  |
| 1 | F2 | mlp | 0.839 | 1.678 | 1.541 | 1.338 | coverage-invalid (-0.68) |  |
| 2 | F2 | rf | 0.908 | 2.067 | 1.524 | 1.324 | -2.72 |  |
| 3 | F2 | rf | 0.818 | 1.816 | 1.535 | 1.336 | coverage-invalid (-1.41) |  |
| 4 | F2 | rf | 0.796 | 1.737 | 1.523 | 1.339 | coverage-invalid (-1.16) |  |
| 5 | F2 | rf | 0.811 | 1.773 | 1.522 | 1.332 | coverage-invalid (-1.33) |  |
| 6 | F2 | gbm | 0.866 | 1.880 | 1.520 | 1.346 | coverage-invalid (-2.08) |  |
| 7 | F2 | rf | 0.805 | 1.773 | 1.519 | 1.327 | coverage-invalid (-1.32) |  |
| 8 | F2 | gbm | 0.867 | 1.842 | 1.527 | 1.344 | coverage-invalid (-1.72) |  |
| 9 | F2 | iso_col | 0.889 | 1.741 | 1.531 | 1.337 | -1.08 |  |
| 0 | F3 | iso_col | 0.890 | 1.745 | 1.527 | 1.349 | -1.23 |  |
| 1 | F3 | iso_col | 0.872 | 1.686 | 1.541 | 1.338 | coverage-invalid (-0.72) |  |
| 2 | F3 | ols | 0.907 | 1.989 | 1.524 | 1.324 | -2.33 |  |
| 3 | F3 | ols | 0.914 | 1.987 | 1.535 | 1.336 | -2.27 |  |
| 4 | F3 | ridge | 0.895 | 2.018 | 1.523 | 1.339 | -2.69 |  |
| 5 | F3 | ols | 0.887 | 2.026 | 1.522 | 1.332 | -2.66 |  |
| 6 | F3 | iso_col | 0.890 | 1.730 | 1.520 | 1.346 | -1.21 |  |
| 7 | F3 | iso_ols | 0.910 | 1.984 | 1.519 | 1.327 | -2.41 | ★ |
| 8 | F3 | iso_col | 0.896 | 1.745 | 1.527 | 1.344 | -1.19 |  |
| 9 | F3 | iso_col | 0.889 | 1.741 | 1.531 | 1.337 | -1.08 |  |
| 0 | F4 | ridge | 0.881 | 1.749 | 1.527 | 1.349 | -1.25 |  |
| 1 | F4 | ridge | 0.844 | 1.750 | 1.541 | 1.338 | coverage-invalid (-1.04) |  |
| 2 | F4 | ridge | 0.873 | 1.727 | 1.524 | 1.324 | coverage-invalid (-1.02) |  |
| 3 | F4 | ridge | 0.879 | 1.770 | 1.535 | 1.336 | coverage-invalid (-1.18) |  |
| 4 | F4 | ridge | 0.881 | 1.751 | 1.523 | 1.339 | -1.24 |  |
| 5 | F4 | ridge | 0.883 | 1.820 | 1.522 | 1.332 | -1.58 |  |
| 6 | F4 | iso_col | 0.893 | 1.734 | 1.520 | 1.346 | -1.24 |  |
| 7 | F4 | ridge | 0.882 | 1.750 | 1.519 | 1.327 | -1.20 |  |
| 8 | F4 | ridge | 0.869 | 1.735 | 1.527 | 1.344 | coverage-invalid (-1.14) |  |
| 9 | F4 | ridge | 0.858 | 1.732 | 1.531 | 1.337 | coverage-invalid (-1.04) |  |
| 0 | F5 | ridge | 0.891 | 1.760 | 1.527 | 1.349 | -1.31 |  |
| 1 | F5 | ridge | 0.874 | 1.745 | 1.541 | 1.338 | coverage-invalid (-1.01) |  |
| 2 | F5 | ridge | 0.875 | 1.740 | 1.524 | 1.324 | coverage-invalid (-1.08) |  |
| 3 | F5 | ridge | 0.867 | 1.800 | 1.535 | 1.336 | coverage-invalid (-1.33) |  |
| 4 | F5 | iso_col | 0.887 | 1.803 | 1.523 | 1.339 | -1.52 |  |
| 5 | F5 | ridge | 0.901 | 1.777 | 1.522 | 1.332 | -1.35 |  |
| 6 | F5 | iso_col | 0.896 | 1.811 | 1.520 | 1.346 | -1.67 |  |
| 7 | F5 | ridge | 0.886 | 1.792 | 1.519 | 1.327 | -1.42 |  |
| 8 | F5 | iso_col | 0.888 | 1.820 | 1.527 | 1.344 | -1.60 |  |
| 9 | F5 | rf | 0.886 | 1.810 | 1.531 | 1.337 | -1.44 |  |
| 0 | F6 | iso_col | 0.885 | 1.726 | 1.527 | 1.349 | -1.12 |  |
| 1 | F6 | iso_col | 0.872 | 1.686 | 1.541 | 1.338 | coverage-invalid (-0.72) |  |
| 2 | F6 | rf | 0.905 | 1.770 | 1.524 | 1.324 | -1.23 |  |
| 3 | F6 | ridge | 0.899 | 1.775 | 1.535 | 1.336 | -1.21 |  |
| 4 | F6 | rf | 0.906 | 1.778 | 1.523 | 1.339 | -1.38 |  |
| 5 | F6 | rf | 0.864 | 1.757 | 1.522 | 1.332 | coverage-invalid (-1.24) |  |
| 6 | F6 | iso_col | 0.890 | 1.730 | 1.520 | 1.346 | -1.21 |  |
| 7 | F6 | rf | 0.879 | 1.807 | 1.519 | 1.327 | coverage-invalid (-1.49) |  |
| 8 | F6 | gbm | 0.870 | 1.810 | 1.527 | 1.344 | coverage-invalid (-1.54) |  |
| 9 | F6 | gbm | 0.881 | 1.815 | 1.531 | 1.337 | -1.47 |  |
| 0 | F7 | iso_col | 0.885 | 1.726 | 1.527 | 1.349 | -1.12 |  |
| 1 | F7 | iso_col | 0.872 | 1.686 | 1.541 | 1.338 | coverage-invalid (-0.72) |  |
| 2 | F7 | rf | 0.905 | 1.769 | 1.524 | 1.324 | -1.23 |  |
| 3 | F7 | ridge | 0.879 | 1.745 | 1.535 | 1.336 | coverage-invalid (-1.05) |  |
| 4 | F7 | rf | 0.905 | 1.781 | 1.523 | 1.339 | -1.40 |  |
| 5 | F7 | rf | 0.867 | 1.790 | 1.522 | 1.332 | coverage-invalid (-1.42) |  |
| 6 | F7 | iso_col | 0.890 | 1.730 | 1.520 | 1.346 | -1.21 |  |
| 7 | F7 | rf | 0.879 | 1.808 | 1.519 | 1.327 | coverage-invalid (-1.50) |  |
| 8 | F7 | gbm | 0.885 | 1.842 | 1.527 | 1.344 | -1.72 |  |
| 9 | F7 | gbm | 0.890 | 1.856 | 1.531 | 1.337 | -1.68 |  |
| 0 | naive_matched | - | 0.765 | 1.527 | 1.527 | 1.349 | — |  |
| 1 | naive_matched | - | 0.781 | 1.541 | 1.541 | 1.338 | — |  |
| 2 | naive_matched | - | 0.800 | 1.524 | 1.524 | 1.324 | — |  |
| 3 | naive_matched | - | 0.796 | 1.535 | 1.535 | 1.336 | — |  |
| 4 | naive_matched | - | 0.781 | 1.523 | 1.523 | 1.339 | — |  |
| 5 | naive_matched | - | 0.782 | 1.522 | 1.522 | 1.332 | — |  |
| 6 | naive_matched | - | 0.769 | 1.520 | 1.520 | 1.346 | — |  |
| 7 | naive_matched | - | 0.793 | 1.519 | 1.519 | 1.327 | — |  |
| 8 | naive_matched | - | 0.772 | 1.527 | 1.527 | 1.344 | — |  |
| 9 | naive_matched | - | 0.770 | 1.531 | 1.531 | 1.337 | — |  |

**ETTh2 — ID** (n_test = 394)

| seed | feature_set | estimator | coverage | width | W_naive | W_oracle | GRR_str | best_exante |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | F0 | rf | 0.690 | 3.346 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F0 | iso_col | 0.701 | 3.385 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F0 | rf | 0.716 | 3.363 | 3.010 | 3.009 | coverage-invalid (-199.32) |  |
| 3 | F0 | rf | 0.695 | 3.385 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F0 | rf | 0.693 | 3.343 | 3.033 | 3.010 | coverage-invalid (-13.43) |  |
| 5 | F0 | ridge | 0.739 | 3.451 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F0 | rf | 0.721 | 3.392 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F0 | rf | 0.693 | 3.374 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F0 | rf | 0.579 | 3.114 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F0 | ridge | 0.668 | 3.277 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F1 | iso_col | 0.683 | 3.333 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F1 | iso_col | 0.701 | 3.385 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F1 | iso_col | 0.718 | 3.387 | 3.010 | 3.009 | coverage-invalid (-213.27) |  |
| 3 | F1 | iso_col | 0.690 | 3.378 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) | ★ |
| 4 | F1 | iso_col | 0.711 | 3.393 | 3.033 | 3.010 | coverage-invalid (-15.61) |  |
| 5 | F1 | iso_col | 0.716 | 3.424 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F1 | iso_col | 0.701 | 3.328 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F1 | iso_col | 0.693 | 3.393 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F1 | iso_col | 0.708 | 3.408 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) | ★ |
| 9 | F1 | iso_col | 0.708 | 3.377 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) | ★ |
| 0 | F1p | ridge | 0.706 | 3.201 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) | ★ |
| 1 | F1p | ridge | 0.721 | 3.367 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) | ★ |
| 2 | F1p | ridge | 0.698 | 3.234 | 3.010 | 3.009 | coverage-invalid (-126.62) | ★ |
| 3 | F1p | ridge | 0.693 | 3.246 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F1p | ridge | 0.693 | 3.292 | 3.033 | 3.010 | coverage-invalid (-11.22) | ★ |
| 5 | F1p | ridge | 0.746 | 3.350 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) | ★ |
| 6 | F1p | ridge | 0.723 | 3.316 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) | ★ |
| 7 | F1p | ridge | 0.698 | 3.251 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) | ★ |
| 8 | F1p | iso_col | 0.708 | 3.408 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F1p | ridge | 0.695 | 3.229 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F2 | iso_ols | 0.657 | 3.247 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F2 | iso_col | 0.708 | 3.328 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F2 | iso_col | 0.761 | 3.284 | 3.010 | 3.009 | coverage-invalid (-155.03) |  |
| 3 | F2 | iso_col | 0.660 | 3.220 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F2 | ridge | 0.675 | 3.296 | 3.033 | 3.010 | coverage-invalid (-11.41) |  |
| 5 | F2 | ridge | 0.690 | 3.319 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F2 | ridge | 0.711 | 3.298 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F2 | ridge | 0.678 | 3.339 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F2 | iso_ols | 0.607 | 3.237 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F2 | iso_col | 0.688 | 3.251 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F3 | ridge | 0.693 | 3.030 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F3 | ridge | 0.685 | 3.130 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F3 | ridge | 0.734 | 3.090 | 3.010 | 3.009 | coverage-invalid (-45.15) |  |
| 3 | F3 | ridge | 0.698 | 3.128 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F3 | ridge | 0.675 | 3.230 | 3.033 | 3.010 | coverage-invalid (-8.52) |  |
| 5 | F3 | ridge | 0.690 | 3.159 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F3 | ridge | 0.698 | 3.107 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F3 | ridge | 0.695 | 3.072 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F3 | iso_col | 0.716 | 3.352 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F3 | ridge | 0.685 | 3.164 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F4 | ridge | 0.685 | 3.248 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F4 | ridge | 0.701 | 3.311 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F4 | ridge | 0.726 | 3.294 | 3.010 | 3.009 | coverage-invalid (-160.54) |  |
| 3 | F4 | ridge | 0.650 | 3.211 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F4 | ridge | 0.706 | 3.351 | 3.033 | 3.010 | coverage-invalid (-13.79) |  |
| 5 | F4 | ridge | 0.708 | 3.375 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F4 | ridge | 0.683 | 3.271 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F4 | ridge | 0.662 | 3.289 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F4 | ridge | 0.698 | 3.332 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F4 | ridge | 0.642 | 3.260 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F5 | ridge | 0.678 | 3.273 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F5 | ridge | 0.713 | 3.375 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F5 | ridge | 0.675 | 3.312 | 3.010 | 3.009 | coverage-invalid (-170.59) |  |
| 3 | F5 | ridge | 0.675 | 3.321 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F5 | ridge | 0.670 | 3.354 | 3.033 | 3.010 | coverage-invalid (-13.90) |  |
| 5 | F5 | ridge | 0.668 | 3.215 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F5 | ridge | 0.665 | 3.248 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F5 | rf | 0.690 | 3.363 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F5 | ridge | 0.665 | 3.297 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F5 | ridge | 0.655 | 3.257 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F6 | iso_col | 0.683 | 3.333 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F6 | iso_col | 0.701 | 3.385 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F6 | iso_col | 0.718 | 3.387 | 3.010 | 3.009 | coverage-invalid (-213.27) |  |
| 3 | F6 | iso_col | 0.690 | 3.378 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F6 | iso_col | 0.711 | 3.393 | 3.033 | 3.010 | coverage-invalid (-15.61) |  |
| 5 | F6 | iso_col | 0.716 | 3.424 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F6 | iso_col | 0.701 | 3.328 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F6 | iso_col | 0.693 | 3.393 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F6 | iso_col | 0.708 | 3.408 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F6 | iso_col | 0.708 | 3.377 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | F7 | iso_col | 0.683 | 3.333 | 2.934 | 2.953 | undefined (W_naive ≤ W_oracle) |  |
| 1 | F7 | iso_col | 0.701 | 3.385 | 2.990 | 2.997 | undefined (W_naive ≤ W_oracle) |  |
| 2 | F7 | iso_col | 0.718 | 3.387 | 3.010 | 3.009 | coverage-invalid (-213.27) |  |
| 3 | F7 | iso_col | 0.690 | 3.378 | 2.996 | 3.000 | undefined (W_naive ≤ W_oracle) |  |
| 4 | F7 | iso_col | 0.711 | 3.393 | 3.033 | 3.010 | coverage-invalid (-15.61) |  |
| 5 | F7 | iso_col | 0.716 | 3.424 | 3.014 | 3.025 | undefined (W_naive ≤ W_oracle) |  |
| 6 | F7 | iso_col | 0.701 | 3.328 | 2.944 | 2.958 | undefined (W_naive ≤ W_oracle) |  |
| 7 | F7 | iso_col | 0.693 | 3.393 | 2.985 | 3.024 | undefined (W_naive ≤ W_oracle) |  |
| 8 | F7 | iso_col | 0.708 | 3.408 | 2.955 | 3.020 | undefined (W_naive ≤ W_oracle) |  |
| 9 | F7 | iso_col | 0.708 | 3.377 | 3.014 | 3.029 | undefined (W_naive ≤ W_oracle) |  |
| 0 | naive_matched | - | 0.548 | 2.934 | 2.934 | 2.953 | — |  |
| 1 | naive_matched | - | 0.586 | 2.990 | 2.990 | 2.997 | — |  |
| 2 | naive_matched | - | 0.589 | 3.010 | 3.010 | 3.009 | — |  |
| 3 | naive_matched | - | 0.556 | 2.996 | 2.996 | 3.000 | — |  |
| 4 | naive_matched | - | 0.563 | 3.033 | 3.033 | 3.010 | — |  |
| 5 | naive_matched | - | 0.563 | 3.014 | 3.014 | 3.025 | — |  |
| 6 | naive_matched | - | 0.594 | 2.944 | 2.944 | 2.958 | — |  |
| 7 | naive_matched | - | 0.551 | 2.985 | 2.985 | 3.024 | — |  |
| 8 | naive_matched | - | 0.530 | 2.955 | 2.955 | 3.020 | — |  |
| 9 | naive_matched | - | 0.566 | 3.014 | 3.014 | 3.029 | — |  |

**ETTh2 — shift** (n_test = 1871)

| seed | feature_set | estimator | coverage | width | W_naive | W_oracle | GRR_str | best_exante |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | F0 | rf | 0.982 | 3.339 | 2.934 | 2.243 | coverage-invalid (-0.59) |  |
| 1 | F0 | iso_col | 0.974 | 3.385 | 2.990 | 2.281 | coverage-invalid (-0.56) |  |
| 2 | F0 | rf | 0.979 | 3.348 | 3.010 | 2.241 | coverage-invalid (-0.44) |  |
| 3 | F0 | rf | 0.982 | 3.388 | 2.996 | 2.252 | coverage-invalid (-0.53) |  |
| 4 | F0 | rf | 0.965 | 3.355 | 3.033 | 2.278 | coverage-invalid (-0.43) |  |
| 5 | F0 | ridge | 0.981 | 3.444 | 3.014 | 2.287 | coverage-invalid (-0.59) |  |
| 6 | F0 | rf | 0.977 | 3.377 | 2.944 | 2.263 | coverage-invalid (-0.64) |  |
| 7 | F0 | rf | 0.976 | 3.370 | 2.985 | 2.292 | coverage-invalid (-0.55) |  |
| 8 | F0 | rf | 0.975 | 3.384 | 2.955 | 2.292 | coverage-invalid (-0.65) |  |
| 9 | F0 | ridge | 0.960 | 3.268 | 3.014 | 2.269 | coverage-invalid (-0.34) |  |
| 0 | F1 | iso_col | 0.979 | 3.333 | 2.934 | 2.243 | coverage-invalid (-0.58) |  |
| 1 | F1 | iso_col | 0.974 | 3.385 | 2.990 | 2.281 | coverage-invalid (-0.56) |  |
| 2 | F1 | iso_col | 0.984 | 3.387 | 3.010 | 2.241 | coverage-invalid (-0.49) |  |
| 3 | F1 | iso_col | 0.985 | 3.378 | 2.996 | 2.252 | coverage-invalid (-0.51) | ★ |
| 4 | F1 | iso_col | 0.981 | 3.393 | 3.033 | 2.278 | coverage-invalid (-0.48) |  |
| 5 | F1 | iso_col | 0.982 | 3.424 | 3.014 | 2.287 | coverage-invalid (-0.56) |  |
| 6 | F1 | iso_col | 0.975 | 3.328 | 2.944 | 2.263 | coverage-invalid (-0.56) |  |
| 7 | F1 | iso_col | 0.976 | 3.393 | 2.985 | 2.292 | coverage-invalid (-0.59) |  |
| 8 | F1 | iso_col | 0.980 | 3.408 | 2.955 | 2.292 | coverage-invalid (-0.68) | ★ |
| 9 | F1 | iso_col | 0.982 | 3.377 | 3.014 | 2.269 | coverage-invalid (-0.49) | ★ |
| 0 | F1p | ridge | 0.921 | 3.217 | 2.934 | 2.243 | coverage-invalid (-0.41) | ★ |
| 1 | F1p | ridge | 0.951 | 3.364 | 2.990 | 2.281 | coverage-invalid (-0.53) | ★ |
| 2 | F1p | ridge | 0.932 | 3.263 | 3.010 | 2.241 | coverage-invalid (-0.33) | ★ |
| 3 | F1p | ridge | 0.935 | 3.283 | 2.996 | 2.252 | coverage-invalid (-0.39) |  |
| 4 | F1p | ridge | 0.927 | 3.313 | 3.033 | 2.278 | coverage-invalid (-0.37) | ★ |
| 5 | F1p | ridge | 0.959 | 3.376 | 3.014 | 2.287 | coverage-invalid (-0.50) | ★ |
| 6 | F1p | ridge | 0.952 | 3.329 | 2.944 | 2.263 | coverage-invalid (-0.56) | ★ |
| 7 | F1p | ridge | 0.925 | 3.312 | 2.985 | 2.292 | coverage-invalid (-0.47) | ★ |
| 8 | F1p | iso_col | 0.980 | 3.408 | 2.955 | 2.292 | coverage-invalid (-0.68) |  |
| 9 | F1p | ridge | 0.932 | 3.265 | 3.014 | 2.269 | coverage-invalid (-0.34) |  |
| 0 | F2 | iso_ols | 0.974 | 3.256 | 2.934 | 2.243 | coverage-invalid (-0.47) |  |
| 1 | F2 | iso_col | 0.939 | 3.272 | 2.990 | 2.281 | coverage-invalid (-0.40) |  |
| 2 | F2 | iso_col | 0.955 | 3.250 | 3.010 | 2.241 | coverage-invalid (-0.31) |  |
| 3 | F2 | iso_col | 0.904 | 3.147 | 2.996 | 2.252 | -0.20 |  |
| 4 | F2 | ridge | 0.965 | 3.361 | 3.033 | 2.278 | coverage-invalid (-0.43) |  |
| 5 | F2 | ridge | 0.974 | 3.348 | 3.014 | 2.287 | coverage-invalid (-0.46) |  |
| 6 | F2 | ridge | 0.965 | 3.260 | 2.944 | 2.263 | coverage-invalid (-0.46) |  |
| 7 | F2 | ridge | 0.960 | 3.337 | 2.985 | 2.292 | coverage-invalid (-0.51) |  |
| 8 | F2 | iso_ols | 0.957 | 3.295 | 2.955 | 2.292 | coverage-invalid (-0.51) |  |
| 9 | F2 | iso_col | 0.908 | 3.159 | 3.014 | 2.269 | -0.20 |  |
| 0 | F3 | ridge | 0.887 | 3.138 | 2.934 | 2.243 | -0.30 |  |
| 1 | F3 | ridge | 0.919 | 3.219 | 2.990 | 2.281 | -0.32 |  |
| 2 | F3 | ridge | 0.909 | 3.205 | 3.010 | 2.241 | -0.25 |  |
| 3 | F3 | ridge | 0.924 | 3.203 | 2.996 | 2.252 | coverage-invalid (-0.28) |  |
| 4 | F3 | ridge | 0.940 | 3.272 | 3.033 | 2.278 | coverage-invalid (-0.32) |  |
| 5 | F3 | ridge | 0.927 | 3.237 | 3.014 | 2.287 | coverage-invalid (-0.31) |  |
| 6 | F3 | ridge | 0.925 | 3.218 | 2.944 | 2.263 | coverage-invalid (-0.40) |  |
| 7 | F3 | ridge | 0.896 | 3.198 | 2.985 | 2.292 | -0.31 |  |
| 8 | F3 | iso_col | 0.966 | 3.332 | 2.955 | 2.292 | coverage-invalid (-0.57) |  |
| 9 | F3 | ridge | 0.926 | 3.236 | 3.014 | 2.269 | coverage-invalid (-0.30) |  |
| 0 | F4 | ridge | 0.937 | 3.228 | 2.934 | 2.243 | coverage-invalid (-0.43) |  |
| 1 | F4 | ridge | 0.893 | 3.225 | 2.990 | 2.281 | -0.33 |  |
| 2 | F4 | ridge | 0.916 | 3.213 | 3.010 | 2.241 | -0.26 |  |
| 3 | F4 | ridge | 0.902 | 3.147 | 2.996 | 2.252 | -0.20 |  |
| 4 | F4 | ridge | 0.935 | 3.293 | 3.033 | 2.278 | coverage-invalid (-0.34) |  |
| 5 | F4 | ridge | 0.905 | 3.262 | 3.014 | 2.287 | -0.34 |  |
| 6 | F4 | ridge | 0.895 | 3.124 | 2.944 | 2.263 | -0.26 |  |
| 7 | F4 | ridge | 0.904 | 3.205 | 2.985 | 2.292 | -0.32 |  |
| 8 | F4 | ridge | 0.901 | 3.215 | 2.955 | 2.292 | -0.39 |  |
| 9 | F4 | ridge | 0.897 | 3.162 | 3.014 | 2.269 | -0.20 |  |
| 0 | F5 | ridge | 0.929 | 3.239 | 2.934 | 2.243 | coverage-invalid (-0.44) |  |
| 1 | F5 | ridge | 0.920 | 3.292 | 2.990 | 2.281 | coverage-invalid (-0.43) |  |
| 2 | F5 | ridge | 0.932 | 3.218 | 3.010 | 2.241 | coverage-invalid (-0.27) |  |
| 3 | F5 | ridge | 0.936 | 3.284 | 2.996 | 2.252 | coverage-invalid (-0.39) |  |
| 4 | F5 | ridge | 0.938 | 3.292 | 3.033 | 2.278 | coverage-invalid (-0.34) |  |
| 5 | F5 | ridge | 0.902 | 3.193 | 3.014 | 2.287 | -0.25 |  |
| 6 | F5 | ridge | 0.916 | 3.139 | 2.944 | 2.263 | -0.29 |  |
| 7 | F5 | rf | 0.964 | 3.363 | 2.985 | 2.292 | coverage-invalid (-0.54) |  |
| 8 | F5 | ridge | 0.902 | 3.226 | 2.955 | 2.292 | -0.41 |  |
| 9 | F5 | ridge | 0.900 | 3.182 | 3.014 | 2.269 | -0.23 |  |
| 0 | F6 | iso_col | 0.979 | 3.333 | 2.934 | 2.243 | coverage-invalid (-0.58) |  |
| 1 | F6 | iso_col | 0.974 | 3.385 | 2.990 | 2.281 | coverage-invalid (-0.56) |  |
| 2 | F6 | iso_col | 0.984 | 3.387 | 3.010 | 2.241 | coverage-invalid (-0.49) |  |
| 3 | F6 | iso_col | 0.985 | 3.378 | 2.996 | 2.252 | coverage-invalid (-0.51) |  |
| 4 | F6 | iso_col | 0.981 | 3.393 | 3.033 | 2.278 | coverage-invalid (-0.48) |  |
| 5 | F6 | iso_col | 0.982 | 3.424 | 3.014 | 2.287 | coverage-invalid (-0.56) |  |
| 6 | F6 | iso_col | 0.975 | 3.328 | 2.944 | 2.263 | coverage-invalid (-0.56) |  |
| 7 | F6 | iso_col | 0.976 | 3.393 | 2.985 | 2.292 | coverage-invalid (-0.59) |  |
| 8 | F6 | iso_col | 0.980 | 3.408 | 2.955 | 2.292 | coverage-invalid (-0.68) |  |
| 9 | F6 | iso_col | 0.982 | 3.377 | 3.014 | 2.269 | coverage-invalid (-0.49) |  |
| 0 | F7 | iso_col | 0.979 | 3.333 | 2.934 | 2.243 | coverage-invalid (-0.58) |  |
| 1 | F7 | iso_col | 0.974 | 3.385 | 2.990 | 2.281 | coverage-invalid (-0.56) |  |
| 2 | F7 | iso_col | 0.984 | 3.387 | 3.010 | 2.241 | coverage-invalid (-0.49) |  |
| 3 | F7 | iso_col | 0.985 | 3.378 | 2.996 | 2.252 | coverage-invalid (-0.51) |  |
| 4 | F7 | iso_col | 0.981 | 3.393 | 3.033 | 2.278 | coverage-invalid (-0.48) |  |
| 5 | F7 | iso_col | 0.982 | 3.424 | 3.014 | 2.287 | coverage-invalid (-0.56) |  |
| 6 | F7 | iso_col | 0.975 | 3.328 | 2.944 | 2.263 | coverage-invalid (-0.56) |  |
| 7 | F7 | iso_col | 0.976 | 3.393 | 2.985 | 2.292 | coverage-invalid (-0.59) |  |
| 8 | F7 | iso_col | 0.980 | 3.408 | 2.955 | 2.292 | coverage-invalid (-0.68) |  |
| 9 | F7 | iso_col | 0.982 | 3.377 | 3.014 | 2.269 | coverage-invalid (-0.49) |  |
| 0 | naive_matched | - | 0.885 | 2.934 | 2.934 | 2.243 | — |  |
| 1 | naive_matched | - | 0.897 | 2.990 | 2.990 | 2.281 | — |  |
| 2 | naive_matched | - | 0.927 | 3.010 | 3.010 | 2.241 | — |  |
| 3 | naive_matched | - | 0.921 | 2.996 | 2.996 | 2.252 | — |  |
| 4 | naive_matched | - | 0.909 | 3.033 | 3.033 | 2.278 | — |  |
| 5 | naive_matched | - | 0.917 | 3.014 | 3.014 | 2.287 | — |  |
| 6 | naive_matched | - | 0.901 | 2.944 | 2.944 | 2.263 | — |  |
| 7 | naive_matched | - | 0.910 | 2.985 | 2.985 | 2.292 | — |  |
| 8 | naive_matched | - | 0.884 | 2.955 | 2.955 | 2.292 | — |  |
| 9 | naive_matched | - | 0.919 | 3.014 | 3.014 | 2.269 | — |  |
