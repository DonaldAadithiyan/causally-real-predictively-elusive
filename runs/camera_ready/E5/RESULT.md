# E5 — Dose-response diagnostics and natural-range interventions

## Provenance

- Git: `93fbb0c` (+uncommitted changes)
- Command: `python -m camera_ready.e5_dose a|c --datasets ETTh1 ETTh2 electricity traffic --seeds <all>; python -m camera_ready.report_e5`
- Seeds: {'ETTh1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ETTh2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'electricity': [0, 1, 2, 3, 4], 'traffic': [0, 1, 2, 3, 4]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684); electricity: Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical; traffic: Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`; electricity `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570`; traffic `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: 1 s
- Generated: 2026-11-14 07:10:07 +0530

## Pre-registration (summary; full text in PREREG.md) — Dose-response diagnostics + natural-range interventions: pre-registration

Committed before the first E5 run.



## Verdicts
| criterion | result |
|---|---|
| reproduction gate | **pass**: 30/30 cells reproduce the stored pool and MVDF set exactly; U_j to ≤ 2.4e-7 |
| **C5a** fit quality + monotonicity | **PASS**: MVDF median linear-fit R² 0.999, median |ρ(λ, U)| 1.000 (240 MVDF features, 20 ETTh cells) |
| **C5b** natural range (λ ≤ 1) | **FAIL**: median Jaccard 0.85 (passes), but one full-λ passing cell (ETTh2 s5) is not reproduced: natural-range enrichment 2.50 vs 2.61, CI lower bound exactly 1.000 |
| **C5c** interchange sign agreement ≥ 70% on ≥ 8/10 seeds | **FAIL** on ETTh1 (7/10), passes on ETTh2 (9/10) |

## Plain-English summary
The six-point dose-response curves behaved cleanly. For MVDF features the linear fit that defines U_j explains a median 99.9% of the variance, every curve is monotone, and 78% of features show statistically detectable but tiny curvature. The concern that the effect is an artefact of pushing activations past their natural range is answered: re-estimating P_j and U_j from scaling-down only (λ ≤ 1) reselects largely the same features (median Jaccard 0.85) with essentially unchanged enrichment. It misses the pre-registered bar on one boundary cell. The interchange test, which sets a feature to a value it naturally takes on another input, agrees in sign with the scaling-based U_j for 81% of MVDF features on ETTh (ETTh1 77%, ETTh2 85%; electricity 98%, traffic 85%). On 3 of 10 ETTh1 seeds agreement is 58–67%, so the "causal role of the natural mechanism" agrees with the artificial intervention in most but not all cases on ETTh1.

## Anomalies / interpretation notes
- **|Spearman monotonicity| = 1 for MVDF is guaranteed by construction:** gate 1 of the MVDF rule admits only monotone uncertainty curves. The informative parts of C5a are the fit R² and the comparison with the rest of the pool (rest-of-pool U fit R² q10 = 0.80).
- **Range exceedance is larger than the spec expected:** at λ = 1.25 a median 8.1% of active windows put the feature above its natural (active, E_select) 99th percentile. Counting inactive windows as zeros, it is 57%. E5b, which removes λ = 1.25, is the answer to this.
- **Curvature:** the quadratic term is significant (p < 0.05) for 78% of MVDF features (U); with six near-noiseless averaged points even negligible curvature is detectable. The linear R² ≥ 0.99 shows it is small.
- **The interventions drop the SAE reconstruction error** (existing code; see E0 DEFINITIONS §6.3), including E5c, for comparability with U_j.
- **C5b is a boundary failure:** the CI rule is strict (> 1), and the failing cell's lower bound is exactly 1.000. The enrichment-CI helper was made order-invariant during the run; before that fix the same data gave C5b PASS. The fix is a correctness fix, not a threshold change, and the post-fix verdict is the one reported.

## For the paper

| ETTh (20 seeds) | MVDF | rest of pool |
|---|---|---|
| linear-fit R² of U_j, median [q10] | 0.999 [0.989] | 0.995 [0.805] |
| natural-range (λ ≤ 1) reselection, Jaccard median [min] | 0.85 [0.50] | — |
| interchange sign agreement with U_j | 81% of features; ≥ 70% on 7/10 (ETTh1), 9/10 (ETTh2) seeds | — |

> "Dose-response curves of the selected features are near-linear (median R² 0.999) and monotone; restricting the intervention to scaling down (λ ≤ 1) reselects largely the same features (median Jaccard 0.85). Setting a feature to values it naturally takes on other inputs moves the difficulty score in the direction predicted by its dose-response slope for 81% of selected features (≥ 70% on 16 of 20 ETTh seeds)."

---

## Detail (auto-generated)

## Reproduction check (reported first, as required)

| tag | pool_matches_stored | mvdf_matches | max_abs_dP | max_abs_dU | max_rel_dU | n_pool | n_wsel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1_seed0 | True | True | 8.51e-09 | 9.62e-08 | 2.25e-03 | 120 | 1872 |
| ETTh1_seed1 | True | True | 7.66e-09 | 9.40e-08 | 1.79e-03 | 120 | 1872 |
| ETTh1_seed2 | True | True | 7.66e-09 | 6.46e-08 | 2.22e-03 | 120 | 1872 |
| ETTh1_seed3 | True | True | 6.39e-09 | 7.90e-08 | 7.39e-02 | 120 | 1872 |
| ETTh1_seed4 | True | True | 8.51e-09 | 8.50e-08 | 2.29e-03 | 120 | 1872 |
| ETTh1_seed5 | True | True | 8.51e-09 | 8.46e-08 | 2.32e-03 | 120 | 1872 |
| ETTh1_seed6 | True | True | 6.81e-09 | 8.96e-08 | 2.47e-03 | 120 | 1872 |
| ETTh1_seed7 | True | True | 8.51e-09 | 7.22e-08 | 1.45e-03 | 120 | 1872 |
| ETTh1_seed8 | True | True | 1.11e-08 | 9.46e-08 | 1.13e-02 | 120 | 1872 |
| ETTh1_seed9 | True | True | 7.66e-09 | 7.56e-08 | 2.27e-03 | 120 | 1872 |
| ETTh2_seed0 | True | True | 1.87e-08 | 1.37e-07 | 3.10e-03 | 120 | 1872 |
| ETTh2_seed1 | True | True | 2.38e-08 | 1.36e-07 | 4.50e-03 | 120 | 1872 |
| ETTh2_seed2 | True | True | 1.70e-08 | 1.35e-07 | 9.43e-03 | 120 | 1872 |
| ETTh2_seed3 | True | True | 1.70e-08 | 1.57e-07 | 2.00e-01 | 120 | 1872 |
| ETTh2_seed4 | True | True | 1.87e-08 | 1.19e-07 | 1.71e-02 | 120 | 1872 |
| ETTh2_seed5 | True | True | 1.87e-08 | 1.99e-07 | 2.78e-02 | 120 | 1872 |
| ETTh2_seed6 | True | True | 2.55e-08 | 1.11e-07 | 3.79e-03 | 120 | 1872 |
| ETTh2_seed7 | True | True | 1.36e-08 | 1.29e-07 | 1.40e-02 | 120 | 1872 |
| ETTh2_seed8 | True | True | 1.87e-08 | 1.90e-07 | 1.44e-03 | 120 | 1872 |
| ETTh2_seed9 | True | True | 2.04e-08 | 2.17e-07 | 7.58e-03 | 120 | 1872 |
| electricity_seed0 | True | True | 9.37e-09 | 1.30e-07 | 7.02e-04 | 120 | 2048 |
| electricity_seed1 | True | True | 1.19e-08 | 1.31e-07 | 7.99e-04 | 120 | 2048 |
| electricity_seed2 | True | True | 1.36e-08 | 1.33e-07 | 5.33e-04 | 120 | 2048 |
| electricity_seed3 | True | True | 1.28e-08 | 1.54e-07 | 8.44e-04 | 120 | 2048 |
| electricity_seed4 | True | True | 1.02e-08 | 1.53e-07 | 4.97e-03 | 120 | 2048 |
| traffic_seed0 | True | True | 1.02e-08 | 1.53e-07 | 9.69e-04 | 120 | 1075 |
| traffic_seed1 | True | True | 1.11e-08 | 1.66e-07 | 8.99e-04 | 120 | 1075 |
| traffic_seed2 | True | True | 1.11e-08 | 2.40e-07 | 2.44e-03 | 120 | 1075 |
| traffic_seed3 | True | True | 1.70e-08 | 2.08e-07 | 9.23e-04 | 120 | 1075 |
| traffic_seed4 | True | True | 1.19e-08 | 1.68e-07 | 4.14e-03 | 120 | 1075 |

All 30 cells: pool identical = True, MVDF identical = True. Stored P_j/U_j are reproduced to ≤ 2.4e-07 absolute (the relative maximum comes from |U_j| ≈ 0 features). Seeds ≥ 3 are compared against the candidate tables written by their own E2 run.

## E5a — diagnostics

| group (ETTh, pooled) | n | U fit R² q10/q25/med/q75 | |ρ(λ,U)| q10/q25/med/q75 | P fit R² median | |ρ(λ,P)| median | U quad p<0.05 | exceed q99 @1.25 (active) median | exceed q99 (incl. zeros) median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVDF | 240 | 0.989 / 0.996 / 0.999 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 0.998 | 1.000 | 0.78 | 0.081 | 0.571 |
| rest of pool | 2160 | 0.805 / 0.964 / 0.995 / 0.999 | 0.943 / 1.000 / 1.000 / 1.000 | 0.997 | 1.000 | 0.78 | 0.084 | 0.581 |

**C5a: PASS** — MVDF median linear-fit R² of U_j = 0.999 (≥ 0.8 required), median |Spearman(λ, D_unc)| = 1.000 (≥ 0.8 required), n = 240 MVDF features over 20 ETTh cells.

Per cell (ETTh):

| dataset | seed | MVDF med U R² | MVDF med |ρU| | MVDF min U R² | MVDF frac dS(0) CI excl. 0 | MVDF frac dS(1.25) CI excl. 0 | MVDF med exceed@1.25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 0.999 | 1.000 | 0.992 | 0.750 | 0.833 | 0.077 |
| ETTh1 | 1 | 0.999 | 1.000 | 0.994 | 0.417 | 0.500 | 0.103 |
| ETTh1 | 2 | 0.996 | 1.000 | 0.974 | 0.583 | 0.667 | 0.086 |
| ETTh1 | 3 | 0.998 | 1.000 | 0.901 | 0.417 | 0.417 | 0.089 |
| ETTh1 | 4 | 0.999 | 1.000 | 0.972 | 0.500 | 0.500 | 0.063 |
| ETTh1 | 5 | 1.000 | 1.000 | 0.994 | 0.833 | 0.833 | 0.083 |
| ETTh1 | 6 | 0.998 | 1.000 | 0.949 | 0.750 | 0.750 | 0.111 |
| ETTh1 | 7 | 0.999 | 1.000 | 0.946 | 0.583 | 0.667 | 0.123 |
| ETTh1 | 8 | 0.999 | 1.000 | 0.980 | 0.167 | 0.250 | 0.066 |
| ETTh1 | 9 | 0.999 | 1.000 | 0.912 | 0.500 | 0.667 | 0.087 |
| ETTh2 | 0 | 0.999 | 1.000 | 0.969 | 0.750 | 0.833 | 0.099 |
| ETTh2 | 1 | 0.998 | 1.000 | 0.924 | 0.583 | 0.750 | 0.073 |
| ETTh2 | 2 | 0.997 | 1.000 | 0.891 | 0.333 | 0.333 | 0.071 |
| ETTh2 | 3 | 1.000 | 1.000 | 0.993 | 0.333 | 0.417 | 0.077 |
| ETTh2 | 4 | 1.000 | 1.000 | 0.997 | 0.833 | 0.917 | 0.065 |
| ETTh2 | 5 | 1.000 | 1.000 | 0.985 | 0.917 | 0.917 | 0.082 |
| ETTh2 | 6 | 0.999 | 1.000 | 0.968 | 0.667 | 0.833 | 0.074 |
| ETTh2 | 7 | 0.996 | 1.000 | 0.951 | 0.583 | 0.667 | 0.084 |
| ETTh2 | 8 | 0.999 | 1.000 | 0.989 | 0.583 | 0.667 | 0.129 |
| ETTh2 | 9 | 0.999 | 1.000 | 0.939 | 0.833 | 0.917 | 0.061 |

Supplementary (electricity/traffic, 10 cells): MVDF median U fit R² 0.998, median |ρ(λ,U)| 1.000.

Histogram: `E5a_hist.png`.

## E5b — natural range (λ ≤ 1 only)

| dataset | seed | jaccard_vs_full | rho_U5_vs_U | full enr [CI] | full_enr_ci_excludes_1 | natural enr [CI] | nat_enr_ci_excludes_1 | nat_enr_p_hypergeom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 0.846 | 0.997 | 2.50 [1.11,4.71] | True | 2.50 [1.07,4.71] | True | 0.014 |
| ETTh1 | 1 | 1.000 | 0.998 | 2.22 [0.94,4.00] | False | 2.40 [1.00,4.44] | False | 0.018 |
| ETTh1 | 2 | 0.500 | 0.998 | 1.25 [0.00,2.94] | False | 1.60 [0.38,3.50] | False | 0.219 |
| ETTh1 | 3 | 0.714 | 0.998 | 1.48 [0.37,3.08] | False | 1.43 [0.36,2.86] | False | 0.294 |
| ETTh1 | 4 | 0.846 | 0.998 | 2.00 [0.69,3.81] | False | 2.31 [0.97,4.12] | False | 0.022 |
| ETTh1 | 5 | 0.846 | 0.998 | 2.17 [0.71,4.21] | False | 2.61 [1.00,5.00] | False | 0.011 |
| ETTh1 | 6 | 1.000 | 0.998 | 1.79 [0.65,3.33] | False | 1.67 [0.61,3.08] | False | 0.146 |
| ETTh1 | 7 | 0.846 | 0.998 | 2.00 [0.67,3.89] | False | 2.31 [1.00,4.29] | False | 0.022 |
| ETTh1 | 8 | 1.000 | 0.994 | 2.08 [0.77,4.09] | False | 2.08 [0.77,4.09] | False | 0.062 |
| ETTh1 | 9 | 0.846 | 0.993 | 2.22 [0.50,4.71] | False | 2.63 [0.87,5.33] | False | 0.023 |
| ETTh2 | 0 | 1.000 | 0.998 | 2.38 [0.77,4.62] | False | 1.90 [0.42,4.12] | False | 0.133 |
| ETTh2 | 1 | 0.846 | 0.999 | 2.17 [0.77,4.38] | False | 2.17 [0.74,4.29] | False | 0.052 |
| ETTh2 | 2 | 1.000 | 0.998 | 2.08 [0.74,4.21] | False | 2.08 [0.74,4.21] | False | 0.062 |
| ETTh2 | 3 | 1.000 | 0.999 | 3.68 [1.74,6.67] | True | 3.68 [1.74,6.67] | True | 0.000 |
| ETTh2 | 4 | 0.846 | 1.000 | 1.74 [0.43,3.64] | False | 2.27 [0.80,4.44] | False | 0.043 |
| ETTh2 | 5 | 0.846 | 0.965 | 2.61 [1.11,5.00] | True | 2.50 [1.00,4.67] | False | 0.014 |
| ETTh2 | 6 | 0.846 | 0.999 | 3.16 [1.30,6.00] | True | 3.33 [1.43,6.67] | True | 0.003 |
| ETTh2 | 7 | 0.846 | 0.999 | 1.82 [0.43,3.85] | False | 1.82 [0.43,3.89] | False | 0.152 |
| ETTh2 | 8 | 0.846 | 0.999 | 2.40 [1.07,4.50] | True | 2.92 [1.43,5.26] | True | 0.002 |
| ETTh2 | 9 | 0.714 | 1.000 | 1.74 [0.43,3.75] | False | 1.30 [0.00,3.00] | False | 0.414 |
| electricity | 0 | 0.714 | 0.998 | 3.85 [1.25,8.75] | True | 4.62 [1.82,10.00] | True | 0.000 |
| electricity | 1 | 1.000 | 0.999 | 4.17 [1.54,10.00] | True | 3.85 [1.43,8.75] | True | 0.003 |
| electricity | 2 | 0.714 | 0.999 | 2.00 [0.00,5.00] | False | 2.00 [0.00,5.00] | False | 0.173 |
| electricity | 3 | 1.000 | 0.998 | 3.57 [1.33,8.00] | True | 3.57 [1.33,8.00] | True | 0.005 |
| electricity | 4 | 0.846 | 0.999 | 4.00 [1.00,10.00] | False | 4.17 [1.43,10.00] | True | 0.002 |
| traffic | 0 | 0.714 | 0.995 | 2.31 [0.53,6.25] | False | 3.33 [0.83,8.58] | False | 0.019 |
| traffic | 1 | 0.600 | 0.997 | 3.33 [0.77,7.78] | False | 3.08 [0.77,7.78] | False | 0.026 |
| traffic | 2 | 0.600 | 0.997 | 4.17 [1.33,10.00] | True | 3.57 [1.18,8.00] | True | 0.005 |
| traffic | 3 | 0.714 | 0.953 | 3.33 [0.71,8.75] | False | 5.00 [2.00,11.43] | True | 0.000 |
| traffic | 4 | 0.846 | 0.995 | 1.43 [0.00,4.29] | False | 2.14 [0.00,5.56] | False | 0.147 |

**C5b: FAIL** — full-λ CI excludes 1 on 5/20 ETTh cells, natural-range on 4/20; every full-λ passing cell also passes in natural range: False. Median Jaccard (natural vs full MVDF) = 0.85 [0.50, 1.00] (≥ 0.5 required).

## E5c — interchange (resample) intervention

| dataset | seed | n_feat | sign_agree | n_eval_med | dS_med | dS CI excl 0 (frac) | implied/observed dS (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 12 | 0.583 | 44.500 | 0.000 | 0.917 | 0.732 |
| ETTh1 | 1 | 12 | 0.583 | 52.500 | 0.000 | 1.000 | 0.784 |
| ETTh1 | 2 | 12 | 0.833 | 85.500 | 0.000 | 0.750 | 1.607 |
| ETTh1 | 3 | 12 | 0.750 | 53.000 | 0.000 | 0.917 | 0.923 |
| ETTh1 | 4 | 12 | 0.833 | 67.500 | 0.000 | 0.917 | 1.723 |
| ETTh1 | 5 | 12 | 0.833 | 60.000 | 0.000 | 0.917 | 2.190 |
| ETTh1 | 6 | 12 | 0.833 | 47.000 | 0.000 | 0.917 | 2.620 |
| ETTh1 | 7 | 12 | 0.667 | 58.000 | 0.000 | 1.000 | 1.204 |
| ETTh1 | 8 | 12 | 1.000 | 48.000 | 0.000 | 1.000 | 1.177 |
| ETTh1 | 9 | 12 | 0.750 | 49.500 | 0.001 | 0.917 | 1.463 |
| ETTh2 | 0 | 12 | 0.750 | 53.000 | 0.001 | 1.000 | 1.402 |
| ETTh2 | 1 | 12 | 0.917 | 95.500 | 0.001 | 1.000 | 1.599 |
| ETTh2 | 2 | 12 | 0.917 | 83.500 | 0.001 | 1.000 | 1.999 |
| ETTh2 | 3 | 12 | 0.833 | 68.500 | 0.001 | 0.917 | 1.215 |
| ETTh2 | 4 | 12 | 0.917 | 87.500 | 0.001 | 0.833 | 1.133 |
| ETTh2 | 5 | 12 | 0.750 | 42.000 | 0.001 | 1.000 | 1.420 |
| ETTh2 | 6 | 12 | 0.917 | 100.500 | 0.001 | 0.833 | 1.903 |
| ETTh2 | 7 | 12 | 0.917 | 117.500 | 0.001 | 0.833 | 1.477 |
| ETTh2 | 8 | 12 | 0.667 | 51.000 | 0.001 | 1.000 | 0.711 |
| ETTh2 | 9 | 12 | 0.917 | 161.000 | 0.001 | 0.833 | 1.699 |
| electricity | 0 | 12 | 1.000 | 78.500 | 0.002 | 1.000 | 1.306 |
| electricity | 1 | 12 | 1.000 | 100.000 | 0.002 | 1.000 | 1.841 |
| electricity | 2 | 12 | 1.000 | 58.000 | 0.002 | 1.000 | 1.464 |
| electricity | 3 | 12 | 0.917 | 87.500 | 0.003 | 1.000 | 1.301 |
| electricity | 4 | 12 | 1.000 | 76.500 | 0.002 | 1.000 | 1.815 |
| traffic | 0 | 12 | 0.833 | 58.000 | 0.006 | 1.000 | 2.834 |
| traffic | 1 | 12 | 1.000 | 93.000 | 0.018 | 1.000 | 2.573 |
| traffic | 2 | 12 | 0.750 | 45.500 | 0.005 | 1.000 | 2.092 |
| traffic | 3 | 12 | 0.667 | 49.500 | 0.008 | 1.000 | 1.623 |
| traffic | 4 | 12 | 1.000 | 84.500 | 0.015 | 1.000 | 3.650 |

**C5c: FAIL** — cells with ≥ 70% sign agreement: ETTh1 7/10 (need 8); ETTh2 9/10 (need 8).

