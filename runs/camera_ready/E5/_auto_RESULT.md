# E5 — Dose-response diagnostics and natural-range interventions

## Provenance

- Git: `763d853` (+uncommitted changes)
- Command: `python -m camera_ready.e5_dose a|c --datasets ETTh1 ETTh2 electricity traffic --seeds <all>; python -m camera_ready.report_e5`
- Seeds: {'ETTh1': [0, 1, 2], 'ETTh2': [0, 1, 2], 'electricity': [0, 1, 2], 'traffic': [0, 1, 2]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684); electricity: Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical; traffic: Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`; electricity `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570`; traffic `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: 2 s
- Generated: 2026-11-14 05:17:57 +0530

## Pre-registration (summary; full text in PREREG.md) — Dose-response diagnostics + natural-range interventions: pre-registration

Committed before the first E5 run.


## Reproduction check (reported first, as required)

| tag | pool_matches_stored | mvdf_matches | max_abs_dP | max_abs_dU | max_rel_dU | n_pool | n_wsel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1_seed0 | True | True | 8.51e-09 | 9.62e-08 | 2.25e-03 | 120 | 1872 |
| ETTh1_seed1 | True | True | 7.66e-09 | 9.40e-08 | 1.79e-03 | 120 | 1872 |
| ETTh1_seed2 | True | True | 7.66e-09 | 6.46e-08 | 2.22e-03 | 120 | 1872 |
| ETTh2_seed0 | True | True | 1.87e-08 | 1.37e-07 | 3.10e-03 | 120 | 1872 |
| ETTh2_seed1 | True | True | 2.38e-08 | 1.36e-07 | 4.50e-03 | 120 | 1872 |
| ETTh2_seed2 | True | True | 1.70e-08 | 1.35e-07 | 9.43e-03 | 120 | 1872 |
| electricity_seed0 | True | True | 9.37e-09 | 1.30e-07 | 7.02e-04 | 120 | 2048 |
| electricity_seed1 | True | True | 1.19e-08 | 1.31e-07 | 7.99e-04 | 120 | 2048 |
| electricity_seed2 | True | True | 1.36e-08 | 1.33e-07 | 5.33e-04 | 120 | 2048 |
| traffic_seed0 | True | True | 1.02e-08 | 1.53e-07 | 9.69e-04 | 120 | 1075 |
| traffic_seed1 | True | True | 1.11e-08 | 1.66e-07 | 8.99e-04 | 120 | 1075 |
| traffic_seed2 | True | True | 1.11e-08 | 2.40e-07 | 2.44e-03 | 120 | 1075 |

All 12 cells: pool identical = True, MVDF identical = True. Stored P_j/U_j are reproduced to ≤ 2.4e-07 absolute (the relative maximum comes from |U_j| ≈ 0 features). Seeds ≥ 3 are compared against the candidate tables written by their own E2 run.

## E5a — diagnostics

| group (ETTh, pooled) | n | U fit R² q10/q25/med/q75 | |ρ(λ,U)| q10/q25/med/q75 | P fit R² median | |ρ(λ,P)| median | U quad p<0.05 | exceed q99 @1.25 (active) median | exceed q99 (incl. zeros) median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVDF | 72 | 0.989 / 0.995 / 0.999 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 0.998 | 1.000 | 0.81 | 0.083 | 0.555 |
| rest of pool | 648 | 0.721 / 0.952 / 0.993 / 0.999 | 0.771 / 1.000 / 1.000 / 1.000 | 0.997 | 1.000 | 0.77 | 0.089 | 0.612 |

**C5a: PASS** — MVDF median linear-fit R² of U_j = 0.999 (≥ 0.8 required), median |Spearman(λ, D_unc)| = 1.000 (≥ 0.8 required), n = 72 MVDF features over 6 ETTh cells.

Per cell (ETTh):

| dataset | seed | MVDF med U R² | MVDF med |ρU| | MVDF min U R² | MVDF frac dS(0) CI excl. 0 | MVDF frac dS(1.25) CI excl. 0 | MVDF med exceed@1.25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 0.999 | 1.000 | 0.992 | 0.750 | 0.833 | 0.077 |
| ETTh1 | 1 | 0.999 | 1.000 | 0.994 | 0.417 | 0.500 | 0.103 |
| ETTh1 | 2 | 0.996 | 1.000 | 0.974 | 0.583 | 0.667 | 0.086 |
| ETTh2 | 0 | 0.999 | 1.000 | 0.969 | 0.750 | 0.833 | 0.099 |
| ETTh2 | 1 | 0.998 | 1.000 | 0.924 | 0.583 | 0.750 | 0.073 |
| ETTh2 | 2 | 0.997 | 1.000 | 0.891 | 0.333 | 0.333 | 0.071 |

Supplementary (electricity/traffic, 6 cells): MVDF median U fit R² 0.998, median |ρ(λ,U)| 1.000.

Histogram: `E5a_hist.png`.

## E5b — natural range (λ ≤ 1 only)

| dataset | seed | jaccard_vs_full | rho_U5_vs_U | full enr [CI] | full_enr_ci_excludes_1 | natural enr [CI] | nat_enr_ci_excludes_1 | nat_enr_p_hypergeom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 0.846 | 0.997 | 2.50 [1.11,4.71] | True | 2.50 [1.05,4.58] | True | 0.014 |
| ETTh1 | 1 | 1.000 | 0.998 | 2.22 [0.94,4.00] | False | 2.40 [0.97,4.29] | False | 0.018 |
| ETTh1 | 2 | 0.500 | 0.998 | 1.25 [0.00,2.94] | False | 1.60 [0.38,3.33] | False | 0.219 |
| ETTh2 | 0 | 1.000 | 0.998 | 2.38 [0.77,4.62] | False | 1.90 [0.48,4.00] | False | 0.133 |
| ETTh2 | 1 | 0.846 | 0.999 | 2.17 [0.77,4.38] | False | 2.17 [0.74,4.29] | False | 0.052 |
| ETTh2 | 2 | 1.000 | 0.998 | 2.08 [0.74,4.21] | False | 2.08 [0.77,4.00] | False | 0.062 |
| electricity | 0 | 0.714 | 0.998 | 3.85 [1.25,8.75] | True | 4.62 [1.88,10.00] | True | 0.000 |
| electricity | 1 | 1.000 | 0.999 | 4.17 [1.54,10.00] | True | 3.85 [1.25,8.75] | True | 0.003 |
| electricity | 2 | 0.714 | 0.999 | 2.00 [0.00,5.00] | False | 2.00 [0.00,5.00] | False | 0.173 |
| traffic | 0 | 0.714 | 0.995 | 2.31 [0.53,6.25] | False | 3.33 [0.83,8.57] | False | 0.019 |
| traffic | 1 | 0.600 | 0.997 | 3.33 [0.77,7.78] | False | 3.08 [0.71,7.50] | False | 0.026 |
| traffic | 2 | 0.600 | 0.997 | 4.17 [1.33,10.00] | True | 3.57 [1.25,7.78] | True | 0.005 |

**C5b: PASS** — full-λ CI excludes 1 on 1/6 ETTh cells, natural-range on 1/6; every full-λ passing cell also passes in natural range: True. Median Jaccard (natural vs full MVDF) = 0.92 [0.50, 1.00] (≥ 0.5 required).

## E5c — interchange (resample) intervention

| dataset | seed | n_feat | sign_agree | n_eval_med | dS_med | dS CI excl 0 (frac) | implied/observed dS (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 12 | 0.583 | 44.500 | 0.000 | 0.917 | 0.732 |
| ETTh1 | 1 | 12 | 0.583 | 52.500 | 0.000 | 1.000 | 0.784 |
| ETTh1 | 2 | 12 | 0.833 | 85.500 | 0.000 | 0.750 | 1.607 |
| ETTh2 | 0 | 12 | 0.750 | 53.000 | 0.001 | 1.000 | 1.402 |
| ETTh2 | 1 | 12 | 0.917 | 95.500 | 0.001 | 1.000 | 1.599 |
| ETTh2 | 2 | 12 | 0.917 | 83.500 | 0.001 | 1.000 | 1.999 |
| electricity | 0 | 12 | 1.000 | 78.500 | 0.002 | 1.000 | 1.306 |
| electricity | 1 | 12 | 1.000 | 100.000 | 0.002 | 1.000 | 1.841 |
| electricity | 2 | 12 | 1.000 | 58.000 | 0.002 | 1.000 | 1.464 |
| traffic | 0 | 12 | 0.833 | 58.000 | 0.006 | 1.000 | 2.834 |
| traffic | 1 | 12 | 1.000 | 93.000 | 0.018 | 1.000 | 2.573 |
| traffic | 2 | 12 | 0.750 | 45.500 | 0.005 | 1.000 | 2.092 |

**C5c: FAIL** — cells with ≥ 70% sign agreement: ETTh1 1/3 (need 3); ETTh2 3/3 (need 3).

