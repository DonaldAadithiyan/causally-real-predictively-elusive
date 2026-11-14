# E4 — Robustness of causal specificity R_j

## Provenance

- Git: `8c8c140` (+uncommitted changes)
- Command: `python -m camera_ready.e4_rj; python -m camera_ready.report_e4`
- Seeds: {'ETTh1': [0, 1, 2], 'ETTh2': [0, 1, 2]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684)
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: 0 s
- Generated: 2026-11-14 05:22:20 +0530

## Pre-registered criteria

- **C4.1 Selection robust:** median Jaccard vs isotonic ≥ 0.5 across seeds for definitions 2, 3, 5.
- **C4.2 Enrichment robust:** type-B enrichment > 1 with CI excluding 1 on ≥ the same number of (dataset, seed) cells as the isotonic definition, for definitions 2, 3, 5.
- **C4.3 Null robust:** best held-out R² ≤ 0 (or CI includes 0) on ≥ 4/5 seeds for every definition.
- Definitions 4 and 6 are contrasts, not pass/fail.

## Result (ETTh1 + ETTh2, 6 cells)

| definition | ρ(R, R_iso) median | Jaccard median [min,max] | enrichment median [min,max] | enr. CI > 1 (cells) | ETTh1 R² null marg/active | ETTh2 R² null marg/active |
| --- | --- | --- | --- | --- | --- | --- |
| 1 isotonic (reference) | 1.00 | 1.00 [1.00,1.00] | 2.20 [1.25,2.50] | 1/6 | 3/3, 3/3 | 3/3, 3/3 |
| 2 binned mean | 0.82 | 0.37 [0.14,0.60] | 0.99 [0.00,1.90] | 0/6 | 3/3, 3/3 | 3/3, 3/3 |
| 3 smoothing spline (GCV) | 0.96 | 0.66 [0.41,0.85] | 1.97 [1.67,2.86] | 2/6 | 3/3, 3/3 | 3/3, 3/3 |
| 4 linear (contrast) | 0.79 | 0.55 [0.20,1.00] | 2.23 [0.83,3.04] | 2/6 | 3/3, 3/3 | 3/3, 3/3 |
| 5 rank residual | 0.96 | 0.60 [0.26,0.71] | 3.00 [2.50,3.33] | 6/6 | 3/3, 3/3 | 3/3, 3/3 |
| 6 none, R=U (contrast) | 0.85 | 0.51 [0.26,1.00] | 1.64 [1.25,2.50] | 1/6 | 3/3, 3/3 | 3/3, 3/3 |

- **C4.1 FAIL** — median Jaccard: binned 0.37, spline 0.66, rank 0.60.
- **C4.2 FAIL** — cells with enrichment CI > 1: isotonic 1, binned 0, spline 2, rank 6 (of 6).
- **C4.3 PASS** — isotonic: ok, binned: ok, spline: ok, linear: ok, rank: ok, none: ok.


## Supplementary: v2_runs Kaggle single-seed candidate tables (8 datasets)

Isotonic definition reproduces the stored MVDF set on 8/8 datasets.

Enrichment:

| dataset | binned | isotonic | linear | none | rank | spline |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.870 | 1.740 | 2.170 | 0.870 | 3.040 | 2.170 |
| ETTh2 | 1.360 | 1.360 | 1.360 | 0.910 | 2.270 | 1.360 |
| ETTm1 | 1.250 | 1.250 | 2.500 | 2.920 | 3.330 | 2.080 |
| ETTm2 | 0.770 | 1.540 | 1.540 | 1.540 | 3.080 | 1.920 |
| electricity | 2.670 | 4.000 | 1.330 | 0.000 | 5.330 | 1.330 |
| solar | 0.830 | 2.500 | 1.670 | 5.000 | 3.330 | 0.830 |
| traffic | 2.730 | 4.550 | 2.730 | 1.820 | 5.450 | 3.640 |
| weather | 0.870 | 1.740 | 1.740 | 1.740 | 2.170 | 2.170 |

Jaccard vs isotonic:

| dataset | binned | isotonic | linear | none | rank | spline |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.140 | 1.000 | 0.500 | 0.600 | 0.500 | 0.500 |
| ETTh2 | 0.710 | 1.000 | 0.710 | 0.410 | 0.410 | 1.000 |
| ETTm1 | 0.710 | 1.000 | 0.410 | 0.200 | 0.260 | 0.260 |
| ETTm2 | 0.260 | 1.000 | 0.330 | 0.500 | 0.410 | 0.600 |
| electricity | 0.500 | 1.000 | 0.500 | 0.260 | 0.500 | 0.500 |
| solar | 0.260 | 1.000 | 0.710 | 0.260 | 0.260 | 0.600 |
| traffic | 0.410 | 1.000 | 0.410 | 0.260 | 0.710 | 0.850 |
| weather | 0.600 | 1.000 | 0.850 | 0.600 | 0.600 | 0.410 |
