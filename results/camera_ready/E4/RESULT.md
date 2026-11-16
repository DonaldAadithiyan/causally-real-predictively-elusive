# E4 — Robustness of causal specificity R_j

## Provenance

- Git: `93fbb0c` (+uncommitted changes)
- Command: `python -m camera_ready.e4_rj; python -m camera_ready.report_e4`
- Seeds: {'ETTh1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ETTh2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'electricity': [0, 1, 2, 3, 4], 'traffic': [0, 1, 2, 3, 4]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684); electricity: Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical; traffic: Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`; electricity `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570`; traffic `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: 0 s
- Generated: 2026-11-14 07:10:06 +0530

## Pre-registered criteria

- **C4.1 Selection robust:** median Jaccard vs isotonic ≥ 0.5 across seeds for definitions 2, 3, 5.
- **C4.2 Enrichment robust:** type-B enrichment > 1 with CI excluding 1 on ≥ the same number of (dataset, seed) cells as the isotonic definition, for definitions 2, 3, 5.
- **C4.3 Null robust:** best held-out R² ≤ 0 (or CI includes 0) on ≥ 4/5 seeds for every definition.
- Definitions 4 and 6 are contrasts, not pass/fail.

## Result (ETTh1 + ETTh2, 20 cells)

| definition | ρ(R, R_iso) median | Jaccard median [min,max] | enrichment median [min,max] | enr. CI > 1 (cells) | ETTh1 R² null marg/active | ETTh2 R² null marg/active |
| --- | --- | --- | --- | --- | --- | --- |
| 1 isotonic (reference) | 1.00 | 1.00 [1.00,1.00] | 2.13 [1.25,3.68] | 5/20 | 10/10, 10/10 | 10/10, 10/10 |
| 2 binned mean | 0.85 | 0.50 [0.14,0.85] | 1.39 [0.00,2.63] | 0/20 | 10/10, 10/10 | 10/10, 10/10 |
| 3 smoothing spline (GCV) | 0.95 | 0.71 [0.41,0.85] | 2.25 [1.48,4.21] | 7/20 | 10/10, 10/10 | 10/10, 10/10 |
| 4 linear (contrast) | 0.88 | 0.60 [0.20,1.00] | 2.49 [0.40,3.68] | 10/20 | 10/10, 10/10 | 10/10, 10/10 |
| 5 rank residual | 0.96 | 0.60 [0.26,1.00] | 3.00 [2.40,3.68] | 17/20 | 10/10, 10/10 | 10/10, 10/10 |
| 6 none, R=U (contrast) | 0.83 | 0.55 [0.26,1.00] | 1.46 [0.87,2.63] | 1/20 | 10/10, 10/10 | 10/10, 10/10 |

- **C4.1 PASS** — median Jaccard: binned 0.50, spline 0.71, rank 0.60.
- **C4.2 FAIL** — cells with enrichment CI > 1: isotonic 5, binned 0, spline 7, rank 17 (of 20).
- **C4.3 PASS** — isotonic: ok, binned: ok, spline: ok, linear: ok, rank: ok, none: ok.


## Plain-English summary
How the selection responds to the residualisation choice: the MVDF set itself is moderately stable (median Jaccard vs isotonic 0.50 binned, 0.71 spline, 0.60 rank; C4.1 PASS at the bar). The L1 null is completely insensitive: every definition's MVDF set gives best held-out R² < 0 on 10/10 seeds of both ETTh datasets, marginal and active (C4.3 PASS). Type-B enrichment *with a confidence interval* is fragile. The isotonic reference clears CI > 1 on only 5/20 ETTh cells; binned-mean residualisation never does (0/20) and its median enrichment drops to 1.39. So C4.2 FAILS on the binned definition, even though spline (7/20) and rank (17/20) exceed the reference. The paper should state H1c as a population-level enrichment that depends on how E[U|P] is estimated, not as robust to every reasonable residualiser.

## Anomalies / interpretation notes
- **Rank residual enrichment is near-mechanical.** High rank(U) relative to an isotonic fit on rank(P) is close to the definition of the type-B quadrant (low |P|, high |U|), so its 17/20 should not be read as independent confirmation.
- **No residualisation (R = U)** gives enrichment 0 on 5/5 electricity and 5/5 traffic seeds: selecting by U alone picks high-|P| (type C) features. Residualisation is what produces enrichment on those datasets.
- **The smoothing spline** was fitted on z-scored P and U (P_j ~ 1e-5 made scipy's GCV ill-conditioned). GCV re-optimises λ, so this is numerical conditioning, not tuning.
- **Reproduction:** the isotonic definition reproduces the stored MVDF set on all 20 local ETTh cells, 10 electricity/traffic cells and 8/8 v2 Kaggle cells.
- **Bug fixed during the run:** the enrichment bootstrap originally depended on the order features were listed in; it was made order-invariant before the final tables. This changed the isotonic CI>1 count from 4 to 5/20; no verdict changed.

## For the paper

| R_j definition | Jaccard vs isotonic, median [min, max] | enrichment CI > 1 (ETTh cells) | R² null holds (ETTh1/ETTh2, marg/active) |
|---|---|---|---|
| isotonic (paper) | 1.00 | 5/20 | 10/10 · 10/10 |
| binned mean | 0.50 [0.14, 0.85] | 0/20 | 10/10 · 10/10 |
| smoothing spline (GCV) | 0.71 [0.41, 0.85] | 7/20 | 10/10 · 10/10 |
| rank residual | 0.60 [0.26, 1.00] | 17/20 | 10/10 · 10/10 |
| linear (contrast) | 0.60 [0.20, 1.00] | 10/20 | 10/10 · 10/10 |
| none, R = U (contrast) | 0.55 [0.26, 1.00] | 1/20 | 10/10 · 10/10 |

> "Replacing the isotonic residualisation of U on P by binned-mean, smoothing-spline or rank residualisation changes 30–50% of the selected features (median Jaccard 0.50–0.71) and never restores natural predictive power (R² < 0 on all 20 ETTh seeds under every definition). Type-B enrichment is reproduced by the spline and rank residualisers but not by a 10-bin mean (median 1.39×, no seed significant)."

---

## Supplementary: electricity, traffic (local models)

| dataset | definition | J_med | enr_med | enr_ci_gt1 | marg_null | n |
| --- | --- | --- | --- | --- | --- | --- |
| electricity | binned | 0.600 | 2.857 | 0 | 5 | 5 |
| electricity | isotonic | 1.000 | 3.846 | 3 | 5 | 5 |
| electricity | linear | 0.600 | 2.667 | 0 | 5 | 5 |
| electricity | none | 0.263 | 0.000 | 0 | 5 | 5 |
| electricity | rank | 0.600 | 5.714 | 5 | 5 | 5 |
| electricity | spline | 0.600 | 3.333 | 2 | 5 | 5 |
| traffic | binned | 0.714 | 2.500 | 0 | 5 | 5 |
| traffic | isotonic | 1.000 | 3.333 | 1 | 5 | 5 |
| traffic | linear | 0.412 | 0.833 | 0 | 5 | 5 |
| traffic | none | 0.200 | 0.000 | 0 | 5 | 5 |
| traffic | rank | 0.500 | 5.000 | 5 | 5 | 5 |
| traffic | spline | 0.846 | 2.500 | 1 | 5 | 5 |

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
