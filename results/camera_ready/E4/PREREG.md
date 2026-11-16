# E4 — Robustness of causal specificity R_j: pre-registration

Committed before the first E4 run.

## Inputs (no retraining)
Per-feature P_j, U_j and 6-point uncertainty curves for every pool feature, from E5a's recomputation (which must first reproduce the stored candidate tables and MVDF sets; see E5 PREREG). Cells: ETTh1, ETTh2 × all available seeds (primary, pass counts); electricity, traffic × available seeds (supplementary); `v2_runs/v2_phase1/<ds>_pre_head/candidates.csv` + `dose_curves.json` (single Kaggle seed, 8 datasets) for selection/enrichment only (no models, so no Phase 3 R²).

## Definitions of R_j = U_j − Ê[U_j | P_j]
1. Isotonic (current; best of increasing/decreasing by SSE) — reference
2. Binned mean: 10 equal-count bins of P_j (by rank), Ê = bin mean of U_j
3. Smoothing spline: `scipy.interpolate.make_smoothing_spline` with λ chosen by GCV on the pool (ties in P averaged)
4. Linear: OLS of U on P
5. Rank residual: rank(U) − isotonic (best direction) fit of rank(U) on rank(P)
6. No residualisation: R_j := U_j

Selection for every definition: the unchanged gate 1 (|U| ≥ 0, monotone uncertainty curve, finite R) then top-12 by R. Type table is the unchanged median split on |P|, |U| (independent of R).

## Metrics per (dataset, seed, definition)
Spearman ρ(R_def, R_iso) over the pool; Jaccard(MVDF_def, MVDF_iso); type-B enrichment with the E2 bootstrap CI and hypergeometric p; Phase-3 best held-out R² (marginal + active-window; original Phase-3 definitions; moving-block CI of the best-on-eval R², block 96) with that definition's MVDF set.

## Pre-registered criteria
- **C4.1 Selection robust:** median Jaccard vs isotonic ≥ 0.5 across seeds for definitions 2, 3, 5.
- **C4.2 Enrichment robust:** type-B enrichment > 1 with CI excluding 1 on ≥ the same number of (dataset, seed) cells as the isotonic definition, for definitions 2, 3, 5.
- **C4.3 Null robust:** best held-out R² ≤ 0 (or CI includes 0) on ≥ 4/5 seeds for every definition. *(Evaluated per dataset and per battery; N-seed threshold ceil(0.8 N).)*
- Definitions 4 and 6 are reported as contrasts. They are not part of the pass/fail.
