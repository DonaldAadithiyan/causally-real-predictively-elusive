# E5 — Dose-response diagnostics + natural-range interventions: pre-registration

Committed before the first E5 run.

## Common setup
Cells: ETTh1, ETTh2 × all seeds with artifacts (0–2, then 3–9 from E2); electricity and traffic × available seeds reported as supplementary (not part of the pass counts, which the spec defines on ETTh). Models: saved weights. Pool and dose windows: regenerated with the exact rng consumption of `frozen_selection` (screen on E_select → `rng.choice` pool of 120 → `rng.choice` window cap 2048). Dose responses: `p126.selection.fast_dose` (matches `sweep_pool` to ~1e-8; verified again here by reproducing stored P_j/U_j). Profile = E_select profile. Interventions follow the existing code: the patched token is the SAE reconstruction with z_j changed; the reconstruction-error term is **not** added back (the spec's description of the existing code is inaccurate on this point; see E0/DEFINITIONS.md §6.3).

**Reproduction gate (reported first):** recomputed P_j, U_j must match the stored candidate tables (max |diff| reported) and `select_mvdf_v2` on the recomputed dose must return the stored MVDF set exactly.

## E5a diagnostics (per pool feature j)
- mean Δloss(λ) and mean ΔS(λ) relative to λ = 1 at each λ, with moving-block bootstrap 95% CI (blocks of 96 consecutive E_select time positions; inactive windows in a resampled block are dropped, as in round-2 WP3; 2000 resamples, seed 0);
- R² of the 6-point linear fit defining P_j and U_j;
- Spearman ρ(λ, D(λ)) over the 6 points, for loss and for S;
- quadratic fit `D = a + bλ + cλ²`, two-sided t-test p-value of c (3 residual df), for loss and for S;
- fraction of active windows with `1.25·a_j > q99`, q99 = 99th percentile of a_j over **active** E_select windows (primary); also with q99 over all E_select windows including zeros (secondary).
- **C5a:** among MVDF features (pooled over ETTh cells), median linear-fit R² of U_j ≥ 0.8 **and** median |Spearman monotonicity| for U_j ≥ 0.8. Distribution reported as histogram + quantiles for MVDF vs rest of pool.

## E5b natural range
P_j, U_j re-estimated as OLS slopes over λ ∈ {0, .25, .5, .75, 1} only (same mean curves, λ = 1.25 point dropped); gate 1 monotonicity checked on the 5-point curve; then the unchanged `feature_type_table`, `select_mvdf_v2` (k = 12, isotonic) and `enrichment`.
Enrichment CI (same rule as E2/E4): 95% percentile CI from 2000 bootstrap resamples of (the 12 selected features' B-membership, the pool's B-membership); one-sided hypergeometric p also reported.
- **C5b:** type-B enrichment > 1 with CI excluding 1 on the same cells as the full-λ protocol; Jaccard of the MVDF set vs the full-λ set ≥ 0.5 (median).

## E5c interchange (stretch)
For each MVDF feature j: evaluation windows = E_eval_ID windows where a_j is active; donor pools = E_select **active** windows whose a_j lies in the top or bottom quintile of a_j's active E_select distribution. Each evaluation window gets one donor value drawn uniformly (rng seed 0) from each pool; z_j is set to that value; decode; head; S (E_select profile) and horizon MSE. Effect = mean over windows of S(top donor) − S(bottom donor) (and the same for loss); also each vs the λ = 1 reconstruction. Block-bootstrap CI over evaluation windows (block 96).
- **C5c:** sign agreement between the interchange effect on S and sgn(U_j) for ≥ 70% of MVDF features, on ≥ 4/5 seeds (≥ ceil(0.8 N) of N).
