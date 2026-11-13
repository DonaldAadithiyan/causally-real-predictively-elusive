# E3 — Ex-ante difficulty baselines + gap recovery: pre-registration

Committed before the first E3 run (`git log -- runs/camera_ready/E3/PREREG.md` gives timestamp + hash).

## Scope
ETTh1, ETTh2 × all seeds with artifacts (0–2 now; 3–9 as E2 lands). Electricity and traffic are run by the same script as part of E1 (criteria C1.3 = C3.1–C3.5 per dataset).

## Fixed operational choices (where the spec leaves room)
1. **Target.** `S_t = max_h |y − ŷ|_{t,h} / profile_h` on the target channel, with `profile` from **E_select** (the dose-response profile; computed once, used for every window). The original Phase-3 definition (E_eval_ID-local profile, eval-mean R²) is also computed for the F0 row as a reproduction check against `runs/phase3_real/AGG_phase3.csv`; it is not used for criteria. Known: E_select's last 96 windows have targets that overlap the first ID origins (profile only).
2. **Predictability split.** Exactly Phase 3's: chronological 60/40 of E_eval_ID (fit = first `int(0.6 n)`, eval = rest). E_eval_shift: the same fitted models are scored on the whole shift block.
3. **R².** Primary = `1 − SS_res / Σ(y − ȳ_fit)²` (fit-split mean, as the spec states). The Phase-3-as-implemented eval-mean R² is also stored (`R2_evalmean`).
4. **Window origin** `t` = `end_idx[t]`; stride 1, so "lag d" = window index t − d. F1 uses only `S_{t'}` with `t' ≤ t − 96`. F1p uses `S^partial_{t'} = max_{h < t−t'} |e_{t',h}| / profile_h` (0-indexed h; steps whose target time ≤ t), `t' ∈ {t−1, t−24, t−48, t−72}`, plus `|e_{t',h=0}| / profile_0` for `t' = t−1 … t−24`. A unit test (`camera_ready/test_leakage.py`) overwrites every target value with timestamp > t and asserts each feature row t is unchanged; E3 refuses to run if it fails.
5. **F2** per channel over the 96-step input (as fed to the model, i.e. train-z-scored), then mean and max across channels: std, range, mean|Δx|, std(Δx), OLS slope, last − mean, lag-1 autocorrelation, spectral entropy (normalised periodogram, DC removed), and the RevIN scale `std + 1e-6` (the scale the model divides by). 18 features.
6. **F3** from ŷ (target channel): std over horizon, range, mean|Δŷ|, mean_h |ŷ_h − x_last|. 4 features.
7. **F4** PCA(12) of the 256-d pre-head target token; **F5** PCA(12) of the full 2048-d SAE code. PCA fit on the fit split only.
8. **Estimators** (settings fixed): OLS; isotonic (a) on the fit-split most-|corr| single column (existing rule) and (b) on the OLS prediction; GBM (200 trees, depth 3, lr 0.05, subsample 0.8 — existing); pinball MLP (2×64 GELU, 300 epochs, median, existing `train_quantile_mlp`, raw features as in Phase 3); **ridge** (standardised features, α ∈ logspace(−3, 3, 13) chosen by 5-fold `TimeSeriesSplit` inside the fit split); **random forest** (500 trees, min_samples_leaf 20, random_state = seed). Per-single-feature isotonic fits are logged in a side table and are **not** eligible as "best".
9. **Which estimator counts for a criterion.** For C3.1/C3.2/C3.5 the estimator is the one with the best **fit-split time-series CV** MSE (5-fold `TimeSeriesSplit` inside the fit split), never the best on eval. The best-on-eval R² is reported as descriptive only. For C3.3, F6's CV-selected estimator is used for both F6 and F7 (paired bootstrap of ΔR² over the same blocks).
10. **Gap recovery.** E_eval_ID split 50/30/20 exactly as `naive_matched` (`phase3_scorer_h6.py:103`). Estimator refit on the first 50% (CV selection inside it), Mondrian bins = deciles of Ŝ on the 30% conformalize block, per-bin half-width = `empirical_quantile` (conformal `+1`) of S in that bin; test = last 20% of ID and the whole shift block (same calibration). `naive_matched` = static constant band at `empirical_quantile` of S over the first 80% (no ACI; static for every method so it is like-for-like). Oracle = Phase-4 construction (LOO within true-S decile) on the same test windows. `GRR = (W_naive − W_method)/(W_naive − W_oracle)`; reported only if |coverage − 0.90| ≤ 0.02, else "coverage-invalid". "Best ex-ante set" for C3.4 = the set among F1, F1p, F2, F3, F6 with the best fit-split CV MSE (chosen per seed before looking at test).
11. **Bootstrap.** Moving-block, block 96, 2000 resamples, RNG seed 0, over eval windows in time order. Sensitivity: block 192 once on all cells; report whether any verdict changes. Pooled CI: two-level (seeds, then blocks).
12. **Seed thresholds.** "≥ 4/5" means k ≥ ceil(0.8 N) for N ≥ 4 (4/5, 8/10); 3/3 for N = 3.

## Pre-registered criteria (verbatim from the spec)
- **C3.1 Ex-ante predictable:** a feature set is called "predictive" on a (dataset, seed) cell if the block-bootstrap 95% CI of held-out R² on E_eval_ID lies entirely above 0 **and** Spearman ρ CI lies entirely above 0. Difficulty is "ex-ante predictable" on a dataset if ≥ 4/5 seeds (or 3/3 before E2) are predictive for at least one of F1, F1p, F2, F3, F6. *(Operationalised: the same set must reach the seed count; a lenient "any set per seed" count is reported alongside.)*
- **C3.2 Dissociation:** holds on a dataset if C3.1 is met for some ex-ante set **and** F0 (MVDF) is not predictive on ≥ 4/5 seeds.
- **C3.3 MVDF adds nothing:** ΔR² = R²(F7) − R²(F6) has a CI containing 0 or below 0 on ≥ 4/5 seeds.
- **C3.4 Recoverable opportunity:** GRR ≥ 0.20 with valid coverage on ≥ 4/5 seeds for the best ex-ante set. If no set reaches it, the paper says "the oracle gap is an upper bound on heterogeneity, none of which was recovered by any ex-ante predictor we tested."
- **C3.5 Representation contains it:** F4 or F5 is predictive (C3.1 rule) on ≥ 4/5 seeds.

## Traffic / electricity declarations (from E1, repeated here)
Traffic: if naive coverage on the ID test block is > 0.97, E_eval_shift is the primary gap-recovery block and ID is reported but not interpreted. Electricity: naive_matched coverage reported on every block.
