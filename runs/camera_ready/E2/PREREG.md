# E2 — More seeds: pre-registration

Written and committed before any E2 training run. Timestamp and git hash: see the commit that adds this file (`git log -- runs/camera_ready/E2/PREREG.md`).

## Plan (fixed in advance)
- **Required:** seeds 3, 4 on ETTh1 and ETTh2. **Stretch (taken because one ETTh seed costs ≈ 5 min on CPU, far under the 2 h bar):** seeds 5–9 on ETTh1 and ETTh2 → N = 10 on ETTh. **If compute allows:** seeds 3, 4 on electricity and traffic → N = 5 there.
- Pipeline per seed, unchanged settings (`experiments/phase3_real_predictability.py::CFG`): iTransformer d_model 256 / 3 layers / 8 heads / d_ff 512, L = H = 96, 20 epochs, patience 5; TopK SAE expansion 8 (m = 2048), k = 64, 12000 steps; pool 120, window cap 2048, λ ∈ {0, .25, .5, .75, 1, 1.25}, k = 12, isotonic residualisation, u_threshold 0. Seed = `derive_seed(seed, ds)`.
- Code path: `p126/r2_wp6.regen` (forecaster → SAE → `p126.selection.frozen_selection`, which calls the unchanged `screen / sweep_pool / feature_type_table / select_mvdf_v2 / enrichment`), outputs redirected to `runs/camera_ready/{weights,arrays,E2/meta}`. Quantile (CQR) model not trained (no E2 metric uses it).
- Device: ETTh on CPU, 4 torch threads (the setting under which seeds 0–2 reproduce exactly). Electricity/traffic on MPS (as their seeds 0–2 were trained in round 2).
- Data: local `data/raw/` (ETTh = HTTP mirror; electricity/traffic byte-identical to the Kaggle-bundle copies). `KAGGLE_DATA_DIR` pointed at an empty directory so the Kaggle ETT copy can never be picked up.

## Health rule (pre-registered)
A seed is **unhealthy** if SAE FVU > 0.05 **or** forecaster val MSE > 1.5 × the seed-0 val MSE of the same dataset. An unhealthy seed is reported, kept in every table, and flagged; a replacement seed (next unused integer) is added only if this rule triggers. No seed is dropped silently.

## Metrics recomputed for all seeds
1. Type-B enrichment of MVDF (H1c), exactly `validation.difficulty_specificity.enrichment`. Added (the code computes no uncertainty): one-sided hypergeometric p-value for #B in MVDF given pool composition, and a 95% CI for the ratio by bootstrap over pool features (2000 resamples, re-running the type split + selection on each resample is NOT done; the CI resamples the 12 selected features' types and the pool base rate). "Enrichment > 1 with CI excluding 1" uses this CI.
2. Phase 3 L1 predictability, marginal and activation-conditioned, the original battery (`_fit_eval_estimators`, 60/40 split, E_eval_ID profile, eval-mean R²) — reproduced from saved arrays; seeds 0–2 must match `runs/phase3_real/AGG_phase3.csv`.
3. Phase 4 oracle gap, ID and shift, original construction — seeds 0–2 must match `runs/phase4_oracle/AGG_oracle.csv`.
4. Continuous scorer vs naive_matched (`v2_h6_topup` protocol: `DifficultyScorer` 50/30/20, global ACI γ as configured), modes dense and mvdf: shift width difference and coverage.
5. E3, E4, E5a/b on the new seeds (criteria in their own PREREGs).

## Statistics
Per seed value + moving-block bootstrap 95% CI over evaluation windows (block 96, 2000 resamples, RNG seed 0; block 192 once as sensitivity), median and min–max across seeds, pass count k/N, pooled two-level bootstrap CI (resample seeds, then blocks), one-sided sign test over seeds for directional claims. No SD over seeds.

## Pre-registered criteria
- **C2.1:** every claim currently stated as "3/3 seeds" holds on ≥ 4/5 seeds (≥ 8/10 where N = 10). Any claim that drops below this is reported as weakened, and the paper text changes to match.
- **C2.2:** no headline median moves by more than its own pooled CI half-width when going from 3 to 5 (or 10) seeds.
