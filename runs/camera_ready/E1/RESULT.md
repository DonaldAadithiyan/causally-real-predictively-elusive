# E1 — Phases 3 + 4 (+ E3) on electricity and traffic

## Provenance

- Git: `93fbb0c` (+uncommitted changes)
- Command: `python -m camera_ready.phase34 --datasets electricity traffic --out runs/camera_ready/E1; python -m camera_ready.e3_exante --datasets electricity traffic; python -m camera_ready.report_e3 --datasets electricity traffic --out runs/camera_ready/E1 --prefix E1_E3; python -m camera_ready.report_e1`
- Seeds: {'electricity': [0, 1, 2, 3, 4], 'traffic': [0, 1, 2, 3, 4]}
- Datasets / copy used: electricity: Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical; traffic: Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical
- Raw-file SHA-256: electricity `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570`; traffic `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: E3 64 min summed over 10 cells; Phase 3/4 39 s; training of seeds 3–4 in E2/meta
- Generated: 2026-11-14 07:10:28 +0530

## Pre-registered criteria (verbatim from PREREG.md)


Committed before the first E1 run.

## What runs
Electricity and traffic, seeds 0/1/2 (+3/4 from E2 when they finish), `max_channels = 48` (the same channel subset as the v2/round-2 runs: 47 most variable + last column as target).
1. Phase 3 battery, marginal + activation-conditioned, exactly `experiments/phase3_real_predictability.py` lines 167–191 (same estimators, 60/40 split, E_eval_ID-local profile, eval-mean R²), computed from saved per-window arrays rather than by retraining (the script retrains the forecaster on CPU from scratch; the saved models are the round-2 models whose MVDF selections reproduce, `runs/p126_round2/wp3/*/summary.json`). The same array path is checked on ETTh seeds 0–2 against `runs/phase3_real/AGG_phase3.csv` first.
2. Phase 4 oracle, exactly `experiments/phase4_oracle_ceiling.py::run_one` from the oracle step onward, from saved arrays; checked on ETTh 0–2 against `runs/phase4_oracle/AGG_oracle.csv`. Block-bootstrap CI (block 96) added next to the script's iid CI.
3. E3 in full (`runs/camera_ready/E3/PREREG.md`).

## Data
Local `data/raw/electricity.csv` and `data/raw/traffic.csv`, byte-identical (SHA-256) to the Kaggle-bundle copies the round-2 models were trained on. The spec names `LD2011_2014.csv` / `PeMS.csv`; the bundle copies actually used by the paper's models are the preprocessed `electricity.csv` / `traffic.csv`, so those are used.
MVDF selections: round-2 frozen selections (the unchanged `frozen_selection`), not re-run. Direction check vs `v2_runs/v2_phase1/AGG_h1c.csv` (electricity 4.00, traffic 4.55, single Kaggle seed): enrichment > 1 expected; any mismatch reported.

## Known issues declared up front
- **Traffic:** the conformal pipeline saturated in-distribution (coverage 1.000) in earlier runs. Phase 3 (R²) and Phase 4 (coverage ≈ 0.90 by construction) are not affected. For E3 gap recovery on traffic, E_eval_shift is the primary block and E_eval_ID is reported but not interpreted if naive coverage on ID is > 0.97.
- **Electricity:** the earlier shift-coverage collapse was caused by the learned conditioning mechanism (naive_matched stayed at 0.894). It does not affect Phase 3/4. naive_matched coverage is reported on every block.
- Round 2 found the traffic ID calibration block contains an outlier stretch (score q90 ≈ 8× typical); recorded here so it is not discovered post hoc.

## Pre-registered criteria
- **C1.1 L1 null generalizes:** best held-out R² of F0 (MVDF) has CI including 0 or below 0 on ≥ 2/3 seeds (≥ 4/5 if five) on each new dataset, for both marginal and active-window batteries. *(CI = moving-block bootstrap, block 96, of the best-on-eval estimator's R² as Phase 3 reports it.)*
- **C1.2 Heterogeneity generalizes:** the oracle gap `(W_naive − W_oracle)/W_naive` has a CI excluding 0 on all seeds of each new dataset. Report the percentage. *(CI = moving-block bootstrap of the per-window width difference.)*
- **C1.3:** E3 criteria C3.1–C3.5 evaluated on each new dataset.

## Verdicts

| criterion | electricity (N = 5) | traffic (N = 5) |
|---|---|---|
| **C1.1** L1 null generalizes (F0 best R² CI ∋ 0 or < 0, both batteries, ≥ 4/5) | **PASS**: 5/5 marginal, 5/5 active; every CI entirely below 0 | **PASS**: 5/5 marginal, 5/5 active; every CI entirely below 0 |
| **C1.2** heterogeneity generalizes (oracle gap CI excludes 0, every seed) | **PASS**: ID 18.1% [17.3, 18.3], shift 37.2% [36.4, 37.9] | **PASS**: ID 39.1% [39.0, 39.6], shift 37.3% [36.3, 37.5] |
| **C1.3 → C3.1** ex-ante predictable | **FAIL**: best set F3/F6 1/5 | **FAIL**: F1 1/5 |
| C3.2 dissociation | FAIL (needs C3.1); F0 not predictive 5/5 | FAIL (needs C3.1); F0 not predictive 5/5 |
| C3.3 MVDF adds nothing | **PASS** 5/5 | **PASS** 4/5 |
| C3.4 recoverable opportunity | **FAIL** (valid coverage on 1/5, GRR −0.41) | **FAIL** (shift block primary as pre-declared; coverage 1.000 on 5/5 → invalid) |
| C3.5 representation contains it | **FAIL** (F4 0/5, F5 0/5) | **FAIL** (F4 0/5, F5 0/5) |

**Block-192 sensitivity (pre-registered, run once):** on electricity, F3 (forecast statistics) becomes predictive on 4/5 seeds, so **C3.1 and C3.2 flip to PASS under block 192**. Block 96 is primary as pre-registered, so the verdict stays FAIL; the flip is reported here, not suppressed. No other verdict on any dataset changes.

Pre-declared issues, as observed:
- **Traffic:** naive_matched covers 1.000 on both ID and shift test blocks on every seed (score q90 of the calibration block is inflated by an outlier stretch, windows 1604–1699 of seed 0, all inside the Phase-3 fit split). E_eval_shift was declared primary, but it is saturated too, so no traffic GRR is interpretable.
- **Electricity:** naive_matched coverage, static path: ID test 0.85, shift 0.57 (median). The shift collapse round 2 attributed to score shift is present for the constant band too.
- **Enrichment direction vs `v2_runs/AGG_h1c.csv`** (electricity 4.00, traffic 4.55): local models give electricity 3.85 [2.00, 4.17] and traffic 3.33 [1.43, 4.17] over 5 seeds. Same direction (> 1 on 10/10); no mismatch.

## Plain-English summary
Both reviewer points on dataset scope now have direct answers. On electricity and traffic, as on ETTh, natural MVDF activation predicts difficulty worse than a constant on every seed (best R² −0.29 to −1.33, every CI below zero). Real heterogeneity exists (an oracle would be 18% narrower in-distribution on electricity, 39% on traffic). Cheap ex-ante features do carry a rank signal here (electricity input/forecast statistics ρ ≈ 0.24–0.33; traffic input statistics ρ ≈ 0.34). But only 0–1 of 5 seeds clear the joint R²-and-ρ bar at block 96. On traffic the R² is badly negative because the fit split contains an outlier stretch that sets the level. No CV-selected ex-ante band is narrower than the constant band at valid coverage. The only positive cases (traffic shift, forecast statistics F3, GRR 0.28–0.41 on 4/5 seeds; not CV-selected) are measured against a constant band inflated by the calibration outlier (coverage 1.000); see `F/RESULT.md` §5.

## Anomalies
- **Traffic bootstrap:** on 3/5 traffic seeds the Phase-3 point R² lies just outside its own block CI (e.g. s0 −1.333 vs CI [−4.95, −1.354]). The heavy-tailed outlier stretch makes the moving-block bootstrap distribution skewed; the CI is still entirely below 0, so C1.1 is unaffected.
- **Traffic R² levels** (F2 ≈ −13.6, F6 down to −28.9) are a level-mismatch artifact of the fit-split outlier stretch (fit mean S 2.23 vs eval mean 1.39). The rank correlations are the informative quantity there.
- **Traffic fingerprint string:** `get_splits(...).fingerprint()` and round-2's `data_fingerprint(...).edge_hash` are different hash functions. Data identity is confirmed by the raw-file SHA-256 and by exact SAE FVU reproduction of all three original seeds.

## For the paper

| dataset | seeds | MVDF best R², marginal (median [min, max]) | oracle gap, ID | ex-ante: best ρ / predictive seeds | GRR (valid) |
|---|---|---|---|---|---|
| ETTh1 | 10 | −0.60 [−0.72, −0.52] | 25.2% [23.6, 25.9] | F6 −0.02 / 1/10 | none |
| ETTh2 | 10 | −0.89 [−0.94, −0.77] | 30.0% [28.2, 30.6] | F4 +0.38 / 10/10 (repr.), F2 +0.32 / 6/10 | none |
| electricity | 5 | −0.29 [−0.33, −0.29] | 18.1% [17.3, 18.3] | F6 +0.33 / 1/5 | −0.41 (1/5 valid) |
| traffic | 5 | −0.70 [−1.33, −0.46] | 39.1% [39.0, 39.6] | F2 +0.34 / 0/5 | none (coverage 1.000) |

> "Across all four datasets (30 trained runs), natural MVDF activation fails to predict difficulty (best held-out R² −0.29 to −1.33, every block-bootstrap CI below zero), while an oracle shows 18–39% of interval width is attributable to difficulty heterogeneity. No ex-ante predictor chosen by cross-validation converts this into a narrower conformal band at valid coverage."

---

## Detail (auto-generated)

## Phase 3 (original definitions), per seed

| dataset | seed | n_fit | n_eval | mvdf_activation_rate | n_active_windows_eval | marginal best R² [block CI] | active best R² [block CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | 0 | 1400 | 934 | 0.404 | 361 | -0.293 [-0.897,-0.068] | -0.331 [-0.883,-0.304] |
| electricity | 1 | 1400 | 934 | 0.421 | 384 | -0.288 [-0.942,-0.062] | -0.350 [-0.929,-0.310] |
| electricity | 2 | 1400 | 934 | 0.503 | 445 | -0.327 [-1.085,-0.059] | -0.477 [-1.377,-0.285] |
| electricity | 3 | 1400 | 934 | 0.422 | 401 | -0.289 [-0.932,-0.066] | -0.313 [-0.944,-0.163] |
| electricity | 4 | 1400 | 934 | 0.397 | 393 | -0.310 [-1.007,-0.075] | -0.478 [-1.134,-0.383] |
| traffic | 0 | 701 | 468 | 0.566 | 270 | -1.333 [-4.951,-1.354] | -2.787 [-10.221,-3.188] |
| traffic | 1 | 701 | 468 | 0.661 | 303 | -0.506 [-2.960,-0.482] | -0.609 [-5.678,-0.684] |
| traffic | 2 | 701 | 468 | 0.387 | 177 | -0.848 [-4.155,-0.862] | -2.216 [-12.312,-3.027] |
| traffic | 3 | 701 | 468 | 0.612 | 298 | -0.696 [-2.801,-0.715] | -1.057 [-5.630,-1.148] |
| traffic | 4 | 701 | 468 | 0.657 | 312 | -0.457 [-2.805,-0.323] | -0.460 [-3.867,-0.451] |

**C1.1:** electricity: marginal 5/5, active 5/5 seeds with CI ∋ 0 or ≤ 0 (need 4) → **PASS**; traffic: marginal 5/5, active 5/5 seeds with CI ∋ 0 or ≤ 0 (need 4) → **PASS**

## Phase 4 oracle (original construction), per seed

| dataset | seed | n_id | n_shift | naive_id_coverage | oracle_id_coverage | gap % id [block CI] | naive_shift_coverage | gap % shift [block CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | 0 | 2334 | 2237 | 0.901 | 0.901 | 18.3 [13.9,22.0] | 0.901 | 37.9 [27.7,46.3] |
| electricity | 1 | 2334 | 2237 | 0.901 | 0.901 | 17.3 [13.2,21.1] | 0.901 | 36.4 [26.1,44.8] |
| electricity | 2 | 2334 | 2237 | 0.901 | 0.901 | 18.0 [13.9,21.7] | 0.901 | 37.2 [26.8,45.6] |
| electricity | 3 | 2334 | 2237 | 0.901 | 0.901 | 18.1 [13.9,21.8] | 0.901 | 36.7 [26.1,45.2] |
| electricity | 4 | 2334 | 2237 | 0.901 | 0.901 | 18.3 [14.1,22.2] | 0.901 | 37.7 [27.4,46.2] |
| traffic | 0 | 1169 | 1074 | 0.901 | 0.906 | 39.6 [9.5,61.3] | 0.901 | 36.6 [11.7,53.4] |
| traffic | 1 | 1169 | 1074 | 0.901 | 0.906 | 39.1 [8.3,61.2] | 0.901 | 36.3 [11.0,53.3] |
| traffic | 2 | 1169 | 1074 | 0.901 | 0.906 | 39.0 [8.3,61.1] | 0.901 | 37.3 [13.2,53.8] |
| traffic | 3 | 1169 | 1074 | 0.901 | 0.906 | 39.5 [8.7,61.5] | 0.901 | 37.4 [12.5,54.5] |
| traffic | 4 | 1169 | 1074 | 0.901 | 0.906 | 39.1 [8.4,60.9] | 0.901 | 37.5 [13.1,53.8] |

**C1.2:** electricity: gap ID 18.1 [17.3, 18.3]%, shift 37.2 [36.4, 37.9]% → **PASS** (CI excludes 0 on every seed, both blocks); traffic: gap ID 39.1 [39.0, 39.6]%, shift 37.3 [36.3, 37.5]% → **PASS** (CI excludes 0 on every seed, both blocks)


E3 tables for electricity/traffic: `_tables_E1_E3.md`.
