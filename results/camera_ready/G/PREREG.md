# G — Final round before the camera-ready: pre-registration

Committed before any G computation. Rules as before: nothing is overwritten (all output under `runs/camera_ready/G/`); block bootstrap uses block 96, 2000 resamples, seed 0; every result is reported as per-seed values, median [min, max] and k/N.

## G1 Table 7 on the local data copy (30 pipelines)
- **Modes:** naive_matched, naive_no_scorer, dense, mvdf, dense+mvdf, correlated, random_k, pca_k, CQR, PID-conformal.
- **Protocol: unchanged.** It is `experiments/phase3_scorer_h6.py` L97–188 for the scorer and naive modes, and `experiments/baseline_comparison.py` L141–195 for CQR and PID:
  - 50/30/20 split of E_eval_ID;
  - block-local profiles (each block its own);
  - global ACI γ = 0.03 with immediate feedback, as in the originals;
  - random_k: rng `derive_seed + 777`, drawn from features active on ≥ 2% of the train block;
  - pca_k: PCA(12) fit on the train block;
  - correlated: the run's stored top-12 |corr| set;
  - CQR: quantile iTransformer, conformal correction on the conformalize block;
  - PID: q0 from the 80% pool, η = 0.03.
- **New training (the only training in G):** the CQR quantile model for the 18 runs that have none (ETTh1/ETTh2 seeds 3–9, electricity/traffic seeds 3–4). The recipe is identical to the existing ones: `QuantileTrainConfig`, seed `derive_seed + 31337`, same forecaster hyper-parameters. ETTh is trained on CPU and electricity/traffic on MPS, as for their forecasters. Seeds 0–2 use the saved round-2 quantile models.
- **Report:** per dataset × mode × block, coverage and width as median [min, max]; the count of runs where mode width < naive_matched width **and** coverage ∈ [0.88, 0.92]; and the count of runs where the paired block-bootstrap CI of (mode − naive_matched) width lies entirely below 0. The compact table is shift block × mode × 4 datasets.

## G2 Pre-evaluation (E_select) profile
For all 30 runs, recompute (a) the Phase-4 oracle gap ratio, ID and shift, and (b) the Phase-3 marginal and active best R², with S_t defined using the E_select profile instead of the block-local profile. Everything else is unchanged. Report both versions side by side, plus the Spearman ρ between the two S_t definitions on E_eval_ID and on E_eval_shift.

## G3 Synthetic positive control, seeds 0–9
- Run `experiments/phase1_synthetic_control.py --seeds 0 … 9 --out runs/camera_ready/G/phase1_synthetic`, unchanged (config frozen; CPU).
- Use the script's own pass rules for links 1–4, unchanged.
- Report k/10 per link and the positive-control marginal best R² as median [min, max].
- Reproduction check: seeds 0–2 must match `runs/phase1_synthetic/AGG_phase1.csv`.
- Note: round 2 showed link 4 passes 3/3 on a corrected evaluation path but is not MVDF-specific. That is not re-tested here; G3 is the paper's protocol on 10 seeds.

## G4 Direct L2 evidence per run
For each of the 30 runs and each pool feature, compute a 95% moving-block bootstrap CI of U_j (block 96 over dose-window time positions; inactive windows dropped; 2000 resamples; the same dose-response as E5a). A feature "moves S" if the CI excludes 0. Per run, compute the fraction that move S among:
- the 12 MVDF features;
- the remaining pool features;
- random 12-feature draws from the pool (1000 draws; report their mean and where the MVDF fraction falls among them).

k = runs where the MVDF fraction > the rest-of-pool fraction. Report per dataset as median [min, max] and k/N. Two versions:
- **(a) in-sample, as requested:** U_j and its CI on the E_select dose windows, the same windows MVDF was selected on. This is **selection-inflated by construction** (MVDF is ranked by a residual of this U_j).
- **(b) out-of-sample:** U_j and its CI re-estimated with the same dose-response on E_eval_ID windows, which played no part in selection.

**If (a) and (b) disagree, (b) is the one the paper's "causally real" claim should rest on.** Both are reported.

## G5
Read round 2's static-calibration comparison (`runs/p126_round2/summary/wp1_continuous_red.csv`, protocol `P1_chrono_static`, segment `shift`, seeds 0–2): the width reduction vs naive_matched for every learned-scorer mode on ETTh1/ETTh2 (and electricity/traffic if present). Report the exact range and whether any mode is narrower (reduction > 0) on all 3 seeds of a dataset.
