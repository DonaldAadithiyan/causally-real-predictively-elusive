# E1 — Phases 3 + 4 (+ E3) on electricity and traffic: pre-registration

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
