# Step 0 — orientation facts (ETTh1 / ETTh2, iTransformer)

All facts verified against code and the regenerated arrays (`p126/regen_arrays.py`).

1. **Shapes.** Lookback L = 96, horizon H = 96, window stride 1. 7 channels per dataset
   (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT). Forecaster is multivariate in, multivariate out.
2. **Channel aggregation of S_t.** None. S_t uses only the target channel OT
   (`target_var = -1`), `validation/errors.py::collect`:
   `S_t = max_h |y[t,h,OT] − ŷ[t,h,OT]| / profile[h]`. Calibration and evaluation use the
   same single-channel, joint-over-horizon definition. The SAE hidden state is also the
   OT variate token.
3. **Index t.** t is the window index inside the test split. The forecast origin is the
   window's last input step, `end_idx = start + L − 1`. S_t is fully observable only at
   `end_idx + H`, i.e. at the origin of window t + H.
4. **Split boundaries** (identical for ETTh1 and ETTh2, HTTP ETDataset mirror,
   fingerprints `c6d1709fd0693b1a` / `c022db5521d23684`):
   - Rows: train [0, 8640), val [8640, 11520), test [11520, 17420) = 5900 rows → 5709 test windows.
   - `three_way_env_masks` assigns windows by end index (cuts at test rows 1967 and 3933):
     - E_select: windows 0–1871 (1872), origins 2017-10-27 23:00 → 2018-01-13 22:00
     - E_eval_ID: windows 1872–3837 (1966), origins 2018-01-13 23:00 → 2018-04-05 20:00
     - E_eval_shift: windows 3838–5708 (1871), origins 2018-04-05 21:00 → 2018-06-22 19:00
   - Scorer / baseline / CQR / PID runs (50/30/20 of E_eval_ID): scorer-train 1872–2854 (983),
     conformalize 2855–3443 (589), width-eval 3444–3837 (394).
   - Phase-3 R² battery (60/40 of E_eval_ID): fit 1872–3050 (1179), eval 3051–3837 (787).
5. **Gaps.** None anywhere. Adjacent windows share 95 of 96 target steps. Window t and
   window t+k share 96−k target steps, so windows within 96 of a boundary share realised
   targets across it. Windows within 192 (= L+H) share some input or target value.
   The 394-window width-eval block is about 2 blocks of 2H.
6. **Where reported numbers are computed.**
   - R² table: `experiments/phase3_real_predictability.py` L168–191, using
     `experiments/phase1_synthetic_control.py::_fit_eval_estimators`. The R² denominator is
     `sum((y − mean(y_eval))²)`, the evaluation-set mean, not the training mean the
     caption describes. Saved in `runs/phase3_real/`.
   - Oracle: `experiments/phase4_oracle_ceiling.py` → `runs/phase4_oracle/AGG_oracle.csv`.
   - `naive_matched` 0.836 and CQR coverage: `experiments/baseline_comparison.py` →
     `runs/baseline_comparison/AGG_baselines.csv`. ETTh1/ETTh2 only, local HTTP data.
   - `naive_matched` 0.895, electricity 0.630 (`mvdf`, seed 1, shift), traffic 1.000:
     `experiments/phase3_scorer_h6.py`, run on Kaggle with the Kaggle data bundle →
     `v2_h6_topup/AGG_h6.csv` (also `v2_confirm/v2_phase3_h6/`).
7. **Which split `profile[h]` comes from.** `collect()` recomputes the profile from
   whatever block it is given. Consequences:
   - In the Phase-3 R² battery, S_t on E_eval_ID is normalised by the E_eval_ID profile.
   - In every conformal run (`baseline_comparison`, `phase3_scorer_h6`), the
     scorer-train, conformalize, width-eval and shift blocks each get their own profile.
     Calibration scores and test scores are therefore normalised by different profiles,
     and the test profile is computed from test outcomes.
   - In the Phase-4 oracle, each block uses its own profile.
   - The E_select profile (`cal_sel`) is used only for dose-response.
   This follow-up uses the E_select profile for every new S_t (`p126/common.py::Run.S`),
   and reproduces the original per-block definition where an original number is checked.
   The two definitions have rank correlation about 0.97 on E_eval_ID.

**Regeneration.** No forecaster or SAE weights and no per-window arrays existed in the
repo. Forecaster, SAE and the CQR quantile model were retrained with the original configs
and seeds, on CPU with 4 torch threads (the default the original runs used). A first
attempt with 3 threads did not reproduce the saved SAE FVU, because thread count changes
reduction order, so it was discarded. See `arrays/*_repro.json` for the match to the
saved `forecaster_val_mse` / `sae_fvu`. MVDF and correlated indices are read, not
re-derived, from `runs/phase3_real/<ds>_seed<s>/mvdf_selection.json`.

**Not available locally.** Electricity, traffic and the Kaggle-bundle ETT files (raw data
is Kaggle-only), so the scorer run's exact numbers cannot be regenerated here. TimeMixer
and electricity/traffic extensions were not attempted. The synthetic control is excluded
at the user's request.
