# Known issues, deviations and limitations

Nothing here was fixed silently. Each entry gives its effect on the paper and where the paper
states it. Sources: `results/camera_ready/CAMERA_READY_SUMMARY.md` §4, `results/camera_ready/F/RESULT.md`,
and checks made while packaging this repository.

## A. Numbers in the camera-ready `main.tex` that do not match the committed results

Found by `tests/test_paper_numbers.py`. All three are pinned as strict expected failures, so CI
fails if any of them changes.

| paper text | committed value | cause | fix in the paper |
|---|---|---|---|
| §5.5 TimeMixer: "$R^2$ −0.49 to −1.17" | −0.49 to **−1.16** (min −1.1648) | double rounding (−1.1648 → −1.165 → −1.17) in the camera-ready change list, which was copied into the paper | print −1.16 (the submitted paper's value was right) |
| §5.1: "median 2.0–3.9×" | **2.0–3.8×** (medians 2.04 / 2.28 / 3.85 / 3.33) | double rounding (3.846 → 3.85 → 3.9) in the camera-ready summary sentence | print 2.0–3.8× (or 2.04–3.85× to match Table 2) |
| §5.4 Localization: correlated features "0.81 … (2/6)" reported as cells "below the pre-set 0.80 bar" | 0.81; **3/6** cells below 0.80 vs dense; **2/6** pass the full pre-registered rule (below 0.80 vs dense *and* PCA, CI excluding 1) | wording: the count is "passing", not "below the bar" | say "pass the pre-set 0.80 rule (2/6)", or report 3/6 below 0.80 |

Two further notes (no mismatch, but a reader should know):
- §3.1 "0.08–0.21" is the PCA-based maximum |corr| per tap (`max_abs_corr_pc`). The raw
  maximum |corr| is 0.087–0.215 (0.09–0.22).
- §5.5 necessity "−0.0004 to −0.0021 on 5 of 6 runs": the sixth run (ETTh2 seed 2) is +0.0249.
  The paper's "on 5 of 6" is correct as written.

## B. Measurement and construction caveats (stated in the paper)

1. **`profile_h` is block-local.** Each scored block is normalised by its own realised errors
   (`src/mechconf/validation/errors.py`, `collect`). Using the E_select profile instead (fixed
   before evaluation) changes no conclusion. The two S_t agree with rank correlation 0.95–0.997,
   and the oracle gap moves by at most 0.03 (G2, `results/camera_ready/G/AGG_G2_select_profile.csv`).
   Paper: §3.1.
2. **Interventions drop the SAE reconstruction error**, also at λ = 1: the patched token is the
   SAE reconstruction. Paper: §3.2. E5c (interchange) follows the same code, although its spec
   text said the error term is "kept fixed".
3. **Immediate-feedback ACI.** naive_matched and the learned scorers use the same online ACI
   correction, updated after each window with that window's full 96-step outcome. That outcome is
   not observable at forecast time, so neither method is deployable as run. The comparison is
   like-for-like. Under static calibration (round 2, 3 seeds) no learned scorer is consistently
   narrower on the ETTh shift block (−0.09 to +0.004). Paper: §6 Limitations; data in
   `results/camera_ready/G/AGG_G5_static_shift.csv`.
4. **Phase-3 R² uses the evaluation-set mean** as reference, not the training mean. Paper: §5.2
   ("computed against the mean of the evaluation windows"). E3 reports both.
5. **Traffic:** every method covers 1.000 in-distribution. An outlier stretch (windows 1604–1699,
   seed 0) sits in the calibration/fit block, so the constant band is about 3.9× the oracle. Traffic
   supports no in-distribution width claim, and its shift-block gap-recovery positives are against
   this inflated band. Paper: §4 Caveats, §5.4.
6. **ETTh in-distribution blocks under-cover (≤ 0.86) for every method**, including naive_matched.
   The gap-recovery ratio there is undefined (W_naive < W_oracle), and the shift block is the
   primary comparison. Paper: §4 Caveats.
7. **Coarse block bootstrap.** 787 evaluation windows are about 8 blocks of 96, so several
   consistent signals have CIs that include 0. At block length 192, electricity's forecast
   statistics become predictive on 4/5 seeds and C3.1/C3.2 flip to PASS there. Block 96 is the
   pre-registered primary. Source: `results/camera_ready/E1/RESULT.md`.
8. **The nominal 5% for "U_j interval excludes 0" is not calibrated.** The percentile
   moving-block bootstrap was not checked against a placebo intervention. Paper: §5.1 ("we read
   this as a large gap rather than a calibrated test").

## C. Bugs in earlier stages (found during the camera-ready work)

1. **Phase-5 "signature-shuffle floor" equals the match rate by construction.** Permuting the rows
   of one SAE's signature matrix does not change mutual-nearest-neighbour matching, so the floor
   is not a null. The paper no longer cites it. The code
   (`experiments/v2/phase5_sae_reproducibility.py`) is unchanged.
2. **Phase-5 report text said "0.5 SD"; the threshold used is 0.25 SD.** Fixed in E0; the
   regenerated report differs only in that text (`results/camera_ready/E0/PHASE5_RESULT.diff`).
3. **H7-A (regime gating) original run used a mismatched feature list.** `experiments/v2/h7a_gate.py`
   retrains a local SAE but reads its cell-B feature IDs from the Kaggle runs' type tables, which
   were produced by a different SAE. The one original pass (ETTh2 seed 1) does not reproduce. On
   the corrected construction 0/20 runs pass (F1). The script is unchanged; F1 is in
   `experiments/camera_ready/f_followup.py`.
4. **Enrichment-CI order dependence** (helper added in the camera-ready round): fixed during the
   run, which changed one C5b cell. See `docs/PREREGISTRATION.md`.

## D. Deviations from the task specifications' wording

- E5c reconstruction error: see B2.
- E3 gap recovery uses static split conformal for every method (like-for-like), not naive_matched's ACI.
- Electricity/traffic files are the preprocessed bundle files `electricity.csv` / `traffic.csv`
  that the paper's models were trained on, not `LD2011_2014.csv` / `PeMS.csv`.
- E4 spline residualiser fitted on z-scored P/U (numerical conditioning).
- F1 regime gating: corrected construction instead of the literal `h7a_gate.py` (see C3).
- G4 adds an out-of-sample version (U_j re-estimated on E_eval_ID), because the requested
  in-sample version is selection-inflated by construction. The paper uses the out-of-sample version.

## E. Incomplete work (committed as partial)

- **G1** (downstream Table 7 on the local data copy) was stopped. Quantile (CQR) models exist for
  12/30 runs. The paper's earlier-12-run widths (3.89/3.94/4.36 vs 3.35) still come from the
  Kaggle-copy runs (`results/v2/v2_h6_topup/`).
- **G3** (synthetic control on 10 seeds) was stopped after seeds 0–1, which reproduce the paper's
  seeds 0–1. The paper's synthetic numbers are the original 3-seed run (`results/v2/phase1_synthetic/`).
- Table 3 of the submitted paper (invariance, mean ± SD over 3 seeds) was not recomputed. The
  section was dropped from the camera-ready.

## F. Reproducibility tolerances

- Bit-exact: the ETTh1 s0 retrain through the camera-ready driver, all 6 TimeMixer runs, all 30
  dose-response pools and MVDF sets, the Phase 3/4 and E5a outputs after the layout restructure.
- Within tolerance: Phase 3/4 recomputed from saved arrays vs the original scripts' stored numbers
  (≤ 3×10⁻³; float32 storage of the arrays feeding gradient boosting and the pinball network).
- Electricity/traffic forecasters and SAEs were trained on Apple MPS; MPS kernels are not
  guaranteed bit-reproducible across machines.
