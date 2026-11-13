# Paper 126 follow-up: results for A, B, C

Scope: ETTh1 and ETTh2, iTransformer, seeds 0–2. Real data only; the synthetic control
was dropped at the user's request. All new code is in `p126/` and all outputs are in
`runs/p126/`. No original file was modified.

## Step 0: orientation (full detail in `STEP0_NOTES.md`)

- **Shapes.** L = 96, H = 96, stride 1, 7 channels.
- **S_t.** Uses only the target channel OT, with no channel aggregation:
  `S_t = max_h |e[t,h]| / profile[h]`. Coverage is joint over the horizon in both
  calibration and evaluation.
- **Index and observability.** t is the window index in the test split. The origin is the
  last input step, and S_t becomes observable H = 96 windows later.
- **Splits.** Test windows 0–5708. E_select is 0–1871; E_eval_ID is 1872–3837 (1966;
  2018-01-13 → 2018-04-05); E_eval_shift is 3838–5708 (1871). The conformal runs split ID
  50/30/20: windows 1872–2854, 2855–3443, 3444–3837. The R² battery splits ID 60/40:
  1872–3050 and 3051–3837.
- **No gap anywhere.** Adjacent windows share 95/96 targets. The 394-window width-eval
  block is about 2 blocks of 2H.
- **`profile[h]`.** It is recomputed from whatever block `collect()` receives. The R²
  battery uses the ID-block profile. In the conformal runs, the scorer-train, conformalize,
  width-eval and shift blocks each get their own profile, so calibration and test scores
  are normalised differently, and the test profile uses test outcomes.
  New analyses here use the E_select profile throughout (rank correlation with the
  original S_t on ID: 0.95–0.98).
- **Regeneration.** No weights or per-window arrays existed. Forecaster, SAE and the CQR
  quantile model were retrained with the original configs and seeds (CPU, 4 threads). All
  6 runs reproduce the saved `forecaster_val_mse` (10 digits) and `sae_fvu` (8 digits)
  exactly, so the saved MVDF/correlated indices were reused as frozen input. A first
  attempt at 3 threads did **not** reproduce the SAE FVU and was discarded.
- **Activation rate.** Per-feature MVDF activation on ID is 2.9–9.6%. *Any* MVDF feature
  is active on 28–76% of windows, so the "active" subset is large, not 4%.

---

## A. Is the negative R² a real null?

**Answer.** The reported negative R² is a level shift. It says nothing about the
features, and the metric was uninformative. In all 6 runs, the constant predictor
`mean(y_fit)` alone reproduces the reported R² to within 0.03. Mean S_t rises from 0.52
to 0.73 (ETTh1) and from 0.95 to 1.34 (ETTh2) between the fit and eval segments. Skill
against that constant is between −0.013 and +0.015 for the best estimator. The
shift-robust metrics then show:

- **ETTh1:** the null stands. MVDF has no rank signal in any test.
- **ETTh2:** the null stands for MVDF, but the picture is **mixed** in two places.
  - On seed 2, held-out Spearman is 0.12–0.17 (linear p = 0.022, GBM p = 0.032,
    circular-shift null), and this does not replicate in blocked CV.
  - The dense hidden state and its PCA projection carry weak rank signal (within-fold
    Spearman 0.14–0.25) that the SAE subsets do not. So on ETTh2 the null is specific to
    the SAE subsets, not a property of the forecaster.

**Table A1: R² decomposition (original S_t and split; best of the 4 estimators)**

| run | reported R² (all) | R2_const (all) | best skill (all) | reported R² (active) | R2_const (active) | best skill (active) |
|---|---|---|---|---|---|---|
| ETTh1 s0 | −0.635 | −0.624 | −0.007 | −0.695 | −0.698 | +0.002 |
| ETTh1 s1 | −0.718 | −0.718 | −0.000 | −0.522 | −0.531 | +0.006 |
| ETTh1 s2 | −0.581 | −0.594 | +0.008 | −0.474 | −0.483 | +0.006 |
| ETTh2 s0 | −0.925 | −0.908 | −0.009 | −0.804 | −0.780 | −0.013 |
| ETTh2 s1 | −0.923 | −0.907 | −0.008 | −0.894 | −0.882 | −0.006 |
| ETTh2 s2 | −0.896 | −0.925 | +0.015 | −0.877 | −0.899 | +0.012 |

My recomputation reproduces the saved R² exactly. S_t distribution, fit → eval:

| | mean | median | q90 | sd |
|---|---|---|---|---|
| ETTh1 | 0.52 → 0.73 | 0.50 → 0.66 | 0.72 → 1.10 | 0.15 → 0.26 |
| ETTh2 | 0.96 → 1.34 | 0.90 → 1.24 | 1.40 → 1.85 | 0.33 → 0.40 |

Active windows contribute roughly their share of squared error: 18–80% of SSE against
28–76% of windows. Files: `A/r2_decomposition.csv`, `A/summary.json`.

**Table A2: shift-robust metrics**

| test | ETTh1 | ETTh2 |
|---|---|---|
| Held-out estimators, Spearman (eval seg., 4 est. × 3 seeds) | −0.115 … +0.074; 0/12 with p<0.05 | −0.008 … +0.173; 2/12 with p<0.05 (both seed 2) |
| Held-out estimators, AUROC top decile (eval threshold) | 0.43–0.51 | 0.37–0.55 |
| Single MVDF features, Spearman (36 feature × seed) | median −0.010, max \|ρ\| 0.17; 3/36 with p<0.05 | median −0.003, max \|ρ\| 0.18; 2/36 with p<0.05 |
| Single MVDF features, Δmedian S (active − inactive) | median −0.002 | median +0.007 |
| Blocked 5-fold CV in ID (embargo 192), MVDF, within-fold Spearman (mean over seeds) | −0.012 … +0.008; 0/12 sig | +0.010 … +0.054; 0/12 sig |
| Same, log S target | −0.008 … +0.017 | +0.012 … +0.057 |
| Same, S minus causal trailing baseline: pooled R² | −0.01 … −0.10 (from −0.31) | −0.03 … −0.12 (from −0.36) |
| Correlation-selected SAE (GBM / linear) | −0.031 / −0.061 | +0.027 / +0.020 |
| Random SAE subset | −0.010 / −0.004 | +0.054 / +0.054 |
| PCA-12 of hidden state | −0.122 / −0.079 | **+0.136 / +0.249** (linear sig on 2/3 seeds) |
| Dense hidden state (GBM) | −0.045 | **+0.165** (0.137–0.188) |

Notes:
- 5/72 single-feature tests at p < 0.05 is about the chance rate (3.6).
- Removing a causal level baseline moves R² toward 0, which is consistent with the
  level-shift reading.
- The pooled blocked-CV Spearman is strongly negative (−0.4). That is an artifact
  (leave-fold-out level effect), which is why the within-fold value is reported.
- Files: `A/rank_metrics.csv`, `A/blocked_cv.csv`.

**A7: feature-level relationship.** I correlated R_j with predictiveness across the
120-feature pool (bootstrap over features).
- R_j vs |Spearman|: ρ from −0.21 to +0.16; the CI includes 0 in 5 of 6 runs.
- R_j vs AUROC: ρ from −0.31 to +0.08; negative on ETTh1, with the CI excluding 0 on 2 seeds.
- Every pool feature has |Spearman| ≤ 0.16 and AUROC ≈ 0.5, so "predictiveness" is
  mostly noise and this correlation carries little weight.

File: `A/feature_level.json`.

**Verified by test vs inferred**
- Verified: the level-shift explanation (R2_const), the circular-shift nulls, and blocked
  CV with embargo.
- Inferred: that the seed-2 ETTh2 signal is a fluke, from non-replication only.
- Caveat: the eval segment holds 787 windows, about 4.1 blocks of 2H. Block-bootstrap CIs
  for A2 are therefore unreliable (below 5 blocks), so significance relies on the
  circular-shift null.

---

## B. How much difficulty is predictable at forecast time?

**Answer.** Essentially none, with these predictors on these data.
- On the shift segment (1871 windows, ~10 blocks, the only segment large enough to
  judge), every forecast-time predictor gives a coverage-matched width reduction between
  −0.23 and +0.02 against `naive_matched`.
- The predictor selected on an inner fold of the fitting data recovers between −125% and
  +8% of the oracle gap (oracle ≈ 24%), at the 0.90-coverage operating point.
- No model representation beats the simple predictors. All of them, MVDF included, sit
  at −0.02 to +0.01.
- Adding MVDF to the best simple predictor changes the width reduction by less than 0.03.

The 24–29% "recoverable" figure is a property of S_t's marginal spread, not of
predictability:
- Permuting S_t leaves the oracle unchanged in 12/12 cases.
- i.i.d. |N(0,1)| scores, where nothing is predictable, give a 44.1% "reduction".

**B1: oracle property**

| input | oracle reduction |
|---|---|
| Real S_t (ID / shift), 6 runs | 23.6–30.0% / 23.5–25.5% |
| Same, window order permuted | identical (12/12) |
| i.i.d. \|N(0,1)\|, n = 1966 | 44.1% (sd 0.9%) — matches the brief's ~44% |
| i.i.d. Exponential(1) | 46.5% |
| i.i.d. lognormal(σ = 0.4) | 29.0% |

**B2: serial structure.**
- At lags ≥ H, |ACF| ≤ 0.28 for both S and log S.
- The sign flips across segments: E_select is negative at lags H and 2H (−0.05 to −0.17),
  ID is positive (+0.06 to +0.28), and shift is mixed.
- The block-bootstrap bands are about ±0.2–0.3 wide and centred near 0. Blocks of 2H
  partly break lags of H and above, so treat the bands as approximate null bands. No
  estimate clearly exceeds them.

File: `B/task2_acf.csv`.

**B3: main table, shift segment.** Calibrated on the conformalize segment with embargo;
mean over 3 seeds, range in brackets.
- "Red." is the coverage-matched width reduction against `naive_matched`, using the
  normalised construction.
- "Red. @0.90" puts both methods at 0.90 empirical coverage, the oracle's operating point.

| row | ETTh1 Spearman | ETTh1 red. | ETTh1 red. @0.90 | ETTh2 Spearman | ETTh2 red. | ETTh2 red. @0.90 |
|---|---|---|---|---|---|---|
| Constant width (`naive_matched`, cov 0.79 / 0.85) | — | 0 | 0 | — | 0 | 0 |
| **Selected best forecast-time predictor** (inner-fold choice) | −0.13 (roll_q90 ×3) | −0.11 [−0.14, −0.07] | −0.28 | −0.06 … +0.25 (gbm_union ×2, knn) | −0.08 [−0.17, +0.02] | −0.09 [−0.19, +0.02] |
| Best forecast-time predictor *chosen on test* (optimistic) | — | — | −0.016 … −0.005 | — | — | +0.005 … +0.019 |
| last observed score | 0.15 | −0.18 | −0.41 | 0.31 | −0.08 | −0.12 |
| EMA of observed scores | 0.06 | −0.12 | −0.17 | 0.29 | −0.04 | −0.07 |
| input std / all-channel std | 0.17 / 0.41 | −0.13 / −0.00 | −0.29 / −0.02 | 0.18 / 0.31 | −0.04 / +0.01 | −0.09 / −0.03 |
| forecast range / roughness | 0.37 / 0.23 | −0.05 / −0.08 | −0.11 / −0.13 | 0.14 / 0.27 | −0.07 / −0.06 | −0.15 / −0.11 |
| kNN on PCA input | −0.06 | −0.02 | −0.01 | 0.26 | +0.02 | +0.01 |
| GBM on all of the above | 0.18 | −0.04 | −0.04 | −0.07 | −0.14 | −0.17 |
| rep: MVDF | −0.00 | −0.006 | −0.007 | −0.04 | −0.007 | −0.017 |
| rep: correlated / random / PCA / dense | −0.01 / 0.01 / −0.04 / −0.05 | −0.003 / 0.000 / −0.010 / −0.012 | −0.01 / 0.00 / −0.02 / −0.02 | −0.02 / −0.03 / −0.20 / −0.20 | −0.01 / −0.01 / −0.04 / −0.05 | −0.02 / −0.01 / −0.09 / −0.08 |
| best + MVDF, vs best alone | 0.01 vs 0.02 | −0.09 vs −0.09 | −0.14 vs −0.14 | −0.03 vs −0.05 | −0.09 vs −0.09 | −0.11 vs −0.11 |
| **Outcome-binned oracle** (upper bound incl. noise) | — | — | **0.240** | — | — | **0.238** |

Reading the table:
- Moderate rank correlations, such as all-channel input std at ρ = 0.41 on ETTh1 shift,
  do not turn into narrower intervals. The g → S relationship fitted on ID does not
  transfer to shift: skill is negative for every predictor on shift (−0.08 to −2.1).
- The binned constructions (K = 5, 10) produce infinite widths whenever a bin of ~40
  calibration windows cannot certify the coverage level being matched. Where finite,
  they agree with the normalised rows.
- Files: `B/main_emb.csv`, `B/main_noemb.csv`, `B/meta_*.json`, and `B/*_preds.npz`
  (every g_t).

**ID test segment.** With embargo it holds 202 windows (≈1 block of 2H), and
`naive_matched` covers only 0.41–0.48 there. Per-predictor reductions range from −0.4 to
+0.2 and flip with seed. That is unreliable and should not be read as signal. Without the
embargo (394 windows, ≈2 blocks) the picture is the same.

**Verified by test vs inferred**
- Verified: the oracle property, and coverage and widths for every predictor.
- Inferred: "predictable ≈ 0" is a lower bound for these predictor families only.
  A predictor using exogenous information (weather, load schedules) might find more.

---

## C. Why does the constant-width baseline cover ~0.84?

**Answer.** The cause is drift within the E_eval_ID block, made visible by a short
evaluation segment and partly hidden by ACI.
- A truly static band calibrated on the first 80% of ID covers only **0.50–0.58** on the
  last 20%.
- The reported 0.83–0.85 is that band after ACI, which uses each window's miss
  immediately (96 steps before it is observable).
- Assigning random contiguous blocks to calibration and test brings static coverage back
  to 0.86–0.88; chronological assignment gives 0.55–0.63.
- The exchangeable test passes for naive, PID and CQR, so their gap is not a bug in those
  paths.
- It **fails for the learned-scorer path** (`DifficultyScorer`): 0.008 coverage on
  i.i.d. data. That is a real code bug, and it does not affect the ETTh numbers (see
  below).

**C1: anomaly table**

| anomaly | cause | evidence | corrected value |
|---|---|---|---|
| `naive_matched` 0.836 (baseline run) vs 0.895 (scorer run) | **Reporting.** The two figures average different dataset sets. | Per (dataset, seed) the runs differ by ≤ 0.005; the residual is from the Kaggle vs HTTP data source. 0.895 includes traffic (1.000) and electricity (0.907). ETTh1/ETTh2 mean is 0.836 in both runs. | n/a |
| `naive_matched` ETTh ID 0.827–0.845 | **Drift**, plus ACI non-causality and a short segment | See C2–C4 and the figure | Static, consistent profile: 0.55–0.63. ACI with H-delayed feedback: 0.78–0.85. |
| CQR ETTh1 ID 0.71–0.77 | **Drift**, on a static calibration with no online update | Conformity-score q90 rises from 0.47–0.49 (conformalize) to 0.59–0.62 (width-eval). No quantile crossings (0/all cells). Coverage from intervals equals `mean(E ≤ q̂)` exactly. | Consistent profile: 0.70–0.78 (unchanged) |
| CQR ETTh2 shift 0.98–0.99 | **Drift in the other direction**; again no online update | Shift-block conformity q90 is 0.54–0.67 against q̂ = 0.94–1.09. Naive and PID adapt down via ACI; CQR cannot. | — |
| Learned scorer, electricity shift 0.630 | **Code bug identified; whether it caused 0.630 is unverified** (no electricity data locally) | `DifficultyScorer.run` passes `base_q = max(q_resid, 1e-6)` to ACI. ACI clips the **total** half-width `q_resid + pred_q(x)` to `[0, 20·base_q]`, and its correction step is scaled by `base_q`. When `q_resid` is near 0, the ceiling falls below `pred_q` and adaptation stalls. On i.i.d. data: coverage 0.008 (100 reps), q_resid ≤ 0 in 38% of reps; signed q_resid with no clip gives 0.902. | On local ETTh proxies, q_resid = 0.24–0.65, the ceiling is 4.8–13, and it binds on 0% of windows, so coverage is identical before and after. The scorer run itself (Kaggle data, 20k-step SAE) could not be re-run here. |
| Traffic ID 1.000, every mode | **Unexplained here** (data not local) | Quantile arithmetic is excluded: n_conf = 350 gives rank ⌈351·0.9⌉ = 316 ≤ 350, so q is finite and not the max. The pattern (all modes, including naive) matches the same drift mechanism with the conformalize block harder than width-eval, as the earlier `FINISHING_TASKS_STATUS.md` diagnosis says. Not tested. | — |
| Shift coverage 0.89–0.90 for every ACI-wrapped method | **ACI immediate feedback over 1871 steps** | Static band on shift: 0.77–0.93. ACI with H-delayed feedback: 0.87–0.90. | Delayed ACI: 0.87–0.90 |

**C2: coverage decomposition, `naive_matched` path, width-eval (394) / shift (1871)**

| run | reported (ACI) | block-bootstrap 95% CI (2 blocks!) | n_eff / n | static, original per-block profiles | static, E_select profile | static, calibration profile | ACI, calibration profile | ACI, H-delayed feedback |
|---|---|---|---|---|---|---|---|---|
| ETTh1 s0 | 0.830 / 0.894 | [0.77, 0.93] / [0.85, 0.94] | 75 / 107 | 0.581 / 0.793 | 0.632 / 0.765 | 0.629 / 0.779 | 0.832 / 0.895 | 0.784 / 0.898 |
| ETTh1 s1 | 0.832 / 0.894 | [0.79, 0.94] / [0.85, 0.94] | 75 / 113 | 0.533 / 0.778 | 0.609 / 0.781 | 0.599 / 0.763 | 0.838 / 0.895 | 0.850 / 0.893 |
| ETTh1 s2 | 0.827 / 0.893 | [0.76, 0.94] / [0.85, 0.94] | 92 / 116 | 0.541 / 0.789 | 0.629 / 0.800 | 0.622 / 0.788 | 0.835 / 0.893 | 0.815 / 0.893 |
| ETTh2 s0 | 0.845 / 0.898 | [0.80, 0.96] / [0.88, 0.93] | 47 / 249 | 0.503 / 0.873 | 0.548 / 0.885 | 0.548 / 0.884 | 0.858 / 0.898 | 0.840 / 0.893 |
| ETTh2 s1 | 0.840 / 0.898 | [0.81, 0.95] / [0.88, 0.93] | 65 / 296 | 0.510 / 0.887 | 0.586 / 0.897 | 0.561 / 0.886 | 0.858 / 0.899 | 0.848 / 0.872 |
| ETTh2 s2 | 0.843 / 0.898 | [0.80, 0.95] / [0.88, 0.93] | 54 / 278 | 0.530 / 0.920 | 0.589 / 0.927 | 0.566 / 0.918 | 0.860 / 0.898 | 0.840 / 0.873 |

The reported column reproduces `AGG_baselines.csv` exactly. PID reproduces exactly too
(0.812–0.848 ID).

**C3: sampling noise.** n_eff for the width-eval coverage series is 47–92 out of 394,
and the segment spans about 2 blocks. The block-bootstrap CIs on the reported
(ACI-adapted) coverage include 0.90, so with ACI the 0.84 cannot be distinguished from
0.90. The underlying static miss (0.50–0.63) is not noise: random contiguous blocks give
0.86–0.88 (5–95%: 0.44–1.00).

**C4: drift.** Within ID, the conformalize → width-eval q90 of S_t rises 0.84–0.86 → 1.14–1.15
(ETTh1) and 1.67–1.69 → 1.94–1.99 (ETTh2). A two-sample KS test with a block-permutation
null gives p = 0.20–0.49. With only 6 blocks it has almost no power, which is itself the
finding: this split cannot detect drift of this size. `figures/C_rolling_coverage.png`
shows rolling coverage of a fixed band swinging between 0.2 and 1.0 in episodes a few
hundred windows long; the width-eval block sits in a hard episode on both datasets.

**C5: other checks**
- **Quantile.** `empirical_quantile` returns the ⌈(n+1)(1−α)⌉-th order statistic, which
  is correct. It clips the rank to n (returning the max instead of +∞) only when n < 9,
  which is not reachable in these runs. (`C/quantile_arith.json`)
- **ACI.** The update `a ← a + γ(err − α)·base_q` has the correct sign and starts at 0,
  and the clip to [0, 20q] never binds for naive. The parallel `alpha_t` is updated but
  never used (cosmetic). Burn-in is included in reported coverage. err_t is used at
  t + 1, not t + H. Trajectories: `C/*_aci_trace.npz`.
- **Brief vs code.** `naive_matched` is not constant-width in the code: it is wrapped in
  global ACI, so widths vary per window.
- **Exchangeable test** (`C/exchangeable_test.json`, 300 reps; scorer 100):

| path | mean coverage | MC SE |
|---|---|---|
| naive + ACI | 0.8997 | 0.0002 |
| static | 0.9017 | 0.0010 |
| PID | 0.9001 | 0.0003 |
| CQR | 0.9023 | 0.0011 |
| ACI with H-delayed feedback | 0.880 | 0.002 |
| learned scorer | **0.008** | 0.005 |

  The H-delayed ACI undercovers on short streams: the controller keeps shrinking for
  H steps before a miss reaches it.

**Bugs found and their before/after values**
1. **`DifficultyScorer` clip/scale** (`calibration/difficulty_scorer.py::run` →
   `ffdci_head.FFDCIHeadACI`). i.i.d. coverage goes from 0.008 to 0.902 once fixed.
   Local ETTh proxy coverage is unchanged (the clip never binds). Electricity and traffic
   could not be checked. Not fixed in the repo; the corrected variant is in
   `p126/work_c.py::scorer_diag`.
2. **Per-block `profile[h]`.** Calibration and test scores use different profiles, and
   the test profile is built from test outcomes. Static ID coverage moves +0.03 to +0.08
   when fixed, and ACI coverage moves +0.002 to +0.020. See the C2 columns.
3. **ACI immediate feedback** (non-causal by H − 1 steps). Corrected values are in the
   C2 last column.
4. **Documentation mismatches.**
   - The R² caption says "training mean", but the code uses the evaluation-set mean.
   - `PHASE4_RESULT.md` says the oracle's naive uses "the first 80% of E_eval_ID", but
     the code uses the whole block in-sample.

No existing output was overwritten. Corrected numbers live only in `runs/p126/`.

**Verified by test vs inferred**
- Verified: the reconciliation, the exact reproduction of the naive/PID/CQR numbers, the
  exchangeable test, the drift decomposition, and the scorer bug mechanism.
- Inferred: the electricity 0.630 cause and the traffic 1.000 cause.

---

## What could not be done
- **Electricity, traffic, TimeMixer.** No raw data or weights locally (Kaggle-only).
  The brief's priority order put them last.
- **Exact scorer-run reproduction.** That run used Kaggle-bundle ETT and a 20k-step SAE.
  Scorer diagnostics here are a local proxy using the Phase-3 SAE and MVDF.
- **Synthetic positive control.** Excluded at the user's request, so A6 and B6 (metric
  reference values and the true-scale ceiling) were not run.
- **Kendall and ACF intervals.** Kendall has a point estimate only (Spearman carries the
  intervals). The ACF block-bootstrap bands are biased toward 0 at lags ≥ H.

## File index (`runs/p126/`)
- `arrays/` — per-window test arrays per run: inputs, y, ŷ, hidden state, sparse SAE
  codes, CQR quantiles, masks, timestamps, plus a `*_repro.json` reproduction check.
- `A/` — `r2_decomposition.csv`, `rank_metrics.csv`, `blocked_cv.csv`, `summary.json`,
  `feature_level.json`, `*_feature_level.csv`, and `*_arrays.npz` (S_t, activations,
  causal baseline).
- `B/` — `task1_oracle_property.csv`, `task2_acf.csv`, `main_{emb,noemb}.csv`,
  `meta_*.json`, and `*_preds.npz` (every predictor g_t).
- `C/` — `coverage_decomposition.csv`, `per_run.json` (CQR, drift, random-block and
  scorer diagnostics), `reconcile.csv`, `exchangeable_test.json`, `quantile_arith.json`,
  `*_aci_trace.npz`, `*_rolling.npz`.
- `figures/C_rolling_coverage.png`.
- Files suffixed `_ETTh10` in `A/` and `B/` are superseded single-run development
  outputs; B's version predates the @0.90 columns.
- Code: `p126/{regen_arrays,common,work_a,work_b,work_c}.py`.
