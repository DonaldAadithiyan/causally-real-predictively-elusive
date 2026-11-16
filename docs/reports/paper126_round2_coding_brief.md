# Coding brief, round two: remaining analyses for the difficulty-feature study (Paper 126)

## 1. Purpose

Round one (results in `mechanistic-conformal/runs/p126/RESULTS.md`) tested three doubtful claims in the paper and found that its central story does not hold as written. This round does all the remaining coding work needed to find out what the paper can honestly claim. It has three aims:

1. Correct the evaluation code and regenerate every affected number.
2. Test the paper's ideas in settings where difficulty is actually predictable, so that a null result means something.
3. Put the causal feature-selection result on firm footing with proper null comparisons.

You are not editing the manuscript. No outcome is preferred. A result that weakens the paper is as useful as one that strengthens it, and mixed results should be reported as mixed.

This brief is self-contained, but read `runs/p126/RESULTS.md` first: it holds the orientation facts (horizon, lookback, split boundaries, where each number is computed) that this round depends on.

## 2. Where things stand after round one

Round one covered ETTh1 and ETTh2 on iTransformer, three seeds.

- **Reproduction.** Forecaster, sparse autoencoder (SAE) and CQR model were retrained from original configs and seeds and match saved metrics exactly, so the saved MVDF feature indices still apply. This holds only with 4 CPU threads. Per-window arrays are saved.
- **Predictability metric.** The paper's strongly negative R² came from a shift in the level of the difficulty score between fitting and evaluation data (mean 0.52 to 0.73 on ETTh1, 0.95 to 1.34 on ETTh2). Against a constant predictor, every estimator scores between -0.013 and +0.015. Rank metrics and blocked cross-validation find nothing for the MVDF features.
- **One exception.** On ETTh2 the full hidden state and its PCA projection carry weak rank signal (0.14 to 0.25) that the SAE feature subsets lack.
- **Oracle.** The paper's "recoverable width" oracle gives the same answer on time-shuffled scores and a larger one (44%) on pure noise. No forecast-time predictor beat the constant-width baseline on the shift segment.
- **Coverage.** Low coverage is mostly drift inside the in-distribution block. A fixed band calibrated on the first 80% covers 0.50 to 0.58 on the last 20%; random contiguous-block assignment gives 0.86 to 0.88.
- **Look-ahead.** The adaptive (ACI) update uses each window's outcome 96 steps before it could be known. A version that waits covers 0.78 to 0.85.
- **Scorer bug.** The learned-scorer path (`DifficultyScorer` feeding `FFDCIHeadACI`) caps the interval half-width at `20 * max(q_resid, 1e-6)`. On structureless data it covers 0.008 instead of 0.90; without the cap, 0.902.
- **Normalisation.** Each block's score is normalised with that block's own errors, so calibration and test scores are on different scales.
- **Not done.** Electricity, traffic and TimeMixer (data and weights only on Kaggle); the synthetic control (skipped on request, now wanted).

## 3. Definitions

- **Difficulty score.** `S_t = max over h of |y[t,h] - yhat[t,h]| / profile[h]`, where `profile[h]` is the mean absolute calibration error at horizon step h, rescaled to mean 1.
- **Dose-response.** For SAE feature j, its activation is scaled by lambda in {0, 0.25, 0.5, 0.75, 1.0, 1.25} and patched back into the forecaster. `D_pred(lambda)` is the mean forecast loss and `D_unc(lambda)` the mean S_t. `P_j` and `U_j` are their least-squares slopes over lambda.
- **Specificity.** `R_j = U_j - E[U_j | P_j]`, with the conditional mean fitted by isotonic regression over the candidate pool.
- **Feature types.** Median split on P_j and U_j: A (high P, low U), B (low P, high U), C (both high), D (both low).
- **MVDF set.** Features that pass a threshold on the size and steadiness of U_j, then the top k by R_j. "Type-B enrichment" is the share of type B in the selected set divided by its share in the pool. The paper reports 2.0x to 4.6x.
- **Representations compared at matched size k.** `mvdf`, `correlated` (correlation-selected SAE features), `random_k` (random SAE subset), `pca_k`, and `dense` (the raw 256-dimensional hidden state).
- **Baselines.** `naive_matched` is a constant-width conformal band calibrated on the same amount of data as the learned scorers; `naive_no_scorer` uses a smaller standard split. Both currently run with the adaptive update.

## 4. Ground rules

- **Selection procedure is frozen.** Run the existing MVDF selection code unchanged, including on new datasets. Null comparisons in section 9 wrap it; they do not modify it.
- **Corrections go alongside originals.** Put corrected code behind new flags or new functions. Keep every original output. Write this round's results to a new directory (suggested: `runs/p126_round2/`).
- **Persist everything.** Save model weights, SAE weights, per-window arrays and per-feature dose-response curves for every run. Round one had to retrain because none of this existed.
- **Pin the environment.** 4 CPU threads for anything that must match existing ETTh artifacts. Record library versions and seeds.
- **No selection on the reporting split.** Choices among predictors, hyperparameters or datasets are made on validation folds inside fitting data, or by the predeclared rules in section 13.
- **Report what you find.** Do not vary metrics or settings until a preferred answer appears.
- **Ignore leftovers.** Files ending in `_ETTh10` under `runs/p126/A/` and `runs/p126/B/` are superseded test outputs. Do not read them as results.

## 5. Shared statistical machinery

Same as round one; reuse that code.

- Moving-block bootstrap over time (block length 2H, sensitivity at H and 4H, at least 2000 resamples, percentile 95% intervals).
- Circular time-shift nulls instead of independent shuffles.
- An embargo of at least L + H windows between fit, calibration and test segments.
- Width comparisons at matched empirical coverage.
- Effective sample size reported beside every coverage figure.

One addition: **like-for-like baselines.** A static method is compared with a static constant-width band. A method wrapped in an adaptive update is compared with a constant-width band using the same update. Never compare a delay-respecting method with a look-ahead baseline.

## 6. Work package 0: corrected evaluation path (prerequisite)

Everything later depends on this.

1. **Delayed feedback.** In every online update (ACI and PID, for all methods including both naive baselines), the update at time t may use only windows whose full horizon has already been observed.
2. **Remove the half-width cap** in the scorer path, or make it a flag that defaults to off.
3. **Consistent normalisation.** Compute `profile[h]` on calibration data only and apply it unchanged to test blocks.
4. **Sanity test.** Rerun round one's exchangeable synthetic test on the corrected path. Every method, including the learned scorer, must average 0.90 coverage within Monte Carlo error.
5. **Baseline parity audit of round one's item B.** Establish which baseline those width reductions were measured against. Recompute them like-for-like, on both the in-distribution and shift segments (the round-one summary quoted only shift). State whether the conclusion "nothing forecast-time beats constant width" changes.
6. **Three evaluation protocols.** Because the in-distribution block drifts, every coverage and width table from here on is reported three ways: chronological static, chronological with delayed adaptive update, and random contiguous-block assignment.

## 7. Work package 1: regenerate the paper's tables on ETTh1 and ETTh2

Using the corrected path, regenerate each of these and give original and corrected values side by side:

- Continuous scoring: coverage and width for `naive_matched`, `naive_no_scorer`, `dense`, `mvdf`, `correlated`, `random_k`, `pca_k`, in-distribution and shift.
- Standard baselines: `naive_matched`, PID-conformal, CQR. Run CQR both static and with a delayed adaptive wrapper so it has a like-for-like comparison.
- Regime gating: the ratio of the 0.9 quantile of S_t between windows in the top 10% of combined type-B activation and the rest.
- Localization: local variance of S_t among nearest neighbours in each representation, relative to dense, with the plus-or-minus 40 window exclusion.
- Round one's item A metrics, to confirm the normalisation fix does not change them.

## 8. Work package 2: synthetic control

The paper's synthetic setup is a regime-switching autoregressive process with a hidden two-state regime that sets the noise scale. It is the only setting where the true amount of predictable difficulty is known. Rerun it with arrays saved, on the corrected path.

1. **Reference values.** Compute round one's item A metrics (R² decomposition, rank correlations, top-decile AUROC) here, so each has a value for "signal present".
2. **True ceiling.** Compute the width reduction from binning on the true regime or true noise scale. Compute the outcome-sorted oracle. Check that the forecast-time predictors from round one's item B land near the true ceiling and not near the oracle.
3. **Downstream link.** The paper found that a learned scorer beat the constant baseline on only 1 of 3 synthetic seeds, with a coverage failure on one, and blamed the scorer method. Rerun with the cap removed and delayed feedback. Report whether the failure persists.
4. **Dial experiment.** Vary the ratio of noise scales between the two regimes from 1.0 (no structure) upward in about five steps, and vary regime persistence at two or three levels. At each setting record: true ceiling, oracle value, best forecast-time predictor's width reduction, rank metrics for the MVDF set, and type-B enrichment. This shows how much predictable difficulty must exist before the pipeline detects it, which turns the real-data null into a bound. The ratio 1.0 setting is a negative control for section 9.

## 9. Work package 3: nulls and diagnostics for the causal selection

The concern: selecting on high U and high R, then counting features with low P and high U, may produce "enrichment" by construction, and the slopes themselves have never been compared with a null.

1. **Persist curves.** For every candidate feature save D_pred and D_unc at all six lambda values, both slopes, their R², a monotonicity measure, and block-bootstrap standard errors.
2. **Matched random-direction null.** For each feature, replace its decoder direction with random unit directions (at least 50) while keeping the same activation pattern and magnitudes, and run the same dose-response. Report each feature's U_j and R_j as a z value against this null.
3. **Selection null.** Permute U_j across features while holding P_j fixed, then rerun the unchanged selection and typing. Repeat at least 1000 times to get the distribution of type-B enrichment that selection alone produces. Compare the observed enrichment with it. Do the same using the random-direction values from task 2 in place of real U_j.
4. **Split-half reliability.** Estimate P_j, U_j and R_j on two disjoint contiguous halves of the feature-selection block. Report the rank correlation of each across halves and the overlap of the two top-k sets against chance.
5. **Out-of-sample effect.** Select on one half; measure the selected set's U_j on the other half and on the evaluation block, against random features matched for firing rate. This checks whether the causal effect survives outside the data used to select it.
6. **Effect size.** Express U_j in units of the standard deviation of S_t on active windows.
7. **Necessity rerun.** Zero the whole MVDF set and measure the change in S_t against a random feature set matched for firing rate and a random direction matched for size, with arrays saved and block-bootstrap intervals.

Run tasks 2 to 5 for each dataset and seed, and on the synthetic ratio 1.0 setting.

## 10. Work package 4: where the hidden-state signal lives, and whether it controls anything

This follows the one positive lead from round one. Start on ETTh2, repeat on ETTh1, then on any dataset from section 12 that shows predictable difficulty.

1. **Confirm the signal.** Rank metrics for the dense hidden state under blocked cross-validation with intervals and a circular-shift null. Then the partial rank correlation after controlling for the simple forecast-time predictors (recent scores, input volatility), to check it is not just tracking the recent level.
2. **Locate the loss.** The same metrics on: the raw hidden state; the SAE reconstruction; the SAE residual; all 2048 SAE activations with a regularised probe; the MVDF subset; and a set of k SAE features chosen by a sparse predictive probe.
3. **Probe direction.** Fit a linear probe for S_t on the hidden state using fitting data only. Normalise its weights to a unit direction w.
4. **Intervene on the probe direction.** Steer the hidden state by lambda times the standard deviation of its projection on w, for lambda in {-2, -1, 0, 1, 2}, and separately project w out. Measure the change in S_t and in forecast loss.
5. **Nulls and a positive control.** Compare with at least 50 random directions of matched norm, and with a second null of directions matched for projected variance. Include a positive control: a direction expected to move S_t at this scale (the top principal component, and the decoder direction of the MVDF feature with the largest U_j). If nothing at this scale moves S_t, the test is uninformative and must be reported as such.
6. **Same protocol for the selected features.** Apply the identical steering protocol to MVDF decoder directions and to the probe-selected SAE features.
7. **Deliver a grid.** Rows: MVDF features, probe direction, probe-selected SAE features. Columns: predicts S_t (rank metrics with intervals), controls S_t (z against both nulls). Report whatever pattern appears.

## 11. Work package 5: activation history

The paper tested only the activation at a single moment. Activations are functions of the input, so their history is available at forecast time with no delay.

For `mvdf`, `correlated` and `random_k`, build features from the trailing history: lagged activations, trailing mean and maximum over windows of roughly H/4, H and 2H, counts of active windows, and time since last activation. Evaluate with the round-one item A metrics under blocked cross-validation. Report the gain over single-moment activations and the incremental value over the simple forecast-time predictors.

## 12. Work package 6: real data where difficulty may be predictable (Kaggle)

If you cannot run on Kaggle yourself, prepare one entry-point script or notebook per task that saves weights and per-window arrays in the same format as the local runs, and state exactly which output files need to be brought back.

1. **Electricity and traffic.** Regenerate forecasts, SAE and MVDF sets with everything saved. Run the corrected tables from section 7 and round one's items A, B and C. Report whether the cap explains the electricity 0.630 coverage and what causes traffic's 1.000.
2. **Step zero on each dataset.** Before any feature analysis, test whether difficulty is predictable from forecast-time information (hour of day, day of week, recent scores, input volatility) using round one's item B machinery and the rule in section 13.
3. **Screen further datasets if needed.** If fewer than two datasets pass step zero, screen candidates with stronger expected structure. Prefer datasets the repository can already load. Otherwise consider the standard long-horizon benchmarks with daily cycles or volatility clustering (for example solar generation, weather, exchange rates, ETTm1 and ETTm2). Screen with one iTransformer seed. Report step-zero results for every dataset screened, not only those that pass.
4. **Full pipeline on datasets that pass.** Three seeds of iTransformer, SAE, unchanged MVDF selection, then sections 7, 9, 10 and 11, plus the incremental value of each representation over the simple predictors.
5. **TimeMixer.** Rerun the paper's three architecture checks (type-B enrichment, scorer width gap, predictability) on ETTh1 and ETTh2 with the corrected path and round one's item A metrics. Lowest priority.

## 13. Predeclared decision rules

- **"Predictable difficulty present" on a dataset:** the best forecast-time predictor, chosen on validation folds, has a like-for-like coverage-matched width reduction whose block-bootstrap 95% interval lies above zero on the test segment, and a rank correlation whose interval lies above zero, in at least 2 of 3 seeds (or the single seed, when screening).
- **"Enrichment exceeds selection alone":** observed type-B enrichment lies above the 95th percentile of the selection null in section 9, task 3.
- **"Direction controls S_t":** absolute z above 2 against both nulls in section 10, in at least 2 of 3 seeds, with the positive control also clearing its null.

Apply these as written. If a rule turns out to be unworkable, say why and report the raw numbers.

## 14. Order of work

1. Section 6 (corrected path), then section 7 (regenerated tables).
2. Section 8 (synthetic control).
3. Section 9 (selection nulls).
4. Section 10 (hidden-state signal and interventions).
5. Section 11 (activation history).
6. Section 12 (Kaggle), prepared in parallel once section 6 is done.

If time or compute runs out, stop at a package boundary and say where.

## 15. What to hand back

A `RESULTS.md` in the new results directory, plus the tables, figures, weights and raw arrays behind it. It should contain:

- **A numbers ledger.** One row per number the paper reports that this work touches: what it measures, original value, corrected value, and status (unchanged, changed, not reproducible).
- **Per work package:** the answer in a short paragraph with supporting numbers, the main table, what was verified by test versus inferred, and anything not done and why.
- **Step-zero results** for every dataset tried.
- **The grid** from section 10.
- **Every code change** made for the corrected path, with file and function names.

## 16. Out of scope

- Editing the manuscript or making paper-ready figures.
- Changing the MVDF selection procedure, the dose-response settings, or forecaster and SAE hyperparameters.
- New conformal methods beyond those already in the repository and the simple constructions from round one.
