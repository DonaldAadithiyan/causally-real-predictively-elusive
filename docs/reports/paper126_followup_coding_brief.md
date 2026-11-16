# Coding brief: three follow-up analyses for the difficulty-feature study (Paper 126)

## 1. Purpose and scope

This repository backs a research paper about a neural time-series forecaster. The paper makes three claims that the current code does not yet support well enough. Your job is to write and run the analyses that settle them, and to report what you find. You are not editing the manuscript, and no particular outcome is wanted: a result that contradicts the paper is as useful as one that confirms it.

The three questions:

- **A.** The paper reports that certain internal features carry no predictive information about forecast difficulty, based on strongly negative held-out R². Is that a real null, or an artifact of the metric?
- **B.** The paper reports that 24.5 to 29.2 percent of interval width is "recoverable", based on an oracle. That oracle cannot distinguish predictable difficulty from irreducible noise. How much difficulty is actually predictable from information available at forecast time?
- **C.** The constant-width conformal baseline covers about 0.84 against a 0.90 target on datasets described as clean, and other methods show coverage as low as 0.63 and as high as 1.00. Why?

I have read the paper but not this codebase, so everything below is described in the paper's terms. Part of the work is finding where each thing lives in the code.

## 2. Background

### 2.1 Setup as the paper describes it

- **Forecaster:** iTransformer (main) and TimeMixer (second architecture), multi-horizon point forecasts.
- **Datasets:** ETTh1, ETTh2, electricity, traffic. Most later-stage results use only ETTh1 and ETTh2. Three seeds each.
- **Splits:** chronological training block, feature-selection block, evaluation block. The evaluation block is divided into an in-distribution (ID) part and a later shifted part. The ID part has 1966 windows for ETTh1 and ETTh2, 2334 for electricity, 1169 for traffic.
- **Sparse autoencoder (SAE):** TopK, k = 64, dictionary size 2048, trained on the 256-dimensional hidden state just before the output head.
- **MVDF set:** a small set of SAE features selected by a causal dose-response procedure. Treat this set as frozen input. These features are active on roughly 3.7 percent of windows.
- **Target coverage:** 0.90 (alpha = 0.10).

### 2.2 The difficulty score

For window t and horizon step h:

```
S_t = max over h of  |y[t,h] - yhat[t,h]| / profile[h]
```

`profile[h]` is the mean absolute calibration-set error at horizon h, rescaled so its mean over all horizons is 1. S_t is therefore the smallest multiplier q such that the band `yhat[t,h] +/- q * profile[h]` covers every horizon step of window t. The paper does not say how multiple channels are aggregated; you need to find out.

### 2.3 Numbers currently reported

| Item | Reported value |
|---|---|
| Best held-out R² predicting S_t from natural MVDF activations, all windows | ETTh1: -0.72 to -0.58; ETTh2: -0.92 to -0.89 |
| Same, restricted to windows where an MVDF feature is active | ETTh1: -0.70 to -0.48; ETTh2: -0.89 to -0.80 |
| Same test on the synthetic positive control | +0.26 to +0.48 |
| Oracle width reduction vs. constant-width baseline | ETTh1: 24.5% (ID), 25.4% (shift); ETTh2: 29.2% (ID), 24.5% (shift) |
| `naive_matched` ID coverage, mean of ETTh1 and ETTh2, in the baseline-comparison run | 0.836 |
| `naive_matched` ID coverage, mean over four datasets, in the scorer run | 0.895 (traffic contributes 1.000) |
| CQR coverage | ETTh1 ID: 0.71 to 0.77; ETTh2 shift: 0.98 to 0.99 |
| Learned scorer on MVDF features, electricity, shift | as low as 0.630 on one seed |
| Traffic ID coverage | 1.000 for every mode and seed |

The R² estimators are linear regression, isotonic regression, gradient boosting, and a small pinball-loss network. The table caption says negative R² means "worse than predicting the training mean".

`naive_matched` is a constant-width split-conformal band calibrated on the same amount of data the learned scorers use. The scorers use a chronological 50% scorer-train / 30% conformalize / 20% width-eval split of the ID evaluation block. Online methods use an adaptive conformal (ACI) update with learning rate 0.03.

## 3. Ground rules

- **Frozen artifacts.** Do not retrain or modify the forecaster, the SAE, or the MVDF selection. Reuse cached forecasts, hidden states and activations where they exist. If something must be regenerated, use the original config and seeds and say so in the report.
- **No selection on the reporting split.** Any choice among models, hyperparameters or predictors is made on folds inside the fitting data. The segment used for reported numbers is only evaluated on.
- **Keep old results.** If you find a bug that changes existing numbers, write corrected outputs alongside the originals. Do not overwrite.
- **Persist raw arrays.** The earlier pipeline saved only summary statistics, which is why some questions cannot be answered retroactively. For every run here, save per-window arrays (S_t, predictions, activations, covered/not-covered indicators, interval widths, timestamps or window indices) so later analyses need no rerun.
- **Report what you find.** Do not swap metrics, splits or settings until a preferred answer appears. If results are mixed, report them as mixed.
- **Priority order.** ETTh1 and ETTh2 on iTransformer, three seeds, first. Extend to electricity, traffic and TimeMixer only after A, B and C are complete, and only where cached artifacts make it cheap.

## 4. Step 0: orientation

Before writing analysis code, establish these facts and write them to a short notes file. Several of them decide how the later steps must be done.

1. Forecast horizon H, lookback length L, window stride, number of channels per dataset.
2. How S_t aggregates channels (max, mean, per-channel) and whether calibration and evaluation do it the same way.
3. What index t refers to (forecast origin or something else) and when S_t becomes observable. With a horizon of H, the score of window t is fully known only H steps later.
4. Exact boundaries of every split and sub-split, as window indices, per dataset.
5. Whether adjacent segments are separated by a gap. Windows at stride 1 overlap heavily, so windows on either side of a boundary can share target values.
6. Where each reported number in section 2.3 is computed: the R² table, the oracle, and each coverage figure.
7. Which split `profile[h]` is computed on.

## 5. Shared statistical machinery

Windows overlap, so S_t is strongly autocorrelated. Standard i.i.d. tools give misleadingly narrow intervals here. Use these throughout.

- **Block bootstrap.** Moving-block bootstrap over time, block length 2H by default, with H and 4H as sensitivity checks. At least 2000 resamples, percentile 95% intervals. If a segment holds fewer than about five blocks, say so explicitly; the interval is then unreliable, and that is itself a finding.
- **Permutation nulls.** Use circular time shifts (shift one series against the other by a random offset larger than L + H). Do not shuffle windows independently.
- **Embargo.** Any new fit/calibrate/test split leaves a gap of at least L + H windows between segments. Record the resulting sample sizes.
- **Coverage-matched width.** Widths are only comparable at equal coverage. For each method, report (i) width and coverage at nominal alpha = 0.10, and (ii) width at the alpha that makes its empirical test coverage equal to the baseline's, found by scanning an alpha grid. Width reduction is `(W_baseline - W_method) / W_baseline` using (ii).
- **Effective sample size.** For any 0/1 coverage series, estimate `n_eff = n / (1 + 2 * sum of autocorrelations)` and report it next to the coverage.

## 6. Work item A: is the negative R² a real null?

### 6.1 Why this is in doubt

Per-feature R² values sit in a narrow band that depends on the dataset and hardly at all on the feature (about -0.6 to -0.7 on ETTh1, -0.85 to -0.95 on ETTh2). A linear model on features that are zero on about 96 percent of windows predicts roughly its intercept almost everywhere, so it should not be much worse than a constant unless the constant itself is wrong. This pattern is what a shift in the level of S_t between the fitting data and the evaluation data would produce. If so, the R² reflects that shift and says nothing about the features.

This is a suspicion to test, not a conclusion.

### 6.2 Tasks

1. **Pin down the existing computation.** Which windows the estimators are fit on, which they are evaluated on, and which mean is in the R² denominator. Note that scikit-learn's `r2_score` uses the evaluation-set mean, which differs from what the caption says.

2. **Decompose the R².** For each dataset, seed, estimator, and window subset (all windows; active windows only), compute:

   ```
   R2_test   = 1 - SSE_model / sum((y - mean(y_eval))^2)
   R2_train  = 1 - SSE_model / sum((y - mean(y_fit))^2)
   R2_const  = R2_test of the constant predictor mean(y_fit)
   skill     = 1 - MSE_model / MSE_const
   ```

   Also report mean, median, standard deviation and the 0.5/0.9 quantiles of S_t on the fitting and evaluation segments, and the share of total squared error that comes from active versus inactive windows. If `R2_const` is close to the reported values, the level shift explains them.

3. **Shift-robust metrics.** For each estimator's held-out predictions, and for each single MVDF feature's raw activation:
   - Spearman and Kendall correlation with S_t.
   - AUROC for detecting windows in the top decile of S_t, with the decile threshold taken from the evaluation segment, and separately from the fitting segment.
   - Difference in median S_t and in the 0.9 quantile of S_t between active and inactive windows.

   Compute each on all windows and on active windows only. Activations are tied at zero on inactive windows, so use tie-aware estimators. Attach block-bootstrap intervals and a circular-shift null.

4. **Shift-controlled refits.** Rerun the four estimators under:
   - blocked K-fold cross-validation inside the ID evaluation segment, with embargo;
   - a log-transformed target;
   - a target with a causal baseline removed (for example a trailing mean of scores that were already observable at forecast time, respecting the H-step delay).

   Report whether R² moves toward zero, stays strongly negative, or turns positive.

5. **Comparison conditions.** Run the same metrics for the other representations the paper uses at matched feature budget: correlation-selected SAE features, a random SAE subset, a PCA projection, and the raw dense hidden state. If nothing, including the dense state, shows rank signal, the null is a property of the forecaster and data, not of the MVDF set.

6. **Calibrate on the synthetic control.** Run tasks 2 and 3 on the synthetic positive control, where a signal is known to exist, so each metric has a reference value for "signal present".

7. **Feature-level relationship.** The paper correlates each feature's causal specificity score R_j with its best R². Recompute this using each feature's absolute Spearman correlation and its AUROC as the predictiveness score, with a bootstrap interval over features.

### 6.3 What settles the question

State which of these holds, per dataset:

- The constant predictor reproduces the negative R², and rank metrics are indistinguishable from the null. The null stands; the original metric was uninformative.
- Rank metrics are clearly non-zero. The reported null was an artifact.
- Mixed across seeds, datasets or subsets. Report exactly where.

## 7. Work item B: a ceiling for predictable difficulty

### 7.1 What is wrong with the existing oracle

The existing oracle sorts windows into ten bins by their own realized S_t and sets each window's width to the 0.9 quantile of the other windows in its bin. It conditions on the outcome. Its output depends only on the marginal distribution of S_t, so it reports a large reduction even when nothing is predictable.

### 7.2 Tasks

1. **Demonstrate the property.** Run the existing oracle unchanged on:
   - the real S_t with window order randomly permuted (the result should be identical, since only the marginal distribution matters);
   - i.i.d. synthetic scores with no predictable structure, for example absolute values of standard normal draws. A rough calculation gives a reduction near 44 percent for this case. Treat a large departure as a sign the procedure differs from the description above, and report what you get.

2. **Measure serial structure.** Autocorrelation of S_t and of log S_t at lags from H upward, with block-bootstrap bands, per dataset and segment. Lags below H are not usable at forecast time.

3. **Build difficulty predictors from forecast-time information only.** Each produces a predicted difficulty g_t for window t:
   - **Past scores:** the most recent fully observed score, an exponential moving average of observed scores, a rolling quantile. All must respect the H-step observation delay.
   - **Input statistics:** volatility of the input window and of its first differences, range, level, hour of day and day of week where timestamps exist, magnitude and roughness of the point forecast.
   - **Nearest neighbours:** k-nearest-neighbour regression on a PCA-reduced input window, excluding temporal neighbours within L + H.
   - **Flexible model:** gradient boosting on the union of the above.
   - **Model representations:** the dense hidden state, its PCA projection, and each SAE feature set (MVDF, correlation-selected, random).
   - **Combination:** the best non-representation predictor with MVDF activations added, to measure incremental value.

4. **Turn each predictor into intervals.** Use the same chronological 50/30/20 split as the existing scorers, with embargo, so results are comparable with the existing tables.
   - **Binned:** bin the calibration segment by g into K bins (K = 5 and K = 10), take the per-bin 0.9 quantile of S on calibration, and apply it to test windows by their bin. Bin edges come from calibration only.
   - **Normalised:** conformalize the ratio S_t / g_t with a single quantile q; the width multiplier for window t is q * g_t.

5. **Report for every predictor:** Spearman correlation with S_t, `skill` as defined in 6.2, coverage, coverage-matched width reduction against `naive_matched`, and block-bootstrap intervals. If one predictor is singled out as "best", choose it on a validation fold inside the fitting data.

6. **Validate on the synthetic control.** There the true noise scale per regime is known. Compute the true ceiling by binning on the true regime or true scale, compute the outcome-binned oracle, and check that the predictors in task 3 land near the true ceiling and not near the oracle.

### 7.3 What to deliver

A table per dataset and segment with three rows of reference and one row per predictor:

| Row | Meaning |
|---|---|
| Constant width | 0% by definition |
| Best forecast-time predictor | Estimated predictable reduction. This is a lower bound: a better predictor might find more. |
| Outcome-binned oracle | Upper bound that includes irreducible noise |
| Each representation | How much of the predictable part it captures |

The key number is the share of the oracle gap that the best forecast-time predictor recovers, and whether any model representation matches or beats the simple past-score and input-statistic predictors.

## 8. Work item C: explain the coverage gap

### 8.1 Tasks, cheapest first

1. **Reconcile the two runs.** `naive_matched` should give identical coverage for the same dataset, seed and split in the scorer run and in the baseline-comparison run. Tabulate both per dataset and seed. If they differ, find the difference in code path, split or score definition.

2. **Exchangeable sanity test.** Feed the full conformal evaluation path with i.i.d. synthetic scores, and separately with synthetic forecasts and targets whose errors are i.i.d. Repeat a few hundred times. Coverage must average 0.90 within Monte Carlo error for every method. A failure here is a code bug, independent of the data.

3. **Quantile arithmetic.** Confirm the calibration quantile uses the finite-sample level `ceil((n + 1) * (1 - alpha)) / n` with a "higher" interpolation rule, and check for off-by-one errors.

4. **Score and coverage consistency.** Reported coverage computed from intervals must equal `mean(S_test <= q)` exactly for the constant-width band. Check that calibration and evaluation agree on: joint-over-horizon versus per-step coverage, channel aggregation, and the split `profile[h]` comes from.

5. **Sampling noise.** The width-eval segment is about 20 percent of roughly 2000 overlapping windows. Compute effective sample size and a block-bootstrap interval for each coverage figure. If the interval comfortably contains 0.90, the gap is noise. Do not use binomial intervals.

6. **Temporal drift.** Compare the S_t distribution between the conformalize and width-eval segments (quantiles, two-sample test with a block-aware null). Plot rolling coverage over time. Rerun the constant-width band with segments assigned to calibration and test in randomly chosen contiguous blocks instead of chronologically. If coverage returns to about 0.90, the cause is drift within the block, not a bug.

7. **Adaptive update.** For every method wrapped in ACI or PID, log the trajectory of the working alpha and the quantile. Check: the sign and form of the update, the starting value, clipping at 0 and 1, whether the update uses only errors already observable at that time (H-step delay), whether burn-in steps are included in reported coverage, and how infinite or zero widths from clipping enter the mean width. Trace the electricity seed that reached 0.630.

8. **CQR.** Check the conformity score `max(lower - y, y - upper)`, which split it is calibrated on, quantile crossing, and whether calibration and reported coverage use the same joint-over-horizon definition.

9. **Traffic at 1.000.** Check whether the calibration quantile hits the maximum score or infinity because the segment is short, and whether `profile[h]` is degenerate.

### 8.2 What to deliver

A table with one row per anomaly from section 2.3 and columns: cause (code bug, drift, sampling noise, unexplained), evidence, and corrected value if a bug was fixed. If a bug is found, rerun every table it touches and report before and after.

## 9. What to hand back

A single `RESULTS.md` plus the tables, figures and raw arrays it draws on. For each of A, B and C:

- the answer in one short paragraph, stated plainly, with the numbers that support it;
- the main table;
- what was verified by a test versus what is inferred;
- anything that could not be done, and why;
- any bug found, with before and after values for affected numbers.

Start the file with the orientation facts from Step 0.

## 10. Out of scope

- Editing the manuscript or producing paper-ready figures.
- Retraining forecasters or SAEs with different settings.
- Changing or re-deriving the MVDF selection procedure.
- New conformal methods beyond the simple binned and normalised constructions in section 7.
