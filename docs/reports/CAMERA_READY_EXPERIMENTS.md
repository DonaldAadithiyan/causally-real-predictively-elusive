# Camera-ready experiments — ICAART 2027 paper #126

> **Redacted copy.** This is the task specification given to the coding agent for the
> camera-ready experiments (E0–E6). Verbatim reviewer text has been replaced by short
> paraphrases in square brackets; nothing else is changed. Paths refer to the working layout of
> the time (`runs/…`, `experiments/…`); in this repository they live under `results/…` and
> `experiments/{v2,round2,camera_ready}/…`.

**Paper:** "Causally Real, Predictively Elusive: Mechanistic Difficulty Representations in Neural Forecasters"
**Status:** accepted as a short paper (8 pages). Camera-ready must answer the reviewers' comments.
**Repo:** `mechanistic-conformal/` (same codebase as Phases 1–5, H7-A, SLCP, TimeMixer).

This file is the full task spec. Read all of it before writing code.

---

## 0. Ground rules (apply to every experiment)

1. **Do not change any existing hyperparameter, split, estimator setting, or selection rule.** Every new experiment reuses the exact settings already used in the paper (iTransformer d_model 256 / 3 layers / 8 heads, lookback 96 → horizon 96; TopK SAE k=64, m=2048; MVDF dose-response λ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25}; α = 0.10; the four-estimator battery with its current settings). If something has to change, stop and write down why in the RESULT file. Do not tune anything.
2. **Pre-register before running.** For each experiment, copy its "Pre-registered criteria" block into `runs/camera_ready/<EXP>/PREREG.md` and commit it (with timestamp and git hash) **before** the first run. Report the outcome against those criteria, whatever it is. No post-hoc changes to thresholds.
3. **Never overwrite existing runs.** All new output goes under `runs/camera_ready/<EXP>/`. Existing `runs/phase*`, `v2_runs/`, `v2_h6_topup/` stay as they are.
4. **Leakage discipline is the same as Phases 3–4:** strict chronological TEST split into `E_select` (first 1/3) / `E_eval_ID` (middle 1/3) / `E_eval_shift` (final 1/3). Anything fitted (estimators, bins, calibration quantiles) is fitted only on data that comes before the windows it is scored on. Reuse the exact fit/eval split that `experiments/phase3_real_predictability.py` uses. Do not invent a new one. Report `n_fit` and `n_eval` in every table.
5. **Seeds.** A "seed" means a full pipeline retrain: forecaster init + SAE init + MVDF selection. Existing seeds are 0/1/2. E2 adds seeds 3 and 4. Write every script to take `--seeds` as a list, so E3/E4/E5 re-run on all five seeds once E2 finishes.
6. **Statistics.** [A reviewer objected to mean ± SD over three seeds.] Never report mean ± SD over 3 seeds. For every headline metric, report:
   - the value for each seed,
   - median and min–max across seeds,
   - a count of seeds that pass the criterion (e.g., "5/5"),
   - a **moving-block bootstrap** 95% CI over evaluation windows within each seed (windows overlap with stride 1, so they are autocorrelated; an iid bootstrap is too narrow). Block length = 96 (= horizon); also run 192 once as a sensitivity check and report whether any conclusion changes. 2,000 resamples, fixed RNG seed.
   - a pooled CI across seeds via a two-level bootstrap (resample seeds, then blocks within each seed).
7. **Provenance.** Every `RESULT.md` starts with: git hash, command line, seed list, dataset fingerprints (SHA-256 of the raw file, as already logged for the Kaggle copies `f56e…`/`1132…` and the HTTP mirror copies `c6d1…`/`c022…`), which data copy was used, GPU, and wall-clock time.
8. **Output format per experiment:** `config.json`, `AGG_<exp>.csv` (one row per dataset × seed × condition), `PER_WINDOW_<exp>.parquet` where window-level data is needed for re-analysis, and `RESULT.md` (provenance → pre-registered criteria → result table → verdict per criterion → one-paragraph plain-English summary → any anomalies).
9. **Page budget.** The paper is 8 pages. Each experiment has to compress to **at most one small table or 2–3 sentences** in the paper. In each `RESULT.md`, finish with a section "**For the paper**" containing (a) one compact table (≤ 6 rows) and (b) the one or two sentences you would write, with exact numbers.

---

## Priority order and scheduling

| Order | ID | What | Answers | Cost |
|---|---|---|---|---|
| 1 | **E0** | Bug fix + definitions audit | R1, R3 (definitions), §5.11 correctness | < 1 h |
| 2 | **E3** | Ex-ante difficulty baselines + gap recovery | R1 (oracle not shown to be recoverable), R1/R2 (restricted estimators) | low, reuses seed 0–2 artifacts |
| 3 | **E4** | R_j robustness (alternative residualization) | R1 (R_j is heuristic) | very low, reuses `candidates.csv` |
| 4 | **E5** | Dose-response diagnostics + natural-range intervention | R1 (artificial intervention vs natural mechanism) | moderate |
| 5 | **E1** | Phases 3 + 4 (+ E3) on electricity and traffic | R1/R2 (mostly two datasets) | moderate–high |
| 6 | **E2** | Seeds 3 and 4 for the core results | R3 (too few repeats) | highest (full retrains) |
| 7 | E6 *(stretch)* | Phase 3 L1 battery on TimeMixer | R2 (architecture generality) | low if TimeMixer artifacts exist |

**Scheduling:** E2 is the longest job but nothing else depends on it until the end. **If a GPU is free, start the E2 retrains (seeds 3, 4 on ETTh1 + ETTh2) in the background first**, then work through E0 → E3 → E4 → E5 → E1 on seeds 0–2 while they run. When E2 finishes, re-run E3/E4/E5 on seeds 3–4 and regenerate the tables.

---

## E0 — Housekeeping (do first)

### E0.1 Phase 5 report bug
`experiments/phase5_sae_reproducibility.py`: `headline_threshold = 0.25` in code, but `write_md` hard-codes the string "0.5 SD" (around lines 290 and 297). Change both strings so they read the threshold from the variable. Regenerate `runs/phase5_sae_repro/PHASE5_RESULT.md` **from the saved results without recomputing** and diff it against the old file. The only change should be the threshold text. Report, per dataset × SAE-seed pair: MVDF cross-seed match rate, candidate-pool base rate, shuffle floor, and the threshold actually used. (The paper text currently says 0.21–0.29 vs base 0.30–0.39. Our last check said 0.08–0.33, mean 0.25, base 0.28–0.43 at 0.25 SD. Confirm which one is correct and give the exact numbers.)

### E0.2 Definitions audit (for the text fixes a reviewer asked for)
Extract the exact formulas **as implemented** and paste them, with file:line references, into `runs/camera_ready/E0/DEFINITIONS.md`:
- `S_t`: how |y − ŷ| is aggregated over channels and horizon steps; how `profile_h` is computed (on which split, normalised how).
- The "90% interval": how the band is built (half-width = quantile of what, applied to which point forecast), and what "covered" means (all 96 steps × all channels inside the band? per-channel?).
- Interval width as reported in the tables (units, normalisation).
- `P_j`, `U_j`, `R_j`, `C_j`, the type A/B/C/D split rule, and the MVDF selection rule (thresholds, top-k, sign conditions).
- Window stride (needed for E3 lag logic).

No new runs. This is so the paper's definitions match the code exactly.

---

## E3 — Ex-ante difficulty baselines (TOP PRIORITY)

### Why
[Reviewer 1, paraphrased: heterogeneity in realised errors does not show that the variation can be predicted with information available before forecasting, so the reported 24.5–29.2% interval-width reduction should not be presented as a recoverable opportunity. The same reviewer noted that the negative predictive results rest on a restricted set of estimators.]

At the moment the paper shows (a) the MVDF features do not predict `S_t` (Phase 3) and (b) an oracle on the true `S_t` could save ~25% width (Phase 4). It does not show whether **anything available at forecast time** predicts `S_t`. E3 fills that gap. Both outcomes help the paper:
- **If nothing predicts `S_t` ex ante:** the oracle is an upper bound on heterogeneity, not a recoverable opportunity. We reword it that way, and the MVDF null is no longer special to the features.
- **If simple ex-ante information does predict `S_t` while MVDF features do not:** difficulty is partly predictable, just not from the causally validated features. That makes the L2 ≠ L1 dissociation sharper, and we can quantify how much of the oracle gap is actually recoverable.

### Datasets / seeds
ETTh1, ETTh2 × seeds 0/1/2 now; seeds 3/4 when E2 lands; electricity and traffic as part of E1.

### Feature sets (predictors of `S_t` at window origin `t`)
Define the window origin `t` = last observed input timestamp. The window forecasts `t+1 … t+96`. Its realized error is fully observed only at time `t+96`.

| Set | Contents | Notes |
|---|---|---|
| **F0 MVDF** | Existing Phase 3 inputs (natural activations of the MVDF features) | Re-run unchanged as the reference row |
| **F1 lagged-S (full)** | `S_{t'}` for windows whose targets are fully observed: `t' ≤ t − 96`. Use `S_{t−96}`, `S_{t−97}`, `S_{t−120}`, `S_{t−144}`, `S_{t−192}`, plus rolling mean / max / median of `S` over origins `[t−96−k, t−96]` for k ∈ {24, 96, 336} | **Hard leakage rule: no `S_{t'}` with `t' > t − 96`.** Add a unit test that fails if any feature uses a target timestamp `> t`. |
| **F1p lagged-S (partial)** | For origins `t' ∈ (t−96, t]`, the observed prefix: `S^{partial}_{t'} = max_{h ≤ t−t'} |e_{t',h}| / profile_h`. Use `t' = t−1, t−24, t−48, t−72`, plus the realized one-step-ahead absolute errors from the last 24 origins | The most aggressive legal information; it is the strongest honest ex-ante baseline. Same unit test. |
| **F2 input-window statistics** | Over the 96-step input, per channel then aggregated by mean and max across channels: std, range, mean |Δx|, std of Δx, linear-trend slope, last value − window mean, lag-1 autocorrelation, spectral entropy, the RevIN/normalisation scale actually used by the model | No model internals |
| **F3 forecast statistics** | From ŷ: std across horizon, range, mean |Δŷ|, distance between ŷ and the last input value | Available at forecast time |
| **F4 dense hidden state** | Mean-pooled hidden state at the same tap the SAE reads, PCA → 12 components (fitted on the fit split only) | "Is the information in the representation at all, just not in the MVDF features?" |
| **F5 full SAE code** | All active SAE features, aggregated, PCA → 12 components (fit split only) | Same question for the whole dictionary |
| **F6 all ex-ante** | F1 ∪ F1p ∪ F2 ∪ F3 | Kitchen sink |
| **F7 all + MVDF** | F6 ∪ F0 | Incremental value of MVDF over everything else: report ΔR² = R²(F7) − R²(F6) |

Lag features that fall before the start of the TEST split are taken from the preceding data (they are in the past, so this is legal). If stride ≠ 1 (see E0.2), convert all lags to the nearest available origin and document it.

### Estimators
The same four as Phase 3, settings unchanged: OLS, isotonic (1-D only: run on each single feature and on the OLS prediction), gradient boosting (depth 3, lr 0.05, subsample 0.8), pinball MLP (2×64 GELU, 300 epochs, median target). **Add two** to answer the restricted-estimators point: ridge regression with α chosen by time-series CV *inside the fit split only*, and a random forest (500 trees, min_samples_leaf = 20). Settings are fixed in advance, with no tuning on eval.

### Metrics (on E_eval_ID and E_eval_shift separately)
- Held-out R² relative to the **fit-split mean** (same definition as Phase 3), with block-bootstrap CI.
- Spearman ρ(Ŝ, S) with block-bootstrap CI. This guards against R² being driven only by a mean shift between splits.
- Best-estimator R² per feature set **and** the full estimator × feature-set grid. Report both; do not report only the best.

### Gap recovery (the conformal translation)
For each feature set, take the estimator with the best **fit-split time-series-CV** score (not the best eval score) and build a Mondrian-on-Ŝ conformal band:
- Use the **same calibration/test partition of E_eval_ID that `naive_matched` uses** (80% calibration budget), so the comparison is like-for-like.
- 10 bins by deciles of Ŝ on the calibration windows; half-width per bin = (1−α) conformal quantile of `S` over calibration windows in that bin (with finite-sample correction as in the existing code).
- Score on the test windows of E_eval_ID, and on E_eval_shift using the same calibration.
- On the **same test windows**, also compute `naive_matched` and the Phase 4 oracle.
- Report coverage, mean width, and **Gap Recovery Ratio** `GRR = (W_naive − W_method) / (W_naive − W_oracle)`. Interpret GRR only when the method's coverage is within ±0.02 of 0.90. Otherwise report it as "coverage-invalid".

### Pre-registered criteria (copy into PREREG.md before running)
- **C3.1 Ex-ante predictable:** a feature set is called "predictive" on a (dataset, seed) cell if the block-bootstrap 95% CI of held-out R² on E_eval_ID lies entirely above 0 **and** Spearman ρ CI lies entirely above 0. Difficulty is "ex-ante predictable" on a dataset if ≥ 4/5 seeds (or 3/3 before E2) are predictive for at least one of F1, F1p, F2, F3, F6.
- **C3.2 Dissociation:** holds on a dataset if C3.1 is met for some ex-ante set **and** F0 (MVDF) is not predictive on ≥ 4/5 seeds.
- **C3.3 MVDF adds nothing:** ΔR² = R²(F7) − R²(F6) has a CI containing 0 or below 0 on ≥ 4/5 seeds.
- **C3.4 Recoverable opportunity:** GRR ≥ 0.20 with valid coverage on ≥ 4/5 seeds for the best ex-ante set. If no set reaches it, the paper says "the oracle gap is an upper bound on heterogeneity, none of which was recovered by any ex-ante predictor we tested."
- **C3.5 Representation contains it:** F4 or F5 is predictive (C3.1 rule) on ≥ 4/5 seeds. This tells us whether the difficulty signal is in the hidden state but not in the MVDF features.

### Output
`runs/camera_ready/E3/AGG_E3_predictability.csv` (dataset, seed, split, feature_set, estimator, n_fit, n_eval, R2, R2_lo, R2_hi, rho, rho_lo, rho_hi), `AGG_E3_gap_recovery.csv` (dataset, seed, split, feature_set, coverage, width, W_naive, W_oracle, GRR, coverage_valid), `RESULT.md`.

---

## E4 — Robustness of causal specificity R_j

### Why
[Reviewer 1, paraphrased: the causal-specificity score is heuristic.] We will not claim a theory. We will show that the conclusions do not depend on the particular residualization (isotonic regression of U on P).

### Inputs
Existing per-feature tables: `v2_runs/<dataset>/candidates.csv` (columns C_j, delta_pred, P_j, U_j, R_j, n_active, corr, type) and the equivalent Phase 1 / camera-ready seed outputs for ETTh1/ETTh2 seeds 0–4. No retraining.

### Alternative definitions of R_j = U_j − Ê[U_j | P_j]
1. **Isotonic** (current, reference)
2. **Binned mean:** 10 equal-count bins of P_j, Ê = bin mean of U_j
3. **Smoothing spline / LOWESS:** fixed smoothing parameter chosen by GCV on the pool (not tuned on any outcome)
4. **Linear:** OLS of U_j on P_j
5. **Rank residual:** rank(U_j) − isotonic fit of rank(U_j) on rank(P_j)
6. **No residualization:** R_j := U_j (shows whether residualizing matters at all)

For each, re-run the existing type assignment and MVDF selection rule (from E0.2, unchanged otherwise).

### Metrics per (dataset, seed, definition)
- Spearman ρ between R_j(def) and R_j(isotonic) over the candidate pool
- Jaccard overlap of the MVDF set vs the isotonic MVDF set
- Type-B enrichment ratio of the MVDF set and its p-value/CI, computed exactly as in `AGG_h1c.csv`
- Phase 3 best held-out R² (marginal + active-window batteries) using that definition's MVDF set (re-run `phase3_real_predictability.py` with the alternative feature list. This step is cheap.)

### Pre-registered criteria
- **C4.1 Selection robust:** median Jaccard vs isotonic ≥ 0.5 across seeds for definitions 2, 3, 5.
- **C4.2 Enrichment robust:** type-B enrichment > 1 with CI excluding 1 on ≥ the same number of (dataset, seed) cells as the isotonic definition, for definitions 2, 3, 5.
- **C4.3 Null robust:** best held-out R² ≤ 0 (or CI includes 0) on ≥ 4/5 seeds for every definition.
- Definitions 4 and 6 are reported as contrasts. They are not part of the pass/fail.

### Output
`runs/camera_ready/E4/AGG_E4_rj_robustness.csv`, `RESULT.md`. The "For the paper" table should have one row per definition: Jaccard (median, range), enrichment pass count, R² null pass count.

---

## E5 — Dose-response diagnostics and natural-range interventions

### Why
[Reviewer 1, paraphrased: the paper should distinguish the causal effect of an artificial intervention on the hidden state from the causal role of a mechanism the model uses naturally.] The paper also admits that dose-response fit quality and monotonicity **were not logged**, so we cannot currently show the six-point curves behaved cleanly. E5 logs them and adds interventions that stay inside the range of activation values the model actually produces.

### E5a — Dose-response diagnostics (required)
Re-run the MVDF dose-response on ETTh1/ETTh2, seeds 0–2 (and 3–4 when available), with identical settings. **Log per feature j:**
- mean Δloss and mean ΔS at each λ, with block-bootstrap CI;
- R² of the linear fit that defines P_j and U_j;
- Spearman ρ between λ and the response across the 6 points (monotonicity), separately for loss and S;
- curvature: fit a quadratic and report the p-value of the quadratic term;
- the fraction of windows where λ·a_j(h_t) exceeds the 99th percentile of a_j's natural (λ=1) distribution on E_select. Only λ = 1.25 can do this; we expect it to be small.

Also check that P_j, U_j, and the MVDF set reproduce the stored values exactly (same seed → same numbers). If they do not, report the discrepancy before anything else.

**Pre-registered (C5a):** among MVDF features, median linear-fit R² ≥ 0.8 and |Spearman monotonicity| ≥ 0.8 for U_j. Report the distribution either way, as a histogram and summary quantiles, for MVDF features vs the rest of the pool.

### E5b — Natural-range intervention (required)
Recompute P_j and U_j using **only λ ∈ {0, 0.25, 0.5, 0.75, 1.0}**. That is, only scaling down toward ablation, never pushing activations above values the model produced. Re-run type assignment, MVDF selection, and type-B enrichment.

**Pre-registered (C5b):** type-B enrichment > 1 with CI excluding 1 on the same cells as the full-λ protocol; Jaccard of the MVDF set vs the full-λ set ≥ 0.5 (median).

### E5c — Interchange (resample) intervention (stretch, do if time allows)
For each MVDF feature j and evaluation window t, replace a_j(h_t) with a_j(h_{t'}), where t' is drawn from E_select windows whose natural activation lies in the top (or bottom) quintile of a_j. This sets the feature to a value it **naturally takes on another input**, instead of scaling it. Decode through the SAE (keep the SAE reconstruction-error term fixed, as in the existing intervention code) and measure the change in S and in loss. Compare the sign and size of the effect with U_j from the scaling protocol.

**Pre-registered (C5c):** sign agreement between the interchange effect on S and sgn(U_j) for ≥ 70% of MVDF features, on ≥ 4/5 seeds.

### Output
`runs/camera_ready/E5/AGG_E5a_diagnostics.csv` (per feature), `AGG_E5b_natural_range.csv`, `AGG_E5c_interchange.csv`, `RESULT.md`.

---

## E1 — Extend the L1 and oracle analyses to electricity and traffic

### Why
[Reviewer 1, paraphrased: the negative predictive results rest mainly on two datasets, which limits generality. Reviewer 2, paraphrased: is the failure specific to the chosen architectures, representations and uncertainty estimators?] Right now Phase 3 (L1 predictability) and Phase 4 (oracle) run on ETTh1/ETTh2 only. The causal results (§5.1, §5.2, §5.6) already cover all four datasets.

### What
Run, on **electricity** and **traffic**, seeds 0/1/2 (3/4 if E2 capacity allows):
1. `experiments/phase3_real_predictability.py`: marginal and active-window batteries, unchanged.
2. `experiments/phase4_oracle_ceiling.py`: unchanged.
3. E3 in full (all feature sets + gap recovery).

### Data
- Kaggle bundle copies: `LD2011_2014.csv` (electricity; `;`-delimited, `,`-decimal, the loader already handles it) and `PeMS.csv` (traffic). Use the existing loaders in `data/datasets.py` and the **same channel subset** the existing v2 runs used, so results match the causal-discovery runs in the paper. Log the fingerprints.
- If Phase 1 (MVDF selection) has to be re-run on these datasets to get self-consistent seed artifacts (as was done for ETT), do it with unchanged settings and check that the type-B enrichment matches `v2_runs/AGG_h1c.csv` in direction. Report any mismatch.

### Known issues to declare up front (write into PREREG.md)
- **Traffic:** the conformal pipeline saturated in-distribution (coverage 1.000) in earlier runs. Phase 3 (R²) and Phase 4 (coverage ≈ 0.90 by construction) are not affected. For E3 gap recovery on traffic, **pre-declare that E_eval_shift is the primary block and E_eval_ID is reported but not interpreted** if naive coverage on ID is > 0.97.
- **Electricity:** the earlier shift-coverage collapse was caused by the learned conditioning mechanism (naive_matched stayed at 0.894). It does not affect Phase 3/4. Report naive_matched coverage on every block anyway.

### Pre-registered criteria
- **C1.1 L1 null generalizes:** best held-out R² of F0 (MVDF) has CI including 0 or below 0 on ≥ 2/3 seeds (≥ 4/5 if five) on each new dataset, for both marginal and active-window batteries.
- **C1.2 Heterogeneity generalizes:** the oracle gap `(W_naive − W_oracle)/W_naive` has a CI excluding 0 on all seeds of each new dataset. Report the percentage.
- **C1.3:** E3 criteria C3.1–C3.5 evaluated on each new dataset.

### Output
`runs/camera_ready/E1/<dataset>/…` with `AGG_E1_phase3.csv`, `AGG_E1_phase4.csv`, the E3 tables for these datasets, and a combined `RESULT.md`. The "For the paper" section should have a 4-dataset table: dataset | MVDF best R² (median [min, max]) | oracle gap % (median [min, max]) | best ex-ante R² | GRR.

---

## E2 — More seeds for the core results

### Why
[Reviewer 3, paraphrased: three repetitions are too few, and a standard deviation over three values is not convincing.]

### What
Full pipeline retrains (forecaster → SAE → MVDF dose-response → type assignment) for **seeds 3 and 4** on **ETTh1 and ETTh2** (required), then on electricity and traffic if compute allows. Stretch goal: seeds 5–9 on ETTh1/ETTh2 if each seed retrain takes < ~2 h. If so, run them and report N = 10.

Then recompute, for all seeds:
1. **Type-B enrichment** of MVDF (H1c), as in `AGG_h1c.csv`
2. **Phase 3 L1 predictability**, marginal and active
3. **Phase 4 oracle gap**, ID and shift
4. **Continuous scorer vs naive_matched**: shift width difference and coverage (`v2_h6_topup` protocol, dense and mvdf modes at minimum)
5. **E3, E4, E5a/b** on the new seeds

Check forecaster val MSE and SAE FVU for each new seed against the existing range (FVU 0.003–0.033). Flag any unhealthy seed. **Do not drop it silently.** If a seed is unhealthy, report it, explain why, and add a replacement seed only if the pre-registered health rule (write it in PREREG.md first: e.g., FVU > 0.05 or val MSE > 1.5× seed-0) triggers.

### Reporting (for every metric above)
One table per metric: rows = dataset, columns = seed 0 … seed N−1 (each with its block-bootstrap CI), then median, min–max, pass count k/N, and the pooled two-level-bootstrap CI. Add a sign test (one-sided binomial over seeds) where the claim is directional. **Do not compute SD over seeds.**

### Pre-registered criteria
- **C2.1:** every claim currently stated as "3/3 seeds" holds on ≥ 4/5 seeds. Any claim that drops below this is reported as weakened, and we change the paper text to match.
- **C2.2:** no headline median moves by more than its own pooled CI half-width when going from 3 to 5 seeds.

### Output
`runs/camera_ready/E2/AGG_E2_<metric>.csv` for each metric, `RESULT.md`, plus a single `SEED_TABLE.md` with all per-seed numbers that the paper quotes.

---

## E6 — L1 battery on TimeMixer (stretch)

### Why
[Reviewer 2, paraphrased: is the null specific to the architecture?] TimeMixer already replicates type-B enrichment (6/6). We have not shown the L1 null there.

### What
If TimeMixer MVDF artifacts exist from `experiments/arch_gen_timemixer.py` (ETTh1/ETTh2 × 3 seeds), run the Phase 3 battery and E3 feature sets F0, F1, F1p, F2, F6 on them. Same criteria as C3.1–C3.3 and C1.1.

### Output
`runs/camera_ready/E6/AGG_E6.csv`, `RESULT.md`.

---

## Final deliverable

`runs/camera_ready/CAMERA_READY_SUMMARY.md` containing:
1. One line per pre-registered criterion: ID, PASS / FAIL / PARTIAL, and the key numbers.
2. The "For the paper" table and sentences from each experiment, in priority order.
3. **A list of every number currently in the paper that changes** (old value → new value, with source file). This includes the Phase 5 match rates from E0.1 and any 3-seed ranges that widen with 5 seeds.
4. Anomalies, failed runs, and anything that could not be done, with the reason.

Report what comes out. A FAIL on any criterion is useful information for the camera-ready and must not be softened, re-thresholded, or dropped.
