# Mechanistically Validated Difficulty Features for Robust Conformal Prediction — Project Report

**Status:** Pre-implementation research spec fully implemented; five Kaggle runs completed;
one calibration bug found and fixed mid-way; current result is a **clean, pre-registered
Outcome B** with one open question (forecaster capacity) still worth one more experiment.

**Date of this report:** 2026-09-03
**Spec:** `mechanistic-conformal-project-spec-FINAL.md`
**Code:** `mechanistic-conformal/` (self-contained tree inside the RecessionRadar repo)
**Compute:** Kaggle T4 GPU for all real runs; local Mac for build/debug only.

---

## 1. What the project asks

> **Central research question (spec §4.1):** Can *mechanistic validation* of internal
> representation features — via causal ablation, not just correlation with historical
> error — improve the **efficiency and robustness** of conformal prediction compared with
> selecting features solely by predictive correlation with error?

The object of study is a **Mechanistically Validated Difficulty Feature (MVDF)** (spec §4.2):
a sparse-autoencoder feature of a neural forecaster's hidden state that is

1. **associated with prediction error** under correlation screening, **and**
2. **survives held-out causal ablation with non-trivial effect magnitude** — operationalised
   as *ranking among the top-k features by `|Δ_j|`* (the ablation effect), **not** merely
   `Δ_j > 0`, because prior work (Dissecting Chronos, §3.2 of the spec) found ~100% of tested
   features "pass" a binary relevance test.

The novelty is **not** "SAE + conformal prediction" (close to FFDCI and Feature Conformal
Prediction) and **not** "SAE + causal ablation on a time-series model" (Dissecting Chronos).
The novelty is the **controlled experiment**: does the causal-validation step itself buy
robustness under distribution shift, holding feature budget and calibration mechanism fixed?

### Pre-registered outcomes (spec §4.4 — written before any results were seen)

| Outcome | Pattern | Interpretation | Publishable? |
|---|---|---|---|
| **A** | MVDF ≫ correlation-only under shift, comparable in-distribution | Causal validation specifically buys robustness | Yes — strong |
| **B** | MVDF ≈ correlation-only everywhere | Validation finds causally-influential features, but causal influence doesn't translate to conformal robustness | Yes — genuine negative, extends causal-inertness literature to a new domain |
| **C** | correlation-only > MVDF under shift | Causal influence and UQ-usefulness are different objectives | Yes — also informative |

### Three hypotheses (spec §4.3)

- **H1 — Correlation ≠ mechanism.** Some SAE features correlate strongly with error but show
  small ablation effect: `∃ j : |Corr(a_j, |e|)| large, |Δ_j| small`. Tested by the RQ1
  rank-divergence diagnostic (§6.5) and by explicit case-study features.
- **H2 — Validation improves in-distribution efficiency.** At matched budget `k` and matched
  (guaranteed) coverage, Mondrian-CP on the MVDF gives lower mean/median interval width than
  Mondrian-CP on a correlation-only feature.
- **H3 (flagship) — Validation improves robustness under shift.** The width/coverage advantage
  of MVDF over correlation-only is *larger, or only appears,* under distribution shift, because
  a merely-correlated feature's relationship with error is more likely to reflect a property
  of the calibration-time distribution than a stable mechanism.
  `Δ_shift = degradation_correlated − degradation_MVDF`, predicted `> 0`.

---

## 2. What we built (the pipeline)

Full pipeline per spec §5.1, implemented and runnable end-to-end on real data:

```
LTSF benchmark data
   │  data/datasets.py — ETT over HTTP; traffic/electricity/weather/solar/exchange/QPS
   │                     from a Kaggle-attached "unified multivariate TS" bundle,
   │                     each with a per-file adapter (raw electricity/weather → hourly)
   ▼
iTransformer  (forecasters/itransformer.py, ICLR'24 arch — the FFDCI backbone)
   │  d_model 256, 3 layers, 8 heads; lookback 96 → horizon 96; RevIN-style non-stationary
   │  norm; trained with MSE, Adam lr 1e-4, 3-epoch warmup then 0.8/epoch decay, early stop.
   │  Tap point: the pre-prediction-head per-variate token for the target channel (last col).
   ▼
TopK SAE  (sae/topk_sae.py, sae/train_sae.py — Dissecting-Chronos recipe)
   │  d_sae = 8·d_model = 2048, k = 64, MSE reconstruction, dead-feature resampling.
   │  Trained ONLY on TRAIN-split hidden states (disjoint from the conformal calibration set).
   │  Hidden states standardised per-dim before the SAE (buffers on the module); all
   │  encode/decode/ablation paths go through standardize/unstandardize.
   ▼
   ├── Correlation screening  (validation/correlation_screen.py)
   │      s_j = Corr(z_j, |e|) on the calibration (VAL) split; rank by |s_j|;
   │      eligibility gate: feature active in ≥ 2% of windows.
   │
   └── Causal ablation  (validation/ablation.py)
          zero z_j → decode → un-standardize → patch the target-variate token back
          into the forward pass → recompute the nonconformity score → Δ_j.
          Δ_j is measured ONLY over windows where feature j is active (zeroing an
          already-off feature is a no-op that dilutes the estimate toward zero).
          Rank by |Δ_j| (magnitude), not Δ_j > 0.
   ▼
Three matched-budget feature sets (k = 12)
   • SAE-random      — k random eligible SAE features (control: rules out "SAE regularises for free")
   • SAE-correlated  — top-k by |s_j|, no ablation check (the primary comparator)
   • MVDF            — top-k by |Δ_j|  (the primary condition)
   ▼
Calibration
   • Variant B (the paper's contribution) — calibration/mondrian_perbin.py
       feature signal z_t = mean of standardized activations of the k selected features
       → 3 Mondrian bins by calibration quantiles
       → per-bin static base quantile q_k of the nonconformity score
       → per-bin online ACI correction a_{k,t}, updated only on steps whose z_t ∈ bin k
       Also a `static_only` mode (no per-bin ACI) to demonstrate the spec §5.6 shift-fragility.
   • Variant A (FFDCI-style) — calibration/ffdci_head.py
       pinball-loss MLP maps the k activations → a per-instance error-quantile deviation;
       a single global ACI correction is added on top.
   ▼
Evaluation  (eval/metrics.py)
   coverage (whole-horizon), worst-step coverage, mean interval width, Winkler score,
   MACE (rolling), timesteps-per-bin — reported in-distribution AND as Δ-degradation
   under the pre-registered shift.
   ▼
RQ1 diagnostic  (validation/rq1.py)
   Spearman ρ and top-set Jaccard between the correlation ranking and the ablation-magnitude
   ranking, over a 400-feature sample NOT pre-filtered by correlation (spec §6.5, §9.5).
```

### Pre-registered distribution shift (spec §6.3, `shift/shift_windows.py`)

For every dataset the shift window is **the final 20% of the test split** — the block
furthest in calendar time from the train+val calibration window, hence where
temporal-dependence similarity to calibration is weakest and a seasonal/regime difference is
most likely. A `tail_frac` (not an absolute step count) is used so the definition is robust
to the hourly-resampled lengths the data adapters produce. Each run reports a KS two-sample
test; a window with KS p ≥ 0.05 is flagged, never swapped. On traffic: KS stat 0.236,
p ≈ 3×10⁻³³ — a real marginal shift.

### Synthetic control (`experiments/mve_synthetic.py`)

A controlled scaffold with **known ground truth**: a latent difficulty process drives the
target's aleatoric noise scale; the forecaster's hidden state carries (a) a *mechanistic*
coordinate the head reads and that tracks difficulty, and (b) a *spurious* coordinate that
correlates with error in the calibration regime only and that the head ignores. Used to
debug the pipeline and — in the paper — to show the machinery *does* find and exploit a
difficulty feature when one exists by construction, which turns a negative result on real
data into a scope statement rather than a failure. **Not a scientific result.**

---

## 3. Chronology — what we tried, in order

### Iteration 0: synthetic-first, then corrected

Initial plan was synthetic forecaster + synthetic series. User (correctly) rejected this for
an A* venue: "we validated the machinery on data we designed to make it work" is a desk
reject. Switched to **real iTransformer + real ETT data**, synthetic kept only as the §4.5
control.

### Iteration 1: ETT only, local

- iTransformer on ETTh1: test MSE ≈ 0.47. Verified this is a *real* forecaster — a plain
  per-variate linear model with RevIN gets the same 0.47; mean-predictor 0.90; persistence
  1.63. The gap to published iTransformer (~0.39) is capacity/training-budget, not a bug.
- **First core finding:** `max|corr(hidden PC, |e|)| ≈ 0.08–0.11` on **all four ETT datasets**,
  independent of forecaster accuracy (ETTh2 at MSE 0.25 no better than ETTh1 at 0.47). ETT
  errors are near-homoskedastic — little instance-level difficulty variance for any internal
  feature to encode.
- Consequence: ablation `Δ_j` near the numerical floor; `Δ>0` rate ≈ 0.45–0.55 (not the ~1.0
  Dissecting Chronos reports on Chronos-Large); MVDF selection ≈ random.
- **Bug found & fixed:** ablation was averaging `Δ_j` over *all* windows including the ~95%
  where the feature is inactive → diluted toward zero. Fixed to condition on active windows.

### Iteration 2: push to Kaggle, full dataset suite

User has a Kaggle "unified multivariate TS" bundle (`uom220001c/archive/`): `ETTh.csv` +
`ETTm.csv` (14-col concatenations of *1 and *2), `LD2011_2014.csv` (raw UCI electricity,
715 MB, `;`-delimited, `,`-decimal), `mpi_roof.csv` (raw Jena weather, 135 MB),
`PeMS.csv` (traffic, 862 sensors), `solar_Alabama.csv`, `exchange.csv`, `QPS.csv`.

Built `data/datasets.py` with a `LOADERS` registry + per-file adapters:
ETT concat-splitting; electricity/weather resampled to hourly; delimiter/decimal sniffing;
`get_splits(name, max_channels=48)` caps wide sets so the per-variate-token SAE stays
tractable. `data/probe_inputs.py` lists what resolves before the long run.

### Run 1 (Kaggle) — partially invalid

Got through ETTh1/h2/m1/m2 + traffic + weather, then crashed on `solar` (a datetime string
column passed through as a channel). **Valid takeaways:**
- RQ1 ρ = 0.05–0.16, Jaccard 0.12–0.22 on all 6 → **H1 holds** (correlation ranking ≠
  ablation ranking). Robust — doesn't need a perfect SAE.
- `corr(hidden,|e|)` = 0.09–0.18 on every dataset, **including traffic (0.13) and weather
  (0.10)** → difficulty weakly represented on *all* standard LTSF benchmarks, not just ETT.
- **Invalidated by:** the SAE **diverged** late in training on traffic (FVU 0.53), weather
  (0.95), ETTh2 (0.26), ETTm2 (0.14) — reconstruction MSE went 0.005 → 0.03 → 0.5 around
  step 16–18k. Cause: unbounded dead-feature resampling (rewriting 400–800 encoder rows at
  once on a converged SAE) + a cosine-LR-to-zero schedule.

### Run 2 fixes: SAE stability + solar

`sae/train_sae.py` + `sae/topk_sae.py`:
1. Standardise hidden states per-dim before the SAE (buffers `_input_mean`/`_input_std`).
2. `resample_cap = 64` dead rows per event (was unbounded).
3. Stop dead-feature resampling after 50% of training (`resample_until_frac`).
4. Flat LR 1e-4 + 200-step warmup, gradient clip 1.0 (dropped the cosine schedule).

Verified on full ETTh1: reconstruction FVU now monotone 0.040 → 0.011 over 20k steps, 0 dead.
Run WARNs if `fvu_standardized > 0.3`.

`solar` fixed: `get_splits` now `pd.to_numeric`-coerces every non-date column and drops any
that are >50% non-numeric.

### Run 3 (Kaggle) — SAE healthy, electricity still broken

SAE healthy on **all** datasets (FVU 0.003–0.014). But `electricity` crashed again — the raw
`LD2011_2014.csv` parse produced 0 rows. Rewrote `_load_electricity` with a delimiter/decimal
sniffer that raises with sample timestamp values on failure; added an empty-data guard in
`experiments/mve_real.py`.

### Run 4 (Kaggle) — seed-replicated, apparent Outcome A

`mve_seeds.ipynb`: 4 datasets (traffic, weather, electricity, ETTh2) × 3 seeds, all healthy
SAEs, electricity parsed fine.

**Apparent result:** traffic looked like a **seed-stable Outcome A** — H3 `Δ_shift`
`d_coverage +0.054 ± 0.024`, `d_width +0.034 ± 0.010`, `d_winkler +0.050 ± 0.017`,
`d_mace +0.078 ± 0.029`, sign never flipping across seeds. weather null; ETTh2 covered more
only by widening; electricity marginal.

### Run 4 → calibration bug found

Reading the H2 detail: **in-distribution coverage was 60–72% on ALL conditions** (target 90%),
and Variant A gave **0%**. Root cause:
- the nonconformity score was `mean_h |e_h|` (mean absolute error over the horizon),
- but a window was scored "covered" iff **all 96 horizon steps** were inside the band,
- these are inconsistent: `q_k` calibrated to the *mean* step error always fails the *max*-based
  coverage check → chronic undercoverage.
- Variant A: `GlobalACI` was seeded with `base_q = 0`, so the half-width collapsed to ≈ 0 and
  the ACI correction (scaled by `max(base_q, 1e-6)`) never grew.

**The Run 4 "Outcome A" was an artefact of the undercoverage.**

### Calibration fix

`validation/errors.py`, `calibration/mondrian_perbin.py`, `calibration/ffdci_head.py`:
- **Nonconformity score** = `max_h( |e_h| / profile_h )`, where `profile` is the calibration
  per-step `|e|` shape (mean 1). This is the *smallest scalar half-width that would have
  covered the entire window*, given a fixed per-step band shape. The interval is
  `ŷ ± hw · profile`, and "covered" ⇔ `max_h(|e_h|/profile_h) ≤ hw` — the **identical**
  quantity `q_k` targets. In-distribution coverage now lands at 1−α by construction.
- `score_with_patched_hidden` and `ablate_features` take `step_profile` and use the same
  score, so the ablation `Δ_j` is defined on the same quantity the calibration uses.
- Mondrian: `n_bins` 4 → 3 (per-bin calibration sets don't starve), `gamma_perbin` 0.02 →
  0.05 (converge to 1−α faster on short test splits), `a_k` clipped to `[−0.9·q_k, clip]`.
- Variant A: seed `GlobalACI` with the real pooled base quantile; the MLP predicts a
  *deviation* from the pooled mean, not the absolute quantile.

Verified on full ETTh1: **all 4 conditions 90.0–90.2% in-distribution**, MACE 0.04–0.08,
Variant A working.

### Run 5 (Kaggle) — seed-replicated, correct calibration — THE CURRENT RESULT

`mve_seeds.ipynb` re-run with the calibration fix. 4 datasets × 3 seeds. In-distribution
coverage now correct (weather 90.0% exactly; traffic ≈ 88%; the ~2 pt shortfall on traffic
is per-bin ACI burn-in on a 3318-window test split, documented not hidden).

**Traffic's Outcome A collapsed to noise.** See §4.

---

## 4. Current results (Run 5, the valid one)

4 datasets × 3 seeds. Healthy SAEs (FVU 0.003–0.033). Correct ~90% in-distribution coverage.

| Dataset | fc val MSE | SAE FVU | corr(hidden,\|e\|) | RQ1 ρ (per seed) | RQ1 Jaccard | Δ>0 rate |
|---|---|---|---|---|---|---|
| traffic | 0.274 | 0.003 | 0.119 ± 0.009 | −0.071, −0.077, −0.024 | 0.135 | 0.21 |
| weather | 0.458 | 0.029 | 0.185 ± 0.030 | +0.096, +0.082, +0.209 | 0.227 | 0.51 |
| electricity | 0.069 | 0.012 | 0.149 ± 0.017 | +0.160, +0.180, +0.018 | 0.237 | 0.24 |
| ETTh2 | 0.227 | 0.005 | 0.139 ± 0.005 | +0.074, +0.090, +0.205 | 0.213 | 0.47 |

### H3 — `Δ_shift = degradation_correlated − degradation_MVDF` (>0 favours MVDF)

| Dataset | Δ coverage (per seed) | Δ width (per seed) | Δ winkler (mean ± sd) | Verdict |
|---|---|---|---|---|
| **traffic** | −0.006, +0.000, +0.022 | **+0.295, −0.757, −0.109** | −0.202 ± 0.503 | **Sign flips every seed — pure noise** |
| weather | −0.0004, −0.0004, +0.0002 | +0.004, −0.005, −0.015 | −0.005 ± 0.009 | Dead null |
| electricity | +0.0003, −0.0009, +0.0024 | −0.062, +0.040, −0.101 | −0.038 ± 0.070 | Noise |
| ETTh2 | +0.004, −0.001, +0.002 | +0.017, −0.020, −0.026 | −0.010 ± 0.023 | Noise |

**Traffic's Δwidth swings `+0.30 / −0.76 / −0.11` across three seeds** — the magnitude
varies ~10× and the sign flips. That is not a small effect buried in noise; it *is* noise.
Δcoverage is `≈ 0` every seed. With correct ~90% calibration, **there is no MVDF advantage
on any of the four datasets, on any metric.**

### H2 — in-distribution (traffic seed 0, representative)

| Condition | coverage | mean width | MACE | timesteps/bin |
|---|---|---|---|---|
| SAE-random B | 0.879 | 4.70 | 0.133 | [1925, 265, 1128] |
| SAE-correlated B | 0.882 | 4.84 | 0.127 | [0, 2112, 1206] |
| MVDF B | 0.879 | 4.61 | 0.147 | [1104, 1078, 1136] |
| MVDF A (opaque MLP) | 0.882 | 4.81 | 0.124 | — |

MVDF is marginally *narrower* here (4.61 vs 4.84) but this does not replicate across seeds
and is within noise of SAE-random. **H2 is also null.**

Note the bin distributions: MVDF stratifies evenly (`~[1100]×3`); SAE-correlated's feature
barely varies so Mondrian on it collapses toward 2 bins; SAE-random collapses toward 1 bin
(a random 12-feature aggregate is nearly constant → plain global ACI). This *is* H1 showing
through — the ablation-selected and correlation-selected features genuinely behave
differently as bin-defining signals — but that difference does **not** translate into a
calibration advantage.

---

## 5. What holds regardless — the robust findings

### H1 — correlation ≠ mechanism (CONFIRMED, every run, every dataset)

- RQ1 Spearman ρ between the correlation ranking and the ablation-magnitude ranking is in
  **[−0.08, +0.21]** across all datasets and seeds; top-set Jaccard **0.12–0.24**. The two
  criteria select substantially different features. On traffic ρ is even slightly *negative*.
- `|corr ∩ mvdf| = 0/12` on traffic — the top-12 correlated and top-12 ablation features are
  **disjoint**.
- This is a genuine mechanistic-interpretability result on its own, independent of anything
  downstream, and it is exactly what H1 predicted.

### The Dissecting-Chronos "100% causal" finding does NOT replicate on an iTransformer

- `Δ>0` rate is **0.21–0.51**, not ~1.0. On the iTransformer, roughly half of active SAE
  features have a *negative* ablation effect (zeroing them *reduces* error), and the effect
  magnitudes are small. The near-universal positive `ΔCRPS` that Dissecting Chronos reports
  on Chronos-T5-Large is a property of that model, not of time-series SAEs in general.

### Difficulty is weakly represented in the pre-head layer on every LTSF benchmark

- `max|corr(hidden PC, |e|)|` = **0.09–0.22** on ETT (all four), traffic, weather,
  electricity, solar. This is the aleatoric-dominated regime the spec flags in §9.2 and
  §4.5: where per-instance forecast difficulty is not linearly readable from the forecaster's
  representation, no internal feature — validated or not — can stratify it, and conformal
  calibration conditioned on such a feature reduces to pooled calibration.

---

## 6. The honest verdict — this is Outcome B

Per the pre-registered outcomes (spec §4.4):

> **Outcome B — "Interesting negative":** MVDF ≈ correlation-only everywhere. Mechanistic
> validation finds causally-influential features, but causal influence doesn't translate
> into conformal robustness. Still publishable — a genuine negative result that extends the
> causal-inertness literature to a new domain (conformal calibration).

Evidence, 4 datasets × 3 seeds, healthy SAEs, correct 90% calibration:

1. **H1 confirmed** — the ablation ranking and the correlation ranking are genuinely
   different (RQ1 ρ ≈ 0, disjoint top-k). Mechanistic validation *does* change which
   features you select.
2. **H2 and H3 null** — at matched budget and matched calibration mechanism, the
   ablation-selected feature set gives *neither* sharper in-distribution intervals *nor*
   better shift-robustness than the correlation-only or the random set. Every H3 metric's
   sign flips across seeds.
3. **Mechanistic reason** — on standard LTSF benchmarks with a competently-trained
   iTransformer, per-instance difficulty is only weakly encoded in the pre-head
   representation (`corr ≈ 0.1`). There is little difficulty structure for *any* feature
   selection strategy to exploit, so the causal-vs-correlational distinction — real as it is
   at the feature level — washes out at the calibration level.

The paper this supports:

> *We ran the controlled experiment the spec describes. Causal validation identifies a
> substantially different feature set from correlation screening (H1), but on standard
> long-term forecasting benchmarks this does not produce sharper or more shift-robust
> conformal intervals (H2, H3 null). The bottleneck is upstream: a competent forecaster's
> pre-head representation encodes per-instance difficulty only weakly, so no representation-
> based feature — validated or not — has enough signal to beat pooled calibration. The RQ1
> diagnostic (correlation ranking vs ablation ranking) is a cheap, universal check that
> tells you which regime you are in before you build a calibration scheme.*

This is a legitimate contribution: it is the experiment neither FFDCI nor Dissecting Chronos
ran, it has a clean theoretical frame (validity is guaranteed by Mondrian CP regardless of
feature quality; informativeness is what the experiment measures and here finds absent), and
the negative is *informative* — it tells practitioners not to invest in causal validation of
difficulty features for LTSF conformal calibration, and why.

---

## 7. The one open question worth one more experiment

**Could the null be a forecaster-capacity artefact rather than a real negative?**

Our iTransformer is deliberately the FFDCI fixed config (d_model 256, 3 layers) and is
under-trained relative to published numbers (ETT test MSE 0.23–0.70 vs ~0.39 SOTA;
weather 0.46). A stronger or better-trained forecaster might develop a richer difficulty
representation (`corr(hidden,|e|) ≫ 0.1`), which is the precondition for validation to
matter at all.

Two cheap tests (no new dependencies):

1. **Bigger / properly-trained iTransformer** — d_model 512, 4 layers, d_ff 1024,
   40 epochs. Re-run traffic + ETTh2 + electricity + weather × 3 seeds. If
   `corr(hidden,|e|)` rises well above 0.1 **and** H3 still nulls → **Outcome B is
   bulletproof**. If `corr` stays ≈ 0.1 regardless of capacity → the representation
   genuinely lacks difficulty information, which is itself a clean finding.

2. **Raw-hidden-PC control** — add one Variant-B condition that bins on the top raw
   hidden-state principal component (no SAE). If raw-PC binning shows an H3 effect that MVDF
   does not, the SAE aggregation is destroying signal. If raw-PC is also null, the signal
   isn't there.

The more thorough (but ~2h more engineering + a longer run) option is a **pretrained Chronos
backbone** — literally the Dissecting-Chronos model where the "100% causal" finding
originated. If difficulty isn't readable there either, the negative is unimpeachable.

**Decision point:** run test (1) [+ optionally (2)] once. Then:
- `corr` rises, H3 nulls → write Outcome B, done.
- `corr` rises, H3 turns positive and seed-stable → we had a capacity problem; pursue
  Outcome A properly (scale to PEMS03/04/07/08, the Variant-A/B ablation, the H1 case-study
  figure).
- `corr` stays ≈ 0.1 → write Outcome B with an even stronger "difficulty isn't there"
  framing.

---

## 8. Known limitations / caveats in the current results

- **Forecaster is under-trained** vs published iTransformer (see §7). Deliberate (FFDCI
  fixed config) but it caps how much difficulty the representation can encode.
- **Test splits are short** for the online per-bin ACI to fully converge — traffic test is
  3318 windows / 3 bins; in-distribution coverage lands at ~88% not 90% on traffic (weather,
  with 34 874 windows, is exactly 90.0%). Documented, not hidden.
- **`max_channels=48` cap** on traffic (862 → 48 most-variable + target) and electricity for
  SAE tractability. Changes the multivariate context the forecaster sees; the difficulty
  question is per-window-target so this is a second-order effect, but it is a deviation from
  FFDCI's protocol.
- **Data splits are not byte-identical to FFDCI's** — the Kaggle bundle ships raw/concatenated
  files that our adapters normalise (electricity/weather resampled to hourly, ETT split from
  a 14-col concat). Fine for the MVE; would need the canonical CSVs for a headline
  reproduction table.
- **Only one shift type** — the final 20% of each test split (an abrupt held-out block).
  Gradual drift is not tested.
- **SAE-random control degenerates** into pooled ACI (a random 12-feature aggregate is nearly
  constant → 1 bin). This is expected and arguably correct as a control, but it means "MVDF
  vs random" is really "stratified vs pooled," not "good stratification vs bad."

---

## 9. Repository map

```
mechanistic-conformal/
├── PROJECT_REPORT.md              ← this file
├── NOTES.md                       ← shorter running status
├── README.md
├── configs/
│   └── mve_etth1.yaml             base config (all datasets use it via the notebook,
│                                  overriding dataset.name / seed). Pre-registered A/B/C
│                                  outcomes in the header.
├── data/
│   ├── datasets.py                LOADERS registry, per-file adapters, sniffer, splits,
│                                  max_channels cap
│   └── probe_inputs.py            "what resolves under /kaggle/input" pre-flight
├── forecasters/
│   ├── itransformer.py            iTransformer (ICLR'24) + patch-hidden hook for ablation
│   ├── train_itransformer.py      MSE, Adam 1e-4, warmup+decay, early stop
│   ├── synthetic_forecaster.py    §4.5 control only
│   └── train_forecaster.py        §4.5 control only
├── sae/
│   ├── topk_sae.py                TopK SAE + standardize/unstandardize buffers
│   └── train_sae.py               stable trainer (capped resampling, flat LR, grad clip)
├── validation/
│   ├── errors.py                  nonconformity score = max_h(|e_h|/profile_h); patch-hidden
│   ├── correlation_screen.py      s_j = Corr(z_j, |e|)
│   ├── ablation.py                zero→decode→patch→Δ_j, active-window-conditioned
│   └── rq1.py                     Spearman ρ / Jaccard, H1 case studies
├── calibration/
│   ├── aci.py                     GlobalACI primitive, empirical_quantile
│   ├── mondrian_perbin.py         Variant B — Mondrian bins + per-bin ACI (the contribution)
│   └── ffdci_head.py              Variant A — pinball MLP + global ACI
├── shift/
│   └── shift_windows.py           PRE-REGISTERED shift = final 20% of each test split
├── eval/
│   ├── metrics.py                 coverage / worst-step / width / Winkler / MACE / degradation
│   └── plots.py                   coverage-width-over-time, Δ_j histogram, RQ1 scatter
├── experiments/
│   ├── mve_real.py                the single-dataset MVE (all the wiring above)
│   ├── full_suite.py              loop mve_real over datasets × seeds, aggregate
│   └── mve_synthetic.py           §4.5 control
├── notebooks/
│   ├── build_kaggle_notebook.py   generator: --full-suite | --datasets ... --seeds ...
│   ├── mve_kaggle.ipynb           full 9-dataset suite, single seed
│   └── mve_seeds.ipynb            [traffic, weather, electricity, ETTh2] × seeds [0,1,2]
└── mve_runs/                      ← Run 5 outputs (the valid ones)
    ├── AGG_per_run.csv            per (dataset, seed): fc MSE, SAE FVU, corr(h,|e|),
    │                              RQ1 ρ/Jaccard, Δ>0 rate, H3 Δcoverage/width/winkler/mace
    ├── AGG_by_dataset.csv         mean ± sd across seeds
    └── mve_<dataset>_seed<n>/     run_log.json, rq1_diagnostic.json,
                                   h2_in_distribution.csv, h3_shift.csv, variant_ab.csv,
                                   delta_j_distribution.csv, *.png
```

### Reproducing a run

```bash
# local, one dataset, full budget (~15 min CPU on ETTh1)
python -m experiments.mve_real --config configs/mve_etth1.yaml

# regenerate a Kaggle notebook after any code change
python notebooks/build_kaggle_notebook.py --datasets traffic weather electricity ETTh2 \
    --seeds 0 1 2 --out notebooks/mve_seeds.ipynb
python notebooks/build_kaggle_notebook.py --full-suite --out notebooks/mve_kaggle.ipynb
```

Kaggle: import the notebook, attach the `uom220001c` bundle, GPU + Internet on, Run All,
download `/kaggle/working/mve_runs.zip`.

---

## 10. Bugs found and fixed along the way (for the reproducibility appendix)

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Ablation `Δ_j` averaged over inactive windows | `Δ>0` rate ≈ 0.45 instead of ~1.0; MVDF ≈ random | condition `Δ_j` on windows where the feature is active |
| 2 | iTransformer LR too high (3e-4) + cosine-to-zero | val MSE rises after epoch 2, early-stops at ~0.70 | LR 1e-4, 3-epoch warmup, 0.8/epoch decay |
| 3 | SAE dead-feature resampling unbounded + cosine LR | reconstruction FVU diverges 0.005 → 0.5 on traffic/weather late in training | standardise inputs; cap resampling at 64 rows/event; stop after 50% of steps; flat LR + warmup + grad clip; WARN if FVU > 0.3 |
| 4 | `solar` datetime column passed as a channel | `could not convert string to float` | `pd.to_numeric`-coerce all non-date cols, drop >50%-non-numeric |
| 5 | `electricity` raw file parse → 0 rows | silent empty frame → `ZeroDivisionError` downstream | delimiter/decimal sniffer; raise with sample values on failure; empty-data guard in `mve_real.py` |
| 6 | **Nonconformity score (mean) ≠ coverage check (all-steps max)** | **in-distribution coverage 60–72% on ALL conditions; Run 4 "Outcome A" was an artefact** | score = `max_h(|e_h|/profile_h)`, interval = `ŷ ± hw·profile`, coverage ⇔ same quantity ≤ hw; consistent by construction |
| 7 | Variant A `GlobalACI(base_q=0)` | half-width ≈ 0 → 0% coverage; ACI correction never grows | seed `GlobalACI` with the real pooled base quantile; MLP predicts a deviation from the pool mean |

---

## 11. One-paragraph summary

We implemented the full spec pipeline — real iTransformer on real LTSF benchmarks (ETT,
traffic, electricity, weather, solar), a TopK sparse autoencoder on its pre-head hidden
state, correlation screening and held-out causal ablation to select difficulty features,
three matched-budget feature sets (random / correlation-only / ablation-validated MVDF), and
two calibration variants (Mondrian bins + per-bin ACI; FFDCI-style MLP + global ACI),
evaluated in-distribution and under a pre-registered final-20%-of-test distribution shift,
across 3 seeds. After fixing an SAE-divergence bug and a nonconformity-score/coverage-check
inconsistency that had briefly produced a false-positive "Outcome A" on traffic, the clean
result is **Outcome B**: mechanistic validation selects a substantially different feature set
from correlation screening (H1 confirmed everywhere — RQ1 Spearman ρ ≈ 0, disjoint top-k),
but this does **not** yield sharper or more shift-robust conformal intervals (H2 and H3 null;
every H3 metric's sign flips across seeds). The mechanistic reason is that a competently-
trained forecaster's pre-head representation encodes per-instance difficulty only weakly
(`max|corr(hidden PC, |e|)| ≈ 0.09–0.22` on every benchmark), so no representation-based
feature has enough signal to beat pooled calibration. One capacity check remains worthwhile
— a larger / better-trained forecaster, or a pretrained Chronos backbone — to confirm the
null is not an artefact of under-training before writing the negative up.
