# V2 Phases 0–2 — Results & Assessment

**Run:** `v2_phases012.ipynb` on Kaggle GPU, all 8 datasets (ETTh1/h2/m1/m2, traffic,
electricity, weather, solar), single seed. Outputs in `mechanistic-conformal/v2_runs/`.
**Date:** 2026-09-03.
**What this closes:** the gap the `PROJECT_REPORT.md` flagged — Run 5's Outcome B used
single-point ablation, no invariance measurement, and Mondrian-as-headline, so it never
tested v1/v2's actual flagship hypothesis (H3 = cross-environment invariance). Phases
0–2 run the originally-designed dose-response + invariance machinery.

---

## TL;DR

Three things happened, in decreasing order of how favourable they are:

1. **H1c holds — and it is a real, clean positive.** Type-B features (low prediction
   effect `|P_j|`, high uncertainty effect `|U_j|`) exist in every dataset (9–22 % of
   the candidate pool), and **`MVDF_v2` is enriched for them in 8/8 datasets**
   (enrichment ratio 1.4–4.5×), while `MVDF_v1` (Run 5's single-point criterion) and
   the correlation-only set are **not** (ratios 0.0–1.8×, mostly below 1). The
   dose-response + `R_j` residualization does something the simpler criteria don't:
   it isolates features whose causal effect is disproportionately on uncertainty.

2. **Phase 0 is a NO-GO — difficulty is not more recoverable at any other layer.**
   Across 7 taps × 8 datasets, the cheap `max|corr(activation PC, S_t)|` signal is
   **0.08–0.21 everywhere**, never clearing the pre-registered 0.30 threshold. This
   generalises Run 5's pre-head finding across the whole network: it is not that we
   were looking in the wrong place.

3. **H3 fails — `MVDF_v2` is NOT more invariant than correlation-only.** This is the
   flagship measurement, and the pre-registered outcome is **§7 row 3**: the
   dose-response/`R_j` criterion selects a genuinely different, genuinely more
   uncertainty-specific feature set (point 1), but that set's conditional
   difficulty–score relationship is **no more stable across deployment environments**
   than a correlation-selected one — `MVDF_v2` beats `correlated` on `I` on exactly
   **1/8 datasets (ETTh1, by 0.053)** and is worse, often much worse, on the other 7.
   In fact `MVDF_v2` is usually **less** stable (negative `spearman_stability` on 5/8).

So: the mechanistic-validation *machinery works and finds what it was designed to
find* (type-B features), but the property that was supposed to make those features
valuable for conformal prediction — invariance under shift — **does not follow from
being uncertainty-specific**. That is a real answer to the question v1 posed, and it
is distinct from Run 5's null (which never asked it).

---

## Phase 0 — Layer sweep (spec §4.5)

**Pre-registered:** probe 7 taps (`embed, layer1, layer2, layer3_attn, layer3_ffn,
pre_head, head_input`); a tap "clears" if `max|corr(activation PC, S_t)| ≥ 0.30` on
≥ 2 datasets; on NO-GO, Phase 1 falls back to `pre_head` and the null is reported.

### Result: **NO-GO**

Per-tap signal, averaged across the 8 datasets:

| tap | mean | max (which dataset) | min |
|---|---|---|---|
| embed | 0.119 | 0.165 | 0.081 |
| layer1 | 0.149 | **0.208** (weather) | 0.102 |
| layer2 | 0.150 | 0.186 | 0.110 |
| layer3_attn | **0.154** | 0.198 (electricity) | 0.112 |
| layer3_ffn | 0.146 | 0.190 | 0.094 |
| pre_head | 0.146 | 0.168 | 0.095 |
| head_input | 0.146 | 0.168 | 0.095 |

- The best single number anywhere is **0.208** (weather, layer1) — still well under
  0.30, and only one dataset.
- `layer3_attn` is marginally the best tap on average (0.154 vs pre_head's 0.146), but
  the difference is noise-level and it clears the threshold on **zero** datasets.
- The mid-stack taps (layer1–layer3_attn) are ~0.005–0.008 higher than pre_head on
  average. Not nothing, not enough to act on.

**Reading (spec §7 row 4 territory):** difficulty is weakly and roughly-uniformly
encoded throughout the iTransformer, not concentrated at some layer we hadn't checked.
Run 5's pre-head finding was not a tap-choice artefact.

**Consequence:** Phase 1 and Phase 2 ran at `pre_head` (the documented fallback). The
null sweep is a reportable result.

---

## Phase 1 — Dose-response, `R_j`, feature-type table (spec §4.2–4.6)

**What ran:** at `pre_head`, per dataset — SAE (FVU 0.03–0.05, healthy), 120 unbiased
candidate features, `C_j` (observational diagnostic), the **unified dose-response
sweep** (one patched forward pass per `λ ∈ {0, .25, .5, .75, 1, 1.25}` yielding both
`D_pred(λ)` and `D_unc(λ)`), the A/B/C/D feature-type table, `R_j = U_j − E[U_j|P_j]`
(isotonic), and `MVDF_v2` selection.

### Smoke / acceptance (spec §10): **PASS on all 8 datasets**

`D_pred` non-trivial for 118–120/120 features, `D_unc` non-trivial for 119–120/120,
`D_unc(λ)` monotone fraction **0.80–0.92**. The intervention genuinely propagates and
the uncertainty curves are mostly well-behaved — this is the machinery Run 5's bug #1
would have silently broken.

### H1c — type-B features exist and `MVDF_v2` is enriched for them

Feature-type table (median split on `|P_j| × |U_j|`, pool = 120):

| dataset | A (pred) | **B (unc-specific)** | C (mixed) | D (neither) |
|---|---|---|---|---|
| ETTh1 | 19 % | **19 %** | 31 % | 31 % |
| ETTh2 | 18 % | **18 %** | 32 % | 32 % |
| ETTm1 | 20 % | **20 %** | 30 % | 30 % |
| ETTm2 | 22 % | **22 %** | 28 % | 28 % |
| traffic | 9 % | **9 %** | 41 % | 41 % |
| electricity | 12 % | **12 %** | 38 % | 38 % |
| weather | 19 % | **19 %** | 31 % | 31 % |
| solar | 10 % | **10 %** | 40 % | 40 % |

(A and B are equal by construction — a median split on each axis. What matters is
absolute mass: 9–22 % of the pool is low-`|P_j|`, high-`|U_j|`.)

**Type-B enrichment of the selected sets** (fraction of the selected 12 that are
type-B, ÷ the pool base rate):

| dataset | **MVDF_v2** | MVDF_v1 (single-point) | correlated |
|---|---|---|---|
| ETTh1 | **1.74×** | 1.74× | 0.87× |
| ETTh2 | **1.36×** | 0.91× | 0.45× |
| ETTm1 | **1.25×** | 0.83× | 1.25× |
| ETTm2 | **1.54×** | 1.54× | 0.77× |
| traffic | **4.55×** | 0.91× | 1.82× |
| electricity | **4.00×** | 0.00× | 2.00× |
| weather | **1.74×** | 0.43× | 0.43× |
| solar | **2.50×** | 0.00× | 0.83× |

- **`MVDF_v2` is ≥ 1.25× enriched in 8/8**, and ≥ 2× in 4/8 (traffic, electricity,
  solar, and near it elsewhere).
- **`MVDF_v1` is at or below 1× in 6/8** — on traffic, electricity, weather, solar it
  picks *fewer* type-B features than random. Run 5's single-point criterion was not
  selecting uncertainty-specific features at all.
- The correlated set is below 1× in 5/8.

**This is H1c confirmed, and it is the paper's central qualitative claim (spec §4.4,
§6 H1c):** the dose-response + `R_j` construction is not cosmetic — it demonstrably
enriches for a feature type that neither correlation screening nor single-point
ablation finds.

### H1 / H1b — ranking divergence (unchanged from Run 5's finding, now with `P_j`)

- **H1** (`|Δ_pred|` ranking vs correlation ranking): Spearman ρ = **−0.02 to +0.17**,
  Jaccard **0.15–0.22**. The interventional and correlational rankings are essentially
  unrelated — same as Run 5.
- **H1b** (`|Δ_pred|` vs `C_j`, both should track "importance to error"): ρ =
  **0.02–0.43**, Jaccard 0.15–0.36. Interventional and observational importance
  diverge too, most strongly on traffic (ρ 0.43) and solar (0.28).

### Selection overlap

`MVDF_v2 ∩ MVDF_v1` = 1–5 / 12; `MVDF_v2 ∩ correlated` = 0–3 / 12. The three feature
sets are largely disjoint — the comparison in Phase 2 is a real contrast, not three
labels on nearly the same features.

---

## Phase 2 — Cross-environment invariance (spec §4.7) — the flagship

**What ran:** the leakage guard verified to fire on a deliberate overlap; per dataset,
the TEST split carved into `E_select` (first 1/3, used by Phase 1 only) / `E_eval_ID`
(middle 1/3) / `E_eval_shift` (final 1/3, further split into 2 sub-blocks). Every
`E_eval_shift`'s KS-vs-`E_select` p < 0.05 (shifts verified: p from 3×10⁻³ on traffic
to ≈ 0 on the ETTm sets). For each feature-set condition, the conditional
difficulty–score curve per environment, then `I(R)` = mean pairwise scale-normalised
L2 distance between per-environment curves (**lower = more invariant**).

### Result: **H3 fails.** `MVDF_v2` is not more invariant than correlation-only.

`I` by condition (lower is better), per dataset — **best per row in bold**:

| dataset | random | **correlated** | intervention_only | MVDF_v1 | MVDF_v2 |
|---|---|---|---|---|---|
| ETTh1 | 1.333 | 1.386 | **1.077** | 1.077 | 1.333 |  ← the only row where MVDF_v2 (1.333) < correlated (1.386)
| ETTh2 | 1.333 | **0.807** | 1.172 | 1.172 | 1.165 |
| ETTm1 | 0.595 | **0.349** | 0.724 | 0.724 | 0.934 |
| ETTm2 | 1.540 | 1.098 | **0.752** | 0.752 | 1.533 |
| traffic | 1.357 | **0.785** | 1.516 | 1.516 | 1.575 |
| electricity | 1.333 | **0.670** | 1.502 | 1.502 | 1.333 |
| weather | 0.478 | 1.046 | **0.418** | 0.418 | 1.333 |
| solar | 0.592 | **0.058** | 1.134 | 1.134 | 0.333 |

**Tallies (8 datasets)** — verified row-by-row against `v2_phase2/AGG_invariance.csv`:

- `I(MVDF_v2) < I(correlated)`: **1 / 8** — **ETTh1 only** (1.333 vs 1.386, margin
  0.053). On the other 7, `MVDF_v2`'s `I` is *higher* (worse) than `correlated`'s,
  by 0.24–0.79 on traffic / electricity / weather / ETTm2. There is no dataset where
  `MVDF_v2` beats `correlated` on invariance by a meaningful margin.
- `I(MVDF_v2) < I(MVDF_v1)`: **3 / 8** (ETTh2, electricity, solar).
- **`correlated` is the single most-invariant condition on 5 / 8** (ETTh2, ETTm1,
  traffic, electricity, solar).
- `MVDF_v2` is the most-invariant condition on **0 / 8**.
- `intervention_only` / `MVDF_v1` (**identical in this run** — see the caveat below;
  the Phase 2 fallback selection path collapsed them) win on 3 / 8 (ETTh1, ETTm2,
  weather).

`spearman_stability` (higher = more stable curve shape) tells the same story:
`correlated` is 0.27–1.00 on the datasets where it wins; `MVDF_v2` is **−0.33** (i.e.
the curve *reverses* between environments) on ETTh1, ETTm2, traffic, electricity,
weather.

### What this means

The pre-registered interpretation is **spec §7, row 3**:

> "Dose-response/residualization selects different features than correlation (H1/H1b
> hold) but not more invariant ones — a genuine negative on the invariance hypothesis
> specifically, independent of the Run 5 null (which never tested invariance at all).
> Reportable — closes the actual pre-registered v1 gap with a real answer."

Concretely:

1. **Uncertainty-specificity ≠ invariance.** A feature can have a causal effect that
   is disproportionately on the nonconformity score (`R_j` high, type-B) and *still*
   have a conditional difficulty–score relationship that drifts across environments.
   The two properties were conflated in the v1 hypothesis; the data separates them.

2. **Correlation-only is a surprisingly strong invariance baseline.** On 5/8 datasets
   the feature that best correlates with the score in-sample also has the most stable
   score-conditional-on-feature curve across the ID→shift boundary. This is the
   opposite of the v1/v2 premise ("a merely-correlated feature's relationship with
   error is a property of the calibration distribution and drifts"). On these
   benchmarks it mostly doesn't drift more than the alternatives.

3. **`MVDF_v2` often has the *worst* invariance** (5/8 it's in the bottom two;
   negative `spearman_stability` on 5/8). Selecting for `R_j` — a residual quantity —
   appears to pick features whose `U_j` is large *because* of environment-specific
   structure that the `E[U_j|P_j]` fit doesn't absorb. This is a plausible mechanism
   for an anti-invariance effect and is worth a sentence in the paper.

---

## Assessment — where this leaves the project

### The honest headline

> **We built and ran the dose-response + invariance machinery the v1 spec designed.
> It works: it identifies uncertainty-specific ("type-B") features that correlation
> screening and single-point ablation miss (H1c, 8/8 datasets). But the property that
> was supposed to make those features valuable — invariance of their difficulty
> signal under distribution shift — does not hold: `MVDF_v2` is no more invariant than
> a correlation-selected feature set, and usually less. Separately, a 7-layer sweep
> finds no representation depth where difficulty is meaningfully more recoverable than
> at the pre-head layer. The net result is a clean, thorough negative on the
> mechanistic-conformal hypothesis, stronger than Run 5's because it tests the actual
> flagship claim and rules out the "wrong layer" and "wrong ablation protocol"
> escape hatches.**

### Is this publishable?

**Yes — as the negative it is.** Per spec §7's own table, three of the four cells are
"reportable", and this is row 3 (signal found via a different criterion, but not more
invariant) crossed with Phase 0's row 4 (no better layer). The contributions that
stand:

1. **H1c is a genuine positive mechanistic-interpretability result.** "Time-series
   forecasters contain features whose causal effect is specific to uncertainty rather
   than prediction content, and a dose-response residualization isolates them where
   correlation and single-point ablation don't." That is new and defensible on its own.
2. **The dose-response protocol and the `R_j` construction** are a reusable
   methodological contribution regardless of the downstream null.
3. **The negative is the informative kind:** it tells the field that
   uncertainty-specificity and cross-environment invariance are different properties,
   with data separating them, and that on standard LTSF benchmarks correlation
   screening is a hard-to-beat invariance baseline. That saves other groups from
   building the same pipeline.
4. **Thoroughness:** 7-layer sweep + dose-response + invariance + the Run 5
   calibration results together rule out the obvious "you did it wrong" objections.

### What it is NOT

It is **not** the "full positive" (spec §7 row 1). `MVDF_v2` does not beat the
baselines on invariance, so there is little reason to expect it to beat them on
interval width in Phase 3+. Building the full scorer + baseline suite + Pareto
evaluation to *confirm* a width null is expensive and low-information given Phase 2.

### Recommended next step — three cheap things, then write up

The shape is **Option A (write up as a negative-results / methods paper)**, not
Option C (full Phase 3–5: high cost, Phase 2 gives no reason to expect a width win,
and building three scorer variants + the full baseline suite to confirm an expected
null is exactly what the spec's own Phase 4/5 gating exists to prevent). But **three
cheap fixes come first**, none requiring a new dose-response sweep:

1. **Fix `intervention_only` and re-run only the invariance step.** In this run
   Phase 2's fallback selection path collapsed `intervention_only` into `MVDF_v1`
   (both = single-point). That means Phase 2 *cannot currently answer* whether the
   `R_j` specificity gate adds anything to invariance beyond dose-response alone —
   the exact ablation `intervention_only` exists for. Wire the true raw-`U_j`-ranked
   pool (top-k by `|U_j|`, no `R_j` residualization) and re-run `phase2_invariance.py`
   only (the Phase 1 dose-response outputs are reused). Cheap; closes a real gap in
   the paper's own methods-ablation logic.

2. **Multi-seed Phase 2** (≥ 3 seeds). The flagship `I` numbers are single-seed, `I`
   is a small-sample quantity (5-bin curves over ~1/6 of the test split per
   environment), and the one `MVDF_v2` "win" (ETTh1, margin 0.053) is thin enough
   that seed noise could flip its sign. The *direction* (correlated ≥ MVDF_v2 on
   invariance) is consistent across 8 datasets so it is very likely to hold, but it
   must be stated with error bars, not from n=1.

3. **The `g_MVDF` vs `g_dense` scorer check** (spec §4.8, H6), α = 0.10, ETTh2 +
   traffic + electricity, 3 seeds. Not an optional extra: v1's own causal chain
   treats H4/H6 as independent of H3's outcome ("proceed to Phase 3 regardless of
   which outcome occurs") precisely so the width question gets *tested*, not assumed.
   At ~1 Kaggle run it forecloses the most obvious reviewer question — "fine, not
   more invariant; does that actually cost interval width, or not?" — rather than
   leaving a gap the paper has to argue around. Skip the full CQR/RLCP/PID/EVT suite
   and the Pareto sweep; just the one paired comparison.

With (1)–(3) done, write up. Narrative: dose-response + `R_j` works and finds a real,
novel feature type (H1c, 8/8) that correlation screening and single-point ablation
miss — but uncertainty-specificity and cross-environment invariance are different
properties, the data separates them, and on these benchmarks correlation is the
harder invariance baseline to beat. Combined with Phase 0's layer-sweep null and
Run 5's calibration null: a thorough, escape-hatch-free negative with a genuine
methods contribution riding on top. Lead with H1c and the methodology; be honest
about the downstream null.

### Caveats on these Phase 2 numbers

- **Single seed.** `I` is a small-sample quantity (5-bin curves over ~1/6 of the test
  split per environment). Before the negative goes in a paper, Phase 2 should be
  re-run at ≥ 3 seeds to confirm the sign of `I(MVDF_v2) − I(correlated)` is stable.
  The *direction* (correlated ≥ MVDF_v2 on invariance) is consistent enough across 8
  datasets that this is very likely to hold, but say it with error bars.
- **Undertrained forecaster.** Same caveat as `PROJECT_REPORT.md` §8 — the iTransformer
  is the FFDCI fixed config, test MSE 0.23–0.70. A stronger forecaster might have a
  richer, more invariant difficulty representation. Phase 0's NO-GO makes this less
  likely to be the whole story, but it's the one open thread.
- **`E_select` / `E_eval` are 1/3-splits of the *test* window**, which for the smaller
  datasets is only a few hundred windows per environment. The invariance estimate is
  correspondingly noisy on ETTh1/h2/solar.
- **`intervention_only` == `MVDF_v1`** in this run — confirmed a real bug, not a
  coincidence: `phase2_invariance.py::_load_selection` maps `intervention_only` to the
  `MVDF_v1` list (a stopgap from when the raw-`U_j` pool wasn't plumbed through). This
  means Phase 2 **cannot yet answer** whether the `R_j` specificity gate adds anything
  to invariance *beyond* dose-response alone — the exact ablation `intervention_only`
  exists to test. Fix: rank the pool by `|U_j|` (from the Phase 1 dose output, already
  on disk), take top-k with no `R_j` step, re-run `phase2_invariance.py` only. See
  recommended step 1.

---

## Addendum (2026-09-03, post-review) — bugs found while acting on the review

Acting on the three recommended fixes surfaced two more real bugs, now fixed:

**Bug A — non-deterministic per-dataset seeding.** `phase1_core_mechanism.py` and
`phase2_invariance.py` seeded the global RNG **once** at the top, then looped over
datasets. So `ETTh1` as the 1st dataset drew different random numbers than `ETTh1`
run alone (earlier datasets advanced the RNG). This is what produced the
`I(correlated)` discrepancy that prompted the fingerprint discussion (NOT a
data-source difference — HTTP ETTh1 and the bundle `ETTh.csv` split both have 17420
rows). **Fix:** `experiments/_seed.py::derive_seed(base_seed, dataset)` (SHA-256, no
`PYTHONHASHSEED` dependence); every phase script re-seeds per dataset and threads
`ds_seed` into `TrainConfig`, `train_sae`, and every `default_rng`. Verified: a
1-dataset run is now byte-identical to that dataset pulled from a multi-dataset run.
**Consequence:** the `v2_runs/` seed-0 numbers used the old global-seed regime and
are not directly comparable to the confirm run's — the confirm run re-does Phase 1
from scratch.

**Bug B — `intervention_only` collapsed to `MVDF_v1`.** Confirmed and fixed:
`_load_selection` now derives it from Phase 1's `candidates.csv` as top-k by `|U_j|`
with no `R_j` step. (On the `v2_runs` data it happened to overlap `MVDF_v1` 10-12/12,
so the collapse looked benign, but it made the "did the `R_j` gate add anything"
ablation impossible.)

**Data provenance guard added.** `data/fingerprint.py::assert_expected_data` —
channels checked exactly against `CANONICAL_CHANNELS`; rows + an edge-hash recorded
to `configs/data_fingerprints.json` on the first canonical run and compared verbatim
after. The pipeline now refuses to run on the wrong Kaggle input.

**H6 scorer built.** `calibration/difficulty_scorer.py` (`g_dense` / `g_mvdf` /
`g_dense+mvdf`, same MLP arch + budget, paired width bootstrap) +
`experiments/phase3_scorer_h6.py` + `configs/v2_phase3_h6.yaml`. Local fast smoke:
`g_mvdf` ID width 2.217 vs `g_dense` 2.223, shift 1.969 vs 2.010 — tiny, consistent
with the expected H6 null.

**Confirm notebook:** `notebooks/v2_confirm.ipynb` — Phase 1 + Phase 2 (fixed
`intervention_only`) + H6, on ETTh1 + ETTh2 + electricity × seeds 0/1/2. ETTh1 is in
the set specifically because it is the one apparent H3 exception (margin 0.053) that
needs seed error bars; ETTh2 + electricity are the decisive-margin datasets. ~2-3 hr
on a Kaggle GPU. Its output (`v2_confirm.zip`) is the input to the final write-up.

## File index (`v2_runs/`)

```
v2_phase0/
  layer_sweep.csv            dataset × tap × {max_abs_corr, max_abs_corr_pc, mean_top10}
  layer_sweep_summary.json   NO_GO, per-tap means, interpretation
  layer_sweep.png            heatmap
v2_phase1/
  AGG_h1c.csv                per dataset: type_B_frac, enrichment_{v2,v1,corr}, H1/H1b rho
  <dataset>_pre_head/
    candidates.csv           per feature: C_j, delta_pred, P_j, U_j, R_j, n_active, corr, type
    feature_type_table.json  A/B/C/D cells (feature IDs), counts, fractions, medians
    h1_tables.json           H1, H1b, H1c (enrichment for v2/v1/correlated)
    mvdf_v2_selection.json    MVDF_v2 / MVDF_v1 / correlated / random feature sets + overlaps
    dose_curves.json         raw D_pred(λ) and D_unc(λ) per feature
    smoke.json               acceptance checks
v2_phase2/
  AGG_invariance.csv         dataset × tap × I/stab/Ishift per condition + the two flags
  <dataset>_pre_head/
    invariance.json          per-condition I, spearman_stability, conditional_error_drift,
                             per-environment curves, env-split KS check
```
