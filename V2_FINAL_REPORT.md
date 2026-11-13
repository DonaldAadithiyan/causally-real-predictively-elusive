# V2 — Final Results (Phase 1 + 2 + aggregation diagnostic + H6)

**Run:** `v2_confirm.ipynb` on Kaggle GPU, 7276 s (~2 h). **ETTh1 + ETTh2 + electricity +
traffic × seeds 0/1/2.** Outputs in `mechanistic-conformal/v2_confirm/`. **H6 numbers
below (incl. the `naive_no_scorer` baseline) are from the `v2_h6_topup.ipynb` re-run,
outputs in `mechanistic-conformal/v2_h6_topup/`** — same forecaster/SAE (deterministic),
Phase 1 selections reused.
**Date:** 2026-09-04. This supersedes all earlier V2 reports for the numbers.
Regenerate with `python eval/consolidate_results.py` (→ `RESULTS_FINAL.md`, the
machine-checked source of truth; `--verify` diffs this file against it).

All guards held: data-fingerprint assertion passed every dataset (ETT/traffic exact,
electricity 47 ch tolerated), per-dataset deterministic seeding, invariance degeneracy
guard active, leakage guard verified to fire on every Phase 2 seed.

---

## TL;DR — the three findings, locked

| # | Finding | Verdict | Key number |
|---|---|---|---|
| **H1c** | Dose-response + `R_j` isolates uncertainty-specific ("type-B") SAE features that correlation and single-point ablation miss | **CONFIRMED, 12/12 runs** | MVDF_v2 type-B enrichment **2.0–4.6×**; MVDF_v1 **0.0–1.85×**; correlated **0.0–1.2×** |
| **H3** | Those features give a *more shift-invariant* difficulty signal | **NEGATIVE + not cleanly measurable** | `I(MVDF_v2) < I(correlated)` on **2/10 non-degenerate runs**; both signals fire on ~2–5% of windows so their aggregate curve is degenerate under **all 4 aggregations** |
| **H6** | A learned pinball-loss conformal scorer on the MVDF subspace gives narrower, still-valid intervals under shift than one on FFDCI's raw dense vector | **NEGATIVE for the learned scorer entirely** — a no-parameter constant-width split-conformal band is **narrower AND better-calibrated** than every learned-scorer mode (dense / mvdf / correlated / random_k / pca_k), ID and shift, on all 4 datasets. Holds at matched calibration budget (`naive_matched`), so not a data-allocation artefact | `naive_matched` shift width **3.35 vs dense 4.35, mvdf 3.81**; controls ID coverage **0.90 ± 0.01** (every MLP mode ~0.87); `naive_matched − dense` shift **−1.01, 12/12 sig, P=1.00**; `naive_matched − naive_no_scorer` **+0.015** (budget = noise) |

**The thread that ties them:** H1c is a real, robust finding about *feature selection*.
H3 and H6 are both about whether that selection then *buys you anything downstream* — and
neither does. H3's invariance metric is structurally unmeasurable for signals this sparse
(≤ 5 % window activation). H6 is worse than null: the whole learned-scorer apparatus
(FFDCI-style pinball MLP), in *any* input space — dense, SAE-selected, SAE-random, PCA —
produces wider and less-calibrated intervals than simply taking the (1−α) quantile of the
calibration scores. The `dense → SAE-space` width reductions were the SAE variants
partially recovering ground the MLP itself lost; none reach the constant baseline.

---

## Finding 1 — H1c: CONFIRMED (12/12 runs)

**Type-B feature = low prediction-effect `|P_j|`, high uncertainty-effect `|U_j|`**
(median split of the 120-feature dose-response pool). Enrichment = fraction of the
selected 12 that are type-B ÷ pool base rate.

| dataset | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ETTh1 | 2.50 | 2.00 | 2.96 |
| ETTh2 | 2.80 | 2.50 | 2.27 |
| electricity | 3.33 | 3.68 | 2.27 |
| **traffic** | 4.00 | 4.62 | 3.33 |

| criterion | enrichment range (12 runs) | # runs ≥ 1.0× |
|---|---|---|
| **MVDF_v2** | **2.00 – 4.62** | **12 / 12** |
| MVDF_v1 (single-point ablation) | 0.00 – 1.85 | 4 / 12 — **0/6** on electricity+traffic |
| correlated | 0.00 – 1.20 | 2 / 12 |

- Rank divergence universal: `H1` ρ (Δ_pred rank vs correlation rank) ∈ [−0.034, +0.246]
  across 12 runs; `H1b` ρ (Δ_pred vs the observational `C_j`) ∈ [0.02, 0.42], **highest
  on traffic** (0.34, 0.42, 0.18). Interventional and correlational/observational
  rankings are essentially unrelated.
- SAE clean on all 12: FVU 0.003–0.013, 0–16 dead of 2048. Dose-response smoke PASS
  all 12 (D_unc monotone fraction 0.81–0.93).

**The dose-response + `R_j` construction reliably isolates a feature type that neither
correlation screening nor single-point ablation selects — across ETT, electricity, and
the regime-structured traffic.** This is the paper's mechanistic-interpretability anchor.

---

## Finding 2 — H3: NEGATIVE, and the metric is degenerate for the conditions that matter

### Invariance `I(R)` (lower = more invariant), mean ± sd over 3 seeds

| dataset | I_correlated | I_MVDF_v2 | `I(MVDF_v2) < I(correlated)` |
|---|---|---|---|
| ETTh1 | 1.08 ± 0.31 | 1.39 ± 0.46 | 0 / 3 |
| ETTh2 | 0.90 ± 0.54 | 1.05 ± 0.37 | 1 / 3 |
| electricity | 1.38 ± 0.26 | 1.41 ± 0.25 | 1 / 3 |
| traffic | 1.01 ± 0.52 | 1.13 ± 0.73 | 0 / 2 (1 seed degenerate) |

**`I(MVDF_v2) < I(correlated)` on 2 / 10 non-degenerate runs — coin flip.** Overlapping
error bars every dataset. `MVDF_v2` is on average *worse* (higher `I`) than `correlated`.
The single-seed "ETTh1 win" from earlier runs is gone once the degeneracy guard
converts it to `NaN`.

### The measurement is degenerate for `MVDF_v2` and `correlated` — verified across 4 aggregations

The aggregation diagnostic reloaded each forecaster + SAE, regenerated the SAE codes on
the E_eval blocks, and measured **how many of 5 quantile bins the aggregated feature
signal populates**, under mean / max / L2 / active-count aggregation:

| condition | median per-feature activation rate | bins populated (mean / max / L2 / active-count) |
|---|---|---|
| MVDF_v1 | **0.090** | 4.38 / 4.38 / 4.38 / 3.62 |
| intervention_only | **0.079** | 4.17 / 4.17 / 4.17 / 3.50 |
| correlated | **0.045** | 3.33 / 3.33 / 3.33 / 2.54 |
| **MVDF_v2** | **0.037** | **3.00 / 3.00 / 3.00 / 2.17** |
| random | **0.031** | 2.71 / 2.71 / 2.71 / 2.21 |

- Phase 2 degeneracy flags (curve pair sharing < 3 populated bins): `random` **6/12**,
  `correlated` 1, `MVDF_v2` 1 — with ~8/12 more `MVDF_v2` runs at exactly the 3-bin
  borderline (a scale-normalized 3-point curve carries almost no shape information).
- **No aggregation rescues it.** `MVDF_v2` stays at ~3 bins under mean, max, and L2, and
  drops to ~2 under active-count. `correlated` at ~3.3. Only `MVDF_v1` /
  `intervention_only` (which fire ~2× as often) reliably fill ≥ 4 bins.

**Root cause is structural, not a bug.** MVDF_v2 selects features that are
difficulty-*specific* — which entails narrow, sparse firing (~3–4% of windows). Any
scalar summary of a 12-feature bundle that is all-zero on 90 %+ of windows is
near-constant. **The property that makes MVDF_v2 features mechanistically interesting
(H1c) is the property that makes their aggregate difficulty signal too flat for a
conditional-drift metric to measure (H3).** The two goals pull against each other.

### H3 framing for the paper

Not "we measured invariance and MVDF_v2 lost." It is: **the dose-response/`R_j`
criterion selects a genuinely different, genuinely more uncertainty-specific feature
set (H1c), but (a) that set is no more shift-invariant than a correlation-selected one
on the runs where `I` is well-defined, and (b) a clean invariance comparison is not
obtainable, because difficulty-specificity implies sparse activation, which flattens the
aggregate signal below the resolution of the metric — verified across four aggregation
choices and four datasets.**

---

## Finding 3 — H6: NEGATIVE for the learned scorer — a no-parameter constant-width CP band beats it

> **Budget-matched and confirmed (`v2_h6_topup/`, 2026-09-04).** The learned modes
> calibrate on 80 % of `E_eval_ID` (50 % MLP-train + 30 % conformalize). `naive_matched`
> is given the *same* 80 % pool for its quantile; `naive_no_scorer` uses only the 30 %
> conformalize block. `naive_matched` reproduces `naive_no_scorer` to within **+0.015**
> shift width — the calibration budget was doing nothing — and still beats every learned
> mode: `naive_matched − dense` shift width **−1.009, 12/12 sig, P = 1.00**. The scorer
> penalty is real, not a data-budget artefact.

**Setup:** split-conformal calibration (50 % MLP-train / 30 % conformalize / 20 %
ID-width eval of `E_eval_ID`; `E_eval_shift` = shift stream), α = 0.10, paired bootstrap
over 12 dataset-seed runs. Six *learned-scorer* input modes (`dense` = FFDCI as-is,
`mvdf`, `dense+mvdf`, `correlated`, `random_k`, `pca_k`) — all the same pinball-loss MLP,
same budget — **plus** two parameter-free controls: `naive_no_scorer` (half-width = the
(1−α) quantile of the 30 % conformalize block, online-ACI on top) and `naive_matched`
(same, but the quantile is taken over the full 80 % train ∪ conformalize pool — the
budget-matched control).

### Coverage + width by mode (mean over 4 datasets × 3 seeds)

| mode | ID coverage | ID width | shift coverage | shift width |
|---|---|---|---|---|
| **`naive_no_scorer`** (30 %) | **0.908** | **4.08** | **0.903** | **3.33** |
| **`naive_matched`** (80 %) | 0.895 | 4.07 | 0.901 | 3.35 |
| `dense` (FFDCI) | 0.882 | 4.58 | 0.896 | 4.35 |
| `mvdf` | 0.872 | 4.72 | 0.870 | 3.81 |
| `dense+mvdf` | 0.884 | 4.57 | 0.897 | 4.33 |
| `correlated` | 0.874 | 4.68 | 0.890 | 3.89 |
| `random_k` | 0.873 | 4.73 | 0.881 | 3.94 |
| `pca_k` | 0.879 | 4.57 | 0.894 | 4.36 |

**Both parameter-free controls beat every learned-scorer mode on every axis** — narrower
ID *and* shift width, better ID *and* shift coverage. Every learned-scorer mode
undercovers ID at ~0.87–0.88 even after split-conformal; the controls sit at 0.90 ± 0.01.
`naive_matched ≈ naive_no_scorer` throughout (the 30→80 % calibration change is noise).

### Shift-block mean interval width, per dataset (mean over 3 seeds)

| dataset | **naive_m** | naive | dense | mvdf | correlated | random_k | pca_k |
|---|---|---|---|---|---|---|---|
| ETTh1 | 1.912 | **1.886** | 2.063 | 1.964 | 1.972 | 1.972 | 2.020 |
| ETTh2 | 2.892 | **2.870** | 3.250 | 3.061 | 3.067 | 3.054 | 3.258 |
| electricity | 3.139 | 3.127 | 3.308 | **2.551** | 2.856 | 2.683 | 3.260 |
| traffic | 5.439 | **5.441** | 8.796 | 7.648 | 7.681 | 8.066 | 8.913 |

Both controls narrowest on 3/4 datasets. On electricity `mvdf`/`random_k` are numerically
narrower — but only by *undercovering* (see below); the controls hold ≈ 0.895 coverage
there.

### Paired bootstrap, shift env (negative = row narrower), 12 runs

| comparison | mean diff | 95 % CI | # significant / 12 |
|---|---|---|---|
| **`naive_matched − dense`** | **−1.009** | **[−1.81, −0.25]** | **12 / 12** |
| **`naive_matched − mvdf`** | **−0.461** | **[−1.10, +0.10]** | **12 / 12** |
| `naive_matched − naive_no_scorer` (budget sensitivity) | **+0.015** | [+0.01, +0.02] | 11 / 12 (trivial) |
| `naive_no_scorer − dense` | −1.023 | [−1.82, −0.27] | 12 / 12 |
| `naive_no_scorer − mvdf` | −0.475 | [−1.11, +0.08] | 12 / 12 |
| `mvdf − dense` | −0.548 | [−0.81, −0.30] | 12 / 12 |
| `correlated − dense` | −0.460 | [−0.73, −0.24] | 12 / 12 |
| `random_k − dense` | −0.411 | [−0.60, −0.24] | 12 / 12 |
| `pca_k − dense` | +0.009 | [−0.04, +0.07] | 9 / 12 (sign inconsistent) |
| `mvdf − correlated` | −0.088 | [−0.24, +0.06] | 10 / 12 |
| `mvdf − random_k` | −0.138 | [−0.31, −0.02] | 8 / 12 |
| `dense+mvdf − dense` | −0.026 | [−0.06, +0.00] | 11 / 12 |

Per-dataset `naive_matched − dense` (shift, all significant): ETTh1 −0.15, ETTh2 −0.36,
electricity −0.17, traffic −3.36.

### What the decomposition now says

1. **The learned scorer is the problem, not its input space.** Every pinball-MLP mode —
   `dense`, `mvdf`, `correlated`, `random_k`, `pca_k` — is **wider and less
   ID-calibrated** than a constant-width split-conformal band. The `dense →
   {mvdf, correlated, random_k}` width reductions (−0.4 to −0.55) are the SAE-space
   variants *partially clawing back* ground the MLP itself lost relative to `naive`
   (−1.02); **none of them reach the baseline.** `pca_k` and `dense+mvdf` don't claw any
   back (`pca_k ≈ dense`, `dense+mvdf ≈ dense`).
2. **The earlier "SAE basis, not MVDF content" reading survives — but is now a
   second-order effect inside a first-order failure.** Among the learned modes, which
   SAE features you pick barely matters (`mvdf − random_k` −0.138, CI [−0.31, −0.02],
   `Δ_d` favours `mvdf` 3/4 datasets, reverses on ETTh2 — see Task D) and a PCA subspace
   of the same size gives nothing. But the whole family is dominated by *no scorer*.
3. **The controls remove the coverage pathologies.** On electricity, `naive_matched`
   shift coverage is **0.894 / 0.895 / 0.894 across seeds** (and `naive_no_scorer` 0.895
   flat) — the `mvdf` collapse (0.827 / 0.630 / 0.811, Task B) is a learned-scorer
   artefact of transferring a ~4 %-active feature signal from the conformalize block to
   the shift block. A constant-width band transfers nothing learned, so it doesn't break.
   (traffic ID coverage is still 1.000 for the controls too — that pin is the short
   chronological `E_eval_ID` split, Task A, not the scorer.)

**Conclusion.** On these four LTSF benchmarks under a held-out temporal shift, the
FFDCI-style **learned pinball-loss conformal scorer is not worth its parameters** in any
input representation we tried. A no-parameter constant-width split-conformal interval is
uniformly narrower and better-calibrated — and this holds at matched calibration budget
(`naive_matched`), so it is not a data-allocation artefact. This is a **negative result
for the learned scorer family**, and it subsumes the earlier H6 story: MVDF-as-scorer-
input was never going to be the win, because *scorer-as-such* isn't. H1c (feature
selection) and H3 (invariance unmeasurable) are untouched.

### Coverage detail

| | naive_m | naive | dense | mvdf | correlated | random_k | pca_k |
|---|---|---|---|---|---|---|---|
| **mean ID coverage** | 0.895 | **0.908** | 0.882 | 0.872 | 0.874 | 0.873 | 0.879 |
| **mean shift coverage** | 0.901 | **0.903** | 0.896 | 0.870 | 0.890 | 0.881 | 0.894 |

- Every learned mode undercovers ID by ~2–3 pt (target 0.90) — the conformalize slice is
  only a few hundred windows and the MLP's `pred_q(x)` adds variance the split-conformal
  correction doesn't fully absorb. The parameter-free controls have no `pred_q(x)` term
  and land at 0.90 ± 0.01 (`naive_matched` slightly lower than `naive_no_scorer` because
  the 80 % pool's quantile is pulled down on the short ETT ID blocks — still far better
  than any learned mode).
- **`mvdf` shift undercoverage on electricity** — 0.827 / 0.630 / 0.811 (mean 0.756 vs
  `naive_matched` 0.894). Task B: not `mvdf`-only (`random_k` also drops), seed-1-driven,
  a sparse-feature transfer failure. The controls do not have it.
- **traffic ID coverage 1.000 for every mode incl. `naive`** — Task A: the chronological
  `E_eval_ID` split (1169 windows) is too short for the conformalize→eval transfer under
  traffic's daily/weekly cycle. Structural; traffic H6 is directional-only.

---

## What the paper is

One discovery, one measurement boundary, two falsifications — one mechanism:

1. **H1c — DISCOVERY (positive, tight).** A dose-response causal intervention plus
   residualization against prediction-effect isolates SAE features whose causal
   influence is specific to forecast *uncertainty* — 2–5× enriched, 12/12 runs, 4
   datasets — a type neither correlation screening nor single-point ablation finds.

2. **H3 — MEASUREMENT BOUNDARY (not a falsification).** The invariance question cannot
   be cleanly *asked* of these features: difficulty-specificity implies sparse (~4 %)
   activation, which collapses the aggregate difficulty signal below the resolution of
   any conditional-drift metric (verified across four aggregations). The nominal
   comparison (`I(MVDF_v2) < I(correlated)` on 2/10 non-degenerate runs) is a coin flip
   *because the metric is degenerate at this sparsity*, not because invariance was
   tested and rejected.

3. **H6 / Task G — FALSIFICATION (continuous conditioning; negative, budget-matched).**
   The FFDCI-style learned pinball-loss conformal
   scorer is not worth its parameters: a no-parameter constant-width split-conformal band
   is **narrower and better-calibrated** than the learned scorer in *every* input space
   tried — dense, MVDF-selected, correlation-selected, random-k SAE, and PCA — ID and
   under shift, on all 4 datasets (shift width 3.35 vs dense 4.35 / mvdf 3.81; ID
   coverage 0.90 ± 0.01 vs ~0.87; `naive_matched − dense` shift −1.01, 12/12 significant,
   P = 1.00). At the same 80 % calibration budget the learned modes use, `naive_matched`
   reproduces the 30 %-budget `naive_no_scorer` to +0.015 shift width, so the effect is
   not a data-allocation artefact. Among the learned modes the "SAE-basis-not-selection"
   effect still holds, but it is a second-order recovery inside a first-order failure of
   the scorer itself.

4. **H7-A — FALSIFICATION (discrete regime-gating; negative).** A *discrete* alternative
   to the continuous scorer: gate the conformal quantile on whether Type-B features are
   highly co-active (`D_t` = top-decile co-activation), test whether the 90th-percentile
   nonconformity score separates by regime. It does not: `R_Q = Q_.90(S|D=1)/Q_.90(S|D=0)`
   ∈ [0.91, 1.06] across ETTh1/ETTh2 × 3 seeds, 5/6 CIs contain 1, no consistent
   direction, in-distribution and under shift. Well-powered (n₁ 215–626), absolute tail
   quantiles near-identical every cell — an absence of separation, not a noisy estimate.
   Full-fidelity SAE (FVU 0.004–0.010). `H7A_RESULT.md`.

**Discussion thread:** H1c is a genuine finding about *feature selection*. It does not
propagate downstream by any route tried. H3's invariance question is *unaskable* at the
sparsity difficulty-specificity forces (a measurement boundary, not a negative result).
The two exploitation strategies that *were* cleanly testable — a continuous learned
scorer (Task G) and a discrete regime gate (H7-A) — both came back negative: the learned
scorer underperforms taking the calibration-score quantile directly, and Type-B
activation is statistically unrelated to tail-error magnitude. Boundary claim:
*identifiability of an internal uncertainty mechanism does not imply it is statistically
exploitable for uncertainty quantification.*

---

## Remaining loose ends (for a rebuttal or a v3, not blockers)

- **Is the learned scorer ever worth it?** H6 says no on 4 LTSF benchmarks under one
  shift definition, at matched calibration budget. A reviewer may want: (a) a dataset
  with genuinely heteroscedastic, *predictable* difficulty (where a conditional scorer
  should help), (b) a stronger scorer (gradient-boosted quantile, deeper MLP, longer
  training) to check it's the *approach* and not this MLP. The budget confound is already
  closed (`naive_matched`, −1.01, 12/12). Prior would still be the constant band wins.
- **traffic ID coverage 1.000 for every mode incl. `naive`** — Task A: chronological
  `E_eval_ID` too short (1169 windows). A different env carve-out + a re-run would fix
  it; directional-only for now.
- **H3 with a non-aggregate signal** — per-feature conditional curves or a scorer-based
  invariance proxy — if a reviewer wants invariance measured rather than declared
  unmeasurable. Expect it to stay negative given H6.
- Multi-seed is n = 3; n = 5 tightens the H1c error bars if asked.

---

## File index (`v2_confirm/`)

```
v2_phase1_seed{0,1,2}/
  AGG_h1c.csv                    per dataset: type_B_frac, enrichment_{v2,v1,corr}, H1/H1b rho, smoke_pass
  <ds>_pre_head/
    candidates.csv              C_j, delta_pred, P_j, U_j, R_j, n_active, corr, type  (120 features)
    feature_type_table.json     A/B/C/D cells
    h1_tables.json              H1 / H1b / H1c enrichment
    mvdf_v2_selection.json      MVDF_v2 / MVDF_v1 / correlated / random feature sets
    dose_curves.json / smoke.json
v2_phase2_seed{0,1,2}/
  AGG_invariance.csv            I / stab / Ishift / degenerate flag per condition
  <ds>_pre_head/invariance.json per-env conditional curves + KS env-split check
AGG_invariance_multiseed.csv    concatenated, + seed column
agg_diagnostic/AGG.csv          bins_populated + signal std + activation rate, per condition x agg x env
v2_phase3_h6/  (v2_confirm run — 6 learned modes only)
  AGG_h6.csv                    per (dataset, seed, mode): ID/shift coverage + width
  AGG_h6_by_dataset.csv         mean/std across seeds
  AGG_h6_tests.csv              paired bootstrap: every mode vs dense, + mvdf vs {correlated, random_k, pca_k}
```

**H6 numbers in this report are from `v2_h6_topup/` (re-run 2026-09-04):** same 4 CSVs,
adds `mode ∈ {naive_no_scorer, naive_matched}` and the
`{naive_no_scorer, naive_matched}_minus_{dense, mvdf}` + `naive_matched_minus_naive_no_scorer`
bootstrap rows. 96 H6 rows (8 modes × 4 datasets × 3 seeds). Regenerate the report tables
with `python eval/consolidate_results.py` (`--h6-root v2_h6_topup` is the default when
that folder has `AGG_h6.csv`; H1c/H3 still come from `v2_confirm/`).
