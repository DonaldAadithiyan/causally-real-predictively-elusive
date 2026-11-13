# V2 Confirm Run — Results & Updated Assessment

**Run:** `v2_confirm.ipynb` on Kaggle GPU. Phase 1 + Phase 2 (fixed `intervention_only`) +
Phase 3 H6, on **ETTh1 + ETTh2 + electricity × seeds 0/1/2**, with the data-fingerprint
guard and per-dataset deterministic seeding. Outputs in `mechanistic-conformal/v2_confirm/`.
**Date:** 2026-09-03. Supersedes the single-seed numbers in `V2_PHASES012_REPORT.md`.

**Why this run:** three fixes from the review — (1) `intervention_only` was collapsed to
`MVDF_v1`, making the "does the `R_j` gate add anything?" ablation impossible; (2) the
flagship `I` numbers were single-seed and the one apparent `MVDF_v2` win (ETTh1) had a
0.053 margin; (3) H6 (scorer interval width) was untested. Plus two bugs found while
building the run: non-deterministic per-dataset seeding, and a fingerprint false-positive
(electricity has 47 channels, not the 48 I hard-coded).

---

## TL;DR — the picture is now clearer and less favourable than the single-seed run suggested

| Claim | Single-seed run said | Confirm run (3 seeds) says |
|---|---|---|
| **H1c** — MVDF_v2 enriched for type-B features | 1.4–4.5× on 8/8 | **2.0–3.7× on 3/3 datasets × 3/3 seeds** — holds, tight |
| **H3** — MVDF_v2 more invariant than correlated | 1/8 (ETTh1, margin 0.053) | **ETTh1's "win" does not survive seeds** — 1/3 seeds each dataset, `I(MVDF_v2)` sd is huge (0.86 on ETTh1). No dataset where MVDF_v2 is reliably more invariant. |
| **Does `R_j` gate help vs dose-response alone?** | untestable (bug) | **`MVDF_v2` is slightly more invariant than `intervention_only` (−0.36 mean `I`, 6/9), but `intervention_only` itself is the *worst* condition** — so the `R_j` gate recovers some of what raw-`U_j` selection loses, without beating correlation. |
| **`correlated` most invariant** | 5/8 | **4/9 dataset-seed runs** (still the modal winner; `MVDF_v2` 3/9, `random` 2/9) |
| **H6** — does MVDF's feature subspace give narrower intervals? | untested | **Mixed but leans YES under shift**: `g_mvdf` shift-width is significantly *lower* than `g_dense` on **9/9 runs** (mean −0.18, driven hard by ETTh2). In-distribution it's a wash (7/9 "significant" but sign flips, mean ≈ 0). |

**The honest new synthesis:** H1c is rock-solid. H3 (invariance) is a **clean negative** —
`MVDF_v2` is not more invariant, and the one single-seed exception was noise. But H6 is
**not** the null we expected: the MVDF feature subspace *does* produce narrower conformal
intervals than FFDCI's raw dense vector **under distribution shift** (9/9 runs, sometimes
by a lot), even though its difficulty signal is no more invariant. That is a genuinely
interesting dissociation and it changes the paper.

---

## Phase 1 — H1c across seeds (CONFIRMED, tight)

Type-B feature fraction and `MVDF_v2` type-B enrichment, per (dataset, seed):

| dataset | seed | type-B frac | **enrich MVDF_v2** | enrich MVDF_v1 | enrich correlated | H1 ρ | H1b ρ |
|---|---|---|---|---|---|---|---|
| ETTh1 | 0 | 0.20 | **2.50** | 1.67 | 0.83 | 0.006 | 0.159 |
| ETTh1 | 1 | 0.21 | **2.00** | 1.20 | 1.20 | −0.011 | 0.066 |
| ETTh1 | 2 | 0.23 | **2.96** | 1.85 | 0.00 | −0.034 | 0.096 |
| ETTh2 | 0 | 0.21 | **2.80** | 1.20 | 0.40 | 0.157 | 0.055 |
| ETTh2 | 1 | 0.20 | **2.50** | 0.42 | 0.83 | 0.048 | 0.133 |
| ETTh2 | 2 | 0.18 | **2.27** | 0.91 | 0.91 | 0.246 | 0.177 |
| electricity | 0 | 0.15 | **3.33** | 0.00 | 0.56 | −0.002 | 0.042 |
| electricity | 1 | 0.16 | **3.68** | 0.53 | 1.05 | 0.044 | 0.021 |
| electricity | 2 | 0.18 | **2.27** | 0.45 | 0.45 | −0.074 | 0.161 |

- **`MVDF_v2` type-B enrichment: 2.0–3.7×, every single run.** sd within each dataset is
  ~0.3–0.7 — the effect is well above noise.
- **`MVDF_v1` (single-point ablation): 0.0–1.85×, mostly ≤ 1.2** — it does *not* reliably
  select type-B features. On electricity it's ≤ 0.53 every seed.
- **`correlated`: 0.0–1.2×** — also does not.
- H1 ρ (Δ_pred vs correlation) is in [−0.07, +0.25], H1b ρ (Δ_pred vs C_j) in [0.02, 0.18].
  The interventional and correlational/observational rankings are essentially unrelated,
  every seed — same as before.
- SAE smoke PASS on all 9; `D_unc` monotone fraction ~0.88.

**H1c is confirmed with tight error bars.** The dose-response + `R_j` construction
reliably isolates a feature type — low prediction-effect, high uncertainty-effect — that
neither correlation screening nor single-point ablation finds. This is the paper's
qualitative anchor and it is solid.

---

## Phase 2 — Invariance across seeds (H3 is a NEGATIVE; the single-seed win was noise)

`I(R)` (lower = more invariant), mean ± sd over 3 seeds, with per-seed values:

| dataset | I_correlated | I_MVDF_v2 | I_intervention_only | I_MVDF_v1 |
|---|---|---|---|---|
| ETTh1 | 1.078 ± 0.308 | **0.925 ± 0.863** | 1.565 ± 0.145 | 1.383 ± 0.386 |
| | per-seed MVDF_v2: `[0.00, 1.065, 1.709]`  ‖  corr: `[1.364, 0.752, 1.119]` | | | |
| ETTh2 | 0.898 ± 0.536 | 1.047 ± 0.375 | 1.417 ± 0.173 | 1.423 ± 0.183 |
| | per-seed MVDF_v2: `[0.693, 1.008, 1.44]`  ‖  corr: `[1.514, 0.537, 0.643]` | | | |
| electricity | 1.378 ± 0.262 | 1.412 ± 0.247 | 1.490 ± 0.085 | 1.507 ± 0.110 |
| | per-seed MVDF_v2: `[1.664, 1.401, 1.17]`  ‖  corr: `[1.447, 1.089, 1.598]` | | | |

**Tallies over 9 dataset-seed runs:**

- **`I(MVDF_v2) < I(correlated)`: 3/9** — exactly 1/3 seeds on each dataset. This is
  coin-flip behaviour, not a real advantage. The single-seed run's "ETTh1 wins" was
  **seed-0-specific noise** — on ETTh1 seed 0, `I(MVDF_v2)` came out at essentially 0
  (a 5-bin curve that happened to be near-identical across environments), but seeds 1–2
  give 1.07 and 1.71. sd = 0.86. As the review anticipated, the 0.053 margin did not
  survive.
- **`correlated` is the most-invariant condition on 4/9**; `MVDF_v2` on 3/9; `random`
  on 2/9; `intervention_only` and `MVDF_v1` on **0/9**.
- **`intervention_only` (dose-response, no `R_j`) is consistently the *worst* or
  near-worst** — mean `I` 1.42–1.57, higher (worse) than `correlated` on 8/9 runs.

**Does the `R_j` specificity gate add anything to invariance?** (the ablation
`intervention_only` finally makes possible) — **Yes, but only relative to raw dose-response,
not relative to correlation.** `I(MVDF_v2) − I(intervention_only)` = **−0.36 mean, better
on 6/9 runs.** So residualizing `U_j` against `P_j` (the `R_j` step) *recovers* some
invariance that raw `|U_j|`-ranking throws away — but the recovered set is still no more
invariant than a plain correlation-selected one.

**Reading (spec §7, row 3, now with error bars):** the dose-response/`R_j` criterion
selects a genuinely different and genuinely more uncertainty-specific feature set (H1c),
and the `R_j` gate is doing real work relative to raw `U_j` selection — but the end result
is **not more invariant across deployment environments than correlation screening.**
`MVDF_v2` ≈ `correlated` on invariance (means 0.93 vs 1.08, 1.05 vs 0.90, 1.41 vs 1.38 —
overlapping error bars every dataset). **H3 fails.**

---

## Phase 3 — H6: scorer interval width (the surprise)

α = 0.10. Calibration set = first 70% of `E_eval_ID`; ID width = last 30% of `E_eval_ID`;
shift width = `E_eval_shift`. Three input modes (same MLP arch, same budget). Mean ± sd
over 3 seeds:

| dataset | mode | ID coverage | ID width | shift coverage | **shift width** |
|---|---|---|---|---|---|
| ETTh1 | dense | 0.849 | 2.229 | 0.892 | 1.977 |
| ETTh1 | **mvdf** | 0.850 | 2.221 | 0.892 | **1.927** |
| ETTh1 | dense+mvdf | 0.850 | 2.247 | 0.892 | 1.969 |
| ETTh2 | dense | 0.854 | 3.519 | 0.898 | 3.354 |
| ETTh2 | **mvdf** | 0.862 | 3.573 | 0.899 | **2.915** |
| ETTh2 | dense+mvdf | 0.855 | 3.504 | 0.898 | 3.348 |
| electricity | dense | 0.902 | 2.005 | 0.892 | 3.226 |
| electricity | **mvdf** | 0.905 | 1.937 | 0.894 | **3.165** |
| electricity | dense+mvdf | 0.902 | 2.013 | 0.892 | 3.241 |

**Paired bootstrap, `g_mvdf` − `g_dense` width** (negative = mvdf narrower):

| env | mean diff | P(mvdf < dense) | # significant / 9 |
|---|---|---|---|
| **shift** | **−0.184** | **1.00** | **9 / 9** |
| ID | −0.007 | 0.54 | 7 / 9 (but sign flips seed to seed; mean ≈ 0) |

- **Under shift, `g_mvdf` produces significantly narrower intervals than `g_dense` on
  every single one of the 9 runs**, at matched (~89%) coverage. On ETTh2 the gap is large
  (2.915 vs 3.354, ~13% narrower). On ETTh1 and electricity it is smaller (~2–3%) but
  consistent and significant.
- **In-distribution it is a wash** — 7/9 "significant" but the sign flips across seeds
  (ETTh1: −0.099, −0.003, +0.079), mean ≈ 0.
- `g_dense+mvdf` (augmentation) is **not** better than `g_dense` — mixed signs, no
  consistent gain. So the value, where it exists, is in *replacing* the dense vector with
  the MVDF subspace, not augmenting.

**Coverage caveat:** the scorers land at **~85–87% ID coverage, not 90%** (electricity is
fine at 90%; ETTh1/h2 undercover). Same per-bin/global-ACI burn-in issue as Run 5 on short
splits — `E_eval_ID`'s calibration portion is only a few hundred windows. The width
comparison is still valid (all three modes undercover by the same ~3–5 pt, so it's
matched), but "g_mvdf gives *valid* 90% intervals that are also narrower" is **not**
established here — only "narrower at matched (sub-nominal) coverage." A clean claim needs
the calibration fixed (more calibration data, or a longer ACI burn-in).

---

## Updated assessment — what the paper is now

### Three findings, in order of strength

1. **H1c (positive, tight).** Dose-response + `R_j` reliably isolates uncertainty-specific
   ("type-B") features — 2.0–3.7× enrichment, 9/9 runs — that correlation screening and
   single-point ablation do not. Reusable method + a real mechanistic-interpretability
   result.

2. **H3 (negative, now with error bars).** Those features are **not more invariant** across
   deployment environments than correlation-selected ones. The single-seed "exception"
   (ETTh1) was noise (sd 0.86). The `R_j` gate does recover invariance that raw-`U_j`
   selection loses (−0.36 vs `intervention_only`, 6/9) — but not enough to beat
   correlation. Uncertainty-specificity ≠ invariance; the data separates them.

3. **H6 (unexpected partial positive).** Despite H3, the MVDF feature subspace produces
   **significantly narrower conformal intervals under distribution shift** than FFDCI's
   raw dense feature vector — 9/9 runs, mean 18% of the half-width on ETTh2, ~2–3%
   elsewhere. In-distribution: no difference. Augmentation (`dense+mvdf`): no gain.
   **Caveat: at sub-nominal (~86%) coverage, so it's "narrower at matched coverage", not
   "narrower and valid" — the latter needs the calibration burn-in fixed.**

### The dissociation is the story

The interesting result is #2 and #3 *together*: a feature set can give you **sharper
intervals under shift** (H6) **without** having a **more invariant difficulty signal**
(H3). That is not what the v1 hypothesis predicted (it assumed the two go together), and
it is worth a paper. Possible mechanism: the MVDF subspace is lower-dimensional and less
prone to the dense-vector scorer's overfitting on the small calibration set, so the
pinball MLP generalises better to the shifted deployment stream — a *regularization* story,
not an *invariance* story. Worth checking directly (compare g_mvdf to g_dense with a
random-k-dim subspace of the dense vector — if random-k also beats dense, it's
dimensionality; if only MVDF beats dense, it's the feature content).

### Recommended next step

**One more targeted run, then write up.** The H6 result is interesting enough that it
should be nailed down before it goes in a paper:

1. **Fix the calibration burn-in** so all scorers hit ~90% ID coverage (more calibration
   windows: use all of `E_eval_ID` for calibration and measure ID width on a fresh slice,
   or lengthen the ACI warm-up). Re-run H6. If `g_mvdf` still beats `g_dense` on shift
   width at *valid* coverage, that's a clean, publishable positive.
2. **Add the random-k-subspace control** to H6 (one extra `mode`). Distinguishes
   "MVDF content helps" from "any low-dim projection helps". Cheap — same run.
3. Keep it to ETTh1 + ETTh2 + electricity + maybe traffic, 3 seeds. ~1 Kaggle run.

Then write up: H1c as the anchor, H3 as the clean negative on invariance (with the
`R_j`-vs-`intervention_only` ablation), H6 as the dissociation — mechanistically-validated
features buy interval sharpness under shift without buying difficulty-signal invariance,
and here's the likely reason. Combined with Phase 0's layer-sweep null and Run 5's
calibration null: a thorough, multi-result paper that is *not* a flat negative — it has a
positive (H1c), a clean negative (H3), and a surprising positive (H6) that reframes what
mechanistic validation is actually good for.

---

## ADDENDUM 2 (2026-09-03) — H3 is not cleanly measurable; the reason is structural

A review of this report prompted a bin-count audit across **all 45 condition-runs**
(9 dataset-seed pairs × 5 conditions), not just the flagged ETTh1 case. Result: the
degeneracy is systematic, not a one-off.

- **7 runs DEGENERATE** (a curve pair shares < 3 populated bins → `I` meaningless):
  `random` on ETTh1 (all 3 seeds), ETTh2 s0, electricity s0 & s2; and `MVDF_v2` on
  ETTh1 s0.
- **~26 runs BORDERLINE** (exactly 3 populated bins — a 3-point scale-normalized
  curve is essentially "is the middle bin above or below the mean", very
  low-information): **`MVDF_v2` on 8/9**, **`correlated` on ~6/9**.
- Only ~12 runs are clean (≥4 bins every environment) — almost all `intervention_only`
  or `MVDF_v1`.

**The `MIN_SHARED_BINS=3` guard I added is not sufficient** — it keeps the 3-bin
curves, and those carry almost no signal. So the "3/9 coin flip" H3 tally in the body
above is computed largely on 3-bin noise for `MVDF_v2` and `correlated` alike.

### The aggregation diagnostic — root cause confirmed

Rather than tweak the metric (rank-correlation fails *silently* on tied data; forced
rank-binning manufactures structure), we tested whether the **aggregation** is the
problem. `r_t = (1/k) Σ_j z_tj` — a plain mean — was flagged as weak before Run 5.
`experiments/aggregation_diagnostic.py` reloads each deterministic forecaster + SAE,
regenerates the SAE codes on the E_eval blocks, and reports signal dispersion +
bins-populated under **four** aggregations (mean / max / L2 / active-count).

**Result (ETTh1 + ETTh2, seeds 0–1): no aggregation rescues it.**

| condition | median per-feature activation rate | bins populated (mean / max / L2) |
|---|---|---|
| `intervention_only` | **0.056** | 4.0 / 4.0 / 4.0 |
| `MVDF_v1` | **0.051** | 4.25 / 4.25 / 4.25 |
| `correlated` | **0.023** | 3.0 / 3.0 / 3.0 |
| `random` | **0.024** | 3.0 / 3.0 / 3.0 |
| **`MVDF_v2`** | **0.020** | **2.5 / 2.5 / 2.5** |

`MVDF_v2` and `correlated` select SAE features that fire on **~2% of windows** (vs
~5–6% for `intervention_only` / `MVDF_v1`). *Any* scalar summary of a 12-feature
bundle that is all-zero on ~90%+ of windows is near-constant — max is 0 on the
all-zero windows, active-count is 0-or-1 almost everywhere, L2 tracks the max.

**This is structural, not a bug.** The H1c selection criterion (difficulty-*specific*
→ narrowly-firing *by design*) is at odds with the H3 measurement (needs an aggregate
signal that varies across windows). They pull against each other.

### Updated H3 framing for the paper

**Not** "we measured invariance and MVDF_v2 didn't win" — it's **"the property that
makes these features mechanistically interesting (specificity → sparsity) is the
property that makes conditional-difficulty-drift unmeasurable for them, verified across
four aggregation choices."** That is a real, informative statement about what kind of
signal these features carry, and it pairs with H6's SAE-basis-vs-PCA finding: across
both, *which* features you pick from the SAE (by causal validation, correlation, or
random) matters less than the SAE representation's *structure*.

## Bugs fixed in this run (reproducibility appendix, cont. from V2_PHASES012_REPORT §Addendum)

| # | Bug | How it showed | Fix |
|---|---|---|---|
| C | Non-deterministic per-dataset seeding | `ETTh1`-alone ≠ `ETTh1`-as-Nth-in-loop; caused the `I(correlated)` 1.39→~0 discrepancy (NOT a data-source diff) | `experiments/_seed.py::derive_seed`, threaded into every seeded component; verified byte-identical |
| D | `intervention_only` collapsed to `MVDF_v1` | the `R_j`-gate ablation was impossible | derive from Phase 1 `candidates.csv` as top-k by `|U_j|`, no `R_j` |
| E | Fingerprint false-positive on electricity | guard raised "47 channels, expected 48" | `data/fingerprint.py`: `CHANNEL_EXACT` set (ETT + weather exact; capped datasets ±1) — electricity keeps 47 (target already in top-47 by variance) |

Data fingerprints recorded this run: ETTh1 `f56e3fc3a29f4258` (17420 rows, 7 ch),
ETTh2 `11324ab9896c76d3` (17420 rows, 7 ch), electricity (35000 rows, 47 ch).
