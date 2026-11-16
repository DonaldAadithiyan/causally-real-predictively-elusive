# Camera-ready summary — ICAART 2027 paper #126

Spec: `CAMERA_READY_EXPERIMENTS.md` (pasted task). All outputs under `runs/camera_ready/`. Every PREREG.md was committed before its experiment's first run (git log: `aaec82f` E2, `cabf8c9` E3+E1, `62f7e35` E4+E5, `763d853` E6). The repo was not under version control; a local git repo was initialised (`fb7e43f` = untouched baseline snapshot; weights, arrays and raw data are git-ignored).

**Compute:** Apple Silicon, no CUDA GPU. ETTh trained on CPU (4 threads: the setting under which the paper's seeds reproduce bit-exactly), electricity/traffic on MPS (as in round 2). 30 iTransformer pipelines in total (ETTh1/ETTh2 × 10 seeds, electricity/traffic × 5), plus 6 regenerated TimeMixer pipelines.

**Reproduction:**
- Every pre-existing artifact used was regenerated and checked before use.
- ETTh1 s0 retrained through the new driver: bit-identical.
- All 6 TimeMixer runs: identical val MSE, FVU and MVDF.
- Traffic s0–2 arrays rebuilt from saved weights: FVU identical.
- All 30 dose-response cells: identical pool and MVDF set.
- Phase 3/4 from arrays: stored numbers matched to ≤ 3e-3.

---

## 1. Criteria

| ID | verdict | key numbers |
|---|---|---|
| E0.1 | done | 5 hard-coded "0.5 SD" strings fixed; report diff = threshold text only. Headline threshold 0.25 SD: MVDF match 0.08–0.33 (mean 0.25) vs pool base 0.28–0.43. **New bug found:** "shuffle floor" ≡ match rate by construction. |
| E0.2 | done | `E0/DEFINITIONS.md`: 6 code-vs-text discrepancies (R² uses eval mean; profile is block-local; interventions drop SAE error; coverage = all 96 steps, one channel; naive_matched uses immediate-feedback ACI; Phase 5 threshold). |
| **C3.1** ex-ante predictable | **FAIL** (ETTh1, ETTh2, electricity, traffic) | best set per dataset: ETTh1 F6 1/10, ETTh2 F2 6/10, electricity F3/F6 1/5, traffic F1 1/5 (need 8/10, 4/5) |
| **C3.2** dissociation | **FAIL** (requires C3.1) | F0 (MVDF) not predictive on 30/30 seeds |
| **C3.3** MVDF adds nothing | **PASS** (all 4) | ΔR² CI ∋ 0 on 10/10, 10/10, 5/5, 4/5 |
| **C3.4** recoverable opportunity | **FAIL** (all 4) | the CV-selected best ex-ante set never reaches GRR ≥ 0.20 at valid coverage; its valid GRRs are all negative. Over *all* sets on the shift block: ETTh valid GRRs −0.20 to −2.72; traffic F3 (not CV-selected) +0.28 to +0.41 on 4/5 seeds against an outlier-inflated constant band (coverage 1.000); see `F/RESULT.md` §5 |
| **C3.5** representation contains it | **PASS ETTh2**, FAIL ETTh1/elec/traffic | ETTh2: dense hidden state predictive 10/10 (ρ ≈ 0.38), SAE code 7/10; MVDF 0/10 |
| C3 block-192 sensitivity | one change | electricity C3.1/C3.2 flip to PASS at block 192 (F3 4/5); block 96 is primary |
| **C4.1** selection robust | **PASS** (at the bar) | median Jaccard: binned 0.50, spline 0.71, rank 0.60 |
| **C4.2** enrichment robust | **FAIL** | CI > 1 on: isotonic 5/20, binned 0/20, spline 7/20, rank 17/20 |
| **C4.3** null robust | **PASS** | R² < 0 on 10/10 seeds, both batteries, both ETTh datasets, all 6 definitions |
| **C5a** dose-response clean | **PASS** | MVDF median linear-fit R² 0.999; monotone (guaranteed by gate 1) |
| **C5b** natural range | **FAIL** (boundary) | Jaccard 0.85 ✓; ETTh2 s5 enrichment 2.50 with CI lower bound = 1.000 exactly |
| **C5c** interchange | **FAIL** (ETTh1 7/10; ETTh2 9/10) | sign agreement 81% of MVDF features on ETTh |
| **C1.1** L1 null generalizes | **PASS** (electricity, traffic; also TimeMixer) | best R² −0.29 to −1.33, every CI < 0 (10/10); TimeMixer −0.49 to −1.16 (6/6) |
| **C1.2** heterogeneity generalizes | **PASS** | oracle gap ID 18.1% (elec), 39.1% (traffic); every CI > 0 |
| **C1.3** = C3.x on new data | see C3 rows | |
| **C2.1** "3/3" claims hold on ≥ 80% of seeds | **FAIL (partly)** | enrichment ≥ 1×, Phase-3 null, oracle gap: hold on 30/30. "naive_matched significantly narrower than scorer on every run": ETTh1 4/10 (dense), 0/10 (mvdf); electricity 0/5, 0/5 |
| **C2.2** medians stable | **PASS** | 32/32 rows: |Δmedian| ≤ pooled half-width |
| F1 regime gating (H7-A), 10 seeds | **0/20 pass** | R_Q ETTh1 0.86–1.20, ETTh2 0.97–1.05; no CI excludes 1 from above; the original 1/6 pass does not reproduce (it came from a mismatched-SAE feature list) |
| F2 pooled enrichment | holds pooled, **not on ETTh1** | 30-run mean 2.52 [2.13, 3.33]. Against the selection null (≈ 1.8× from shuffled U): Fisher p = 5.6e-10 overall; ETTh1 0/10 seeds, Fisher p = 0.62 |
| Health rule | 0/30 unhealthy | new-seed FVU 0.0052–0.0160; val-MSE ratio ≤ 1.006 |
| **E6** C1.1 (TimeMixer) | **PASS** | 6/6, every CI < 0 |
| **E6** C3.1–C3.2 (TimeMixer) | **FAIL** | 0/3 every set; ETTh2 F2 ρ ≈ 0.32 but CI ∋ 0 |

---

## 2. For the paper (priority order)

**E0: §5.11 reproducibility (replace the sentence).**
> "At a 0.25-SD signature threshold, 8–33% of MVDF features (mean 25%) have a mutual-nearest-neighbour counterpart in an independently seeded SAE, versus 28–43% for arbitrary pool features."

Delete "never above a shuffled-signature floor"; that floor is identical to the match rate by construction.

**E3: ex-ante baselines (answers R1 "recoverable opportunity", "restricted estimators").**

| CV-selected estimator, E_eval_ID | ETTh1 ρ / predictive seeds | ETTh2 ρ / predictive seeds |
|---|---|---|
| MVDF activations (F0) | −0.01 / 0/10 | +0.00 / 0/10 |
| all ex-ante (F6) / input stats (F2) | −0.02 / 1/10 | +0.32 / 6/10 |
| dense hidden state (F4) | −0.01 / 0/10 | +0.38 / 10/10 |
| full SAE code (F5) | −0.02 / 0/10 | +0.26 / 7/10 |
| GRR of best ex-ante band | none ≥ 0 | none ≥ 0 |

> "Information available at forecast time does not predict realised difficulty on ETTh1 (0–1 of 10 seeds for every feature set; six estimators including ridge and random forests; model chosen by time-series CV). On ETTh2 a weak rank signal exists (ρ ≈ 0.38) and is carried by the dense hidden state on 10/10 seeds, but not by the causally validated MVDF features (0/10), which add nothing on top of ex-ante features. No ex-ante Mondrian band narrowed intervals at valid coverage, so the oracle gap is an upper bound on heterogeneity, none of which was recovered by any ex-ante predictor we tested."

**E4: R_j robustness (answers R1 "heuristic").**

| R_j definition | Jaccard vs isotonic | enrichment CI > 1 | R² null |
|---|---|---|---|
| isotonic (paper) | 1.00 | 5/20 | 20/20 |
| binned mean | 0.50 | 0/20 | 20/20 |
| smoothing spline | 0.71 | 7/20 | 20/20 |
| rank residual | 0.60 | 17/20 | 20/20 |

> "Alternative residualisations change 30–50% of the selected features and never restore natural predictive power (R² < 0 on all 20 ETTh seeds under every definition); type-B enrichment is reproduced by spline and rank residualisation but not by a 10-bin mean."

**E5: interventions (answers R1 "artificial vs natural").**
> "Dose-response curves of the selected features are near-linear (median R² 0.999) and monotone; restricting the intervention to scaling down (λ ≤ 1) reselects largely the same features (median Jaccard 0.85). Setting a feature to values it naturally takes on other inputs moves the difficulty score in the direction predicted by its dose-response slope for 81% of selected features (≥ 70% agreement on 16 of 20 ETTh seeds)."

**E1: four datasets (answers R1/R2 "two datasets").**

| dataset | seeds | MVDF best R² | oracle gap ID | ex-ante best ρ / pred. seeds | GRR (valid) |
|---|---|---|---|---|---|
| ETTh1 | 10 | −0.60 [−0.72, −0.52] | 25.2% [23.6, 25.9] | −0.02 / 1/10 | none |
| ETTh2 | 10 | −0.89 [−0.94, −0.77] | 30.0% [28.2, 30.6] | +0.38 (hidden) / 10/10 | none |
| electricity | 5 | −0.29 [−0.33, −0.29] | 18.1% [17.3, 18.3] | +0.33 / 1/5 | −0.41 (1/5 valid) |
| traffic | 5 | −0.70 [−1.33, −0.46] | 39.1% [39.0, 39.6] | +0.34 / 0/5 | none (coverage 1.000) |

> "Across all four datasets (30 trained runs), natural MVDF activation fails to predict difficulty (best held-out R² −0.29 to −1.33, every block-bootstrap CI below zero), while an oracle shows 18–39% of interval width is attributable to difficulty heterogeneity. No ex-ante predictor chosen by cross-validation converts this into a narrower conformal band at valid coverage."

**E2: seeds (answers R3).** Report per-seed values, median [min, max] and k/N (no SD). Table in `E2/RESULT.md` "For the paper"; all per-seed numbers in `E2/SEED_TABLE.md`.
> "Across 30 independently retrained pipelines, MVDF is type-B-enriched on every run (median 2.0–3.8×), natural MVDF activation predicts difficulty worse than a constant on every run, and the oracle gap is positive on every run; no headline median moved by more than its pooled CI half-width when going from 3 seeds to 5 or 10. The constant-width band's advantage over the learned scorer is significant on only 4/10 (dense) and 0/10 (MVDF) ETTh1 seeds."

**E6: TimeMixer (answers R2 architecture).**
> "On TimeMixer, natural MVDF activation again fails to predict difficulty (best held-out R² −0.49 to −1.16 across six runs, every CI below zero), and the oracle gap (24–30%) matches iTransformer."

---

## 3. Numbers in the paper that change

| location (submitted PDF, v6) | old | new | source |
|---|---|---|---|
| Abstract; §1; §5.5; §7 | "24.5 to 29.2 percent of interval width" (ETTh, mean of 3) | ETTh: 23.5–30.6% over 10 seeds (ID 23.6–30.6, shift 23.5–27.2); all four datasets: 17.3–39.6% | `E2/AGG_E2_phase4.csv`, `E1/AGG_phase4.csv` |
| Table 6 | ETTh1 24.5 / 25.4; ETTh2 29.2 / 24.5 (mean, 3 seeds) | ETTh1 25.2 / 25.6; ETTh2 30.0 / 24.0 (median, 10 seeds); add electricity 18.1 / 37.2, traffic 39.1 / 37.3 | E2, E1 |
| Table 4 | ETTh1 marg −0.72 to −0.58, active −0.70 to −0.48; ETTh2 marg −0.92 to −0.89, active −0.89 to −0.80 | ETTh1 −0.72 to −0.52, active −0.96 to −0.47; ETTh2 −0.94 to −0.77, active −0.96 to −0.69 (10 seeds); add electricity −0.33 to −0.29, traffic −1.33 to −0.46 | `E2/AGG_E2_phase3.csv` |
| Table 4 caption | "Negative = worse than predicting the training mean" | R² as implemented uses the **evaluation-set** mean; fix the caption, or report the fit-mean R² from E3 | `E0/DEFINITIONS.md` §6.1 |
| Table 9, iTransformer column | enrichment 2.0–4.6×; Δw −0.14 to −0.43; R² −0.58 to −0.92; "6/6" | enrichment 1.25–3.68× (ETTh, 20 seeds, local data); Δw (naive − dense, shift) −0.11 to −0.18 (ETTh1), −0.33 to −0.45 (ETTh2); R² −0.52 to −0.94; 20/20 | E2 |
| Table 9, TimeMixer column | R² −0.49 to −1.16 | **unchanged** −0.49 to −1.16 (reproduced exactly; −1.1648 rounds to −1.16. An earlier version of this list wrongly said −1.17 after double rounding) | `E6/RESULT.md` |
| Table 2 | MVDF ≥ 1.0× 12/12, min 2.00, max 4.62, mean 3.02 (Kaggle, 4 datasets × 3) | local models, 30 runs: ≥ 1.0× on 30/30, min 1.25, max 4.17; bootstrap CI excludes 1 on 9/30 | `E2/AGG_E2_enrichment.csv` |
| §5.6 | "both differences are significant in every dataset-seed run, 12 of 12 each" | not supported: naive_matched significantly narrower than dense on 4/10 ETTh1, 10/10 ETTh2, 0/5 electricity, 5/5 traffic seeds; than mvdf on 0/10, 10/10, 0/5, 5/5 (block bootstrap, local data). Direction holds everywhere except electricity mvdf (narrower at 0.82 coverage) | `E2/AGG_E2_scorer_vs_naive.csv` |
| §5.11 | match rate 0.21–0.29 vs base 0.30–0.39; "never above a shuffled-signature floor" | 0.08–0.33 (mean 0.25) vs 0.28–0.43 per comparison at 0.25 SD; delete the floor clause | `E0/RESULT.md` |
| §5.7 regime gating | "Only one of six seed-runs passed: R_Q ranged from 0.91 to 1.06, five of six intervals contained 1" | none of 20 seed-runs passes; R_Q 0.86–1.20 (ETTh1), 0.97–1.05 (ETTh2); the original pass came from cell-B IDs taken from a different SAE | `F/AGG_F1_regime_gating.csv` |
| §5.1 enrichment wording | "MVDF reliably finds type-B features" (vs pool base rate) | the selection rule alone gives ≈ 1.8× on shuffled U; beyond that null, enrichment holds pooled (p = 6e-10) and on ETTh2/electricity/traffic but not ETTh1 (p = 0.62) | `F/AGG_F2_pooled.csv` |
| §3 / §5.4 seed statements | "3 seeds" / "mean ± sd over 3 seeds" | 10 seeds (ETTh), 5 (electricity, traffic); per-seed + median [min, max] + k/N | E2 |
| G2: S_t normalisation for Table 6 / Table 4 | S_t normalised by each evaluation block's own error profile | with the pre-evaluation (E_select) profile: rank agreement between the two S_t 0.95–0.997; oracle gap moves ≤ 0.03 (e.g. ETTh1 ID 25.2% → 24.7% median); Phase-3 MVDF R² more negative on ETTh (ETTh1 s0 −0.64 → −0.83), essentially unchanged on electricity/traffic | `G/AGG_G2_select_profile.csv` |
| G4: direct L2 evidence (proposed replacement for enrichment) | not in paper | share of features whose U_j CI excludes 0, out-of-sample on E_eval_ID, all pool / MVDF / runs MVDF > pool: ETTh1 0.19 [0.13, 0.23] / 0.17 [0.08, 0.50] / 6/10; ETTh2 0.22 [0.15, 0.28] / 0.25 [0.08, 0.50] / 6/10; electricity 0.47 [0.34, 0.53] / 0.58 [0.17, 0.67] / 3/5; traffic 0.17 [0.15, 0.22] / 0.08 [0.00, 0.17] / 0/5; overall 15/30 (in-sample on the selection windows 30/30, selection-inflated). Expected under no effect ≈ 5% (nominal) | `G/AGG_G4_L2_evidence.csv` |
| G5: §5.6 static-calibration sentence | not in paper | round 2, static calibration, shift block, seeds 0–2: every learned scorer's width reduction vs naive_matched on ETTh is −0.088 to +0.004, none narrower on all 3 seeds; electricity pca_k narrower on 3/3 by 0.5–1.0%; traffic erratic (calibration outlier; naive_no_scorer 37–44% narrower than naive_matched) | `G/AGG_G5_static_shift.csv` |

---

## 4. Anomalies, failures, and what could not be done

1. **Phase 5 shuffle floor is invalid** (E0): permuting rows does not change mutual-NN matching. Code not fixed (a valid null needs a new design).
2. **GRR on the ID test block is undefined** on every ETTh cell. Static naive_matched covers only 0.56–0.62 on the last 20% of E_eval_ID, so the LOO oracle (≈ 0.90) is wider than naive. This is the chronological calibration-transfer problem; the ID block (394 windows ≈ 4 blocks) cannot support a gap-recovery claim.
3. **Traffic is saturated** (coverage 1.000 on ID **and** shift for the static constant band) because of an outlier stretch in the calibration/fit block (windows 1604–1699, seed 0). This also produces R² of −13 to −29 for ex-ante sets despite ρ ≈ 0.34. Traffic gap recovery is uninterpretable on both blocks; on 3/5 seeds the Phase-3 point R² sits just outside its own block CI.
4. **Block bootstrap is coarse:** 787 eval windows ≈ 8 blocks of 96. Several consistent signals (ETTh2 F2 ρ ≈ 0.32 on 10/10 seeds; electricity F2/F3 ρ ≈ 0.24–0.29 on 5/5) miss C3.1 because their CI includes 0. With block 192 the electricity verdict flips to PASS. Both are reported; block 96 is primary as pre-registered.
5. **Order-dependence bug in the added enrichment CI**, found and fixed during the run (bootstrap draws depended on feature order). Fixing it changed one C5b cell, and C5b went from PASS to FAIL. The post-fix verdict is reported.
6. **Deviations from the spec text, each documented in its PREREG/RESULT:**
   - E5c follows the existing code, which *drops* the SAE reconstruction error; the spec said it is "kept fixed".
   - Gap recovery uses static split conformal for every method; original naive_matched has immediate-feedback ACI, which is unavailable at H = 96.
   - Electricity/traffic use `electricity.csv` / `traffic.csv`, the preprocessed bundle files the paper's models were trained on, byte-identical locally; not `LD2011_2014.csv` / `PeMS.csv`.
   - Enrichment CIs and hypergeometric p-values were added because `AGG_h1c.csv` has none.
   - Spline residualisation is fitted on z-scored P, U for numerical conditioning.
7. **Not reproduced:** the paper's §5.6 / Table 7 scorer numbers come from Kaggle runs (Kaggle data copy). E2 re-runs that protocol on local data; it is a re-run, not a reproduction.
8. **Not in scope / not done:**
   - Table 3 (invariance I(R), "mean ± sd over 3 seeds") was not in the E2 metric list and was not recomputed. It still reports SD over 3 seeds, which Reviewer 3 objected to.
   - Regime gating (§5.7) was not re-run.
   - Electricity/traffic were run to 5 seeds, not 10.
9. **The predicted "small" fraction is not small:** 8.1% of active windows at λ = 1.25 exceed the natural q99 (57% counting inactive windows as zero). E5b answers this directly.

## Follow-up (F)
Items 1–5 of the follow-up request are answered in `F/RESULT.md`:
- **1. Regime gating:** 0/20 seed-runs pass.
- **2. Pooled enrichment:** holds against the selection null overall, but not on ETTh1.
- **3. ACI:** learned scorers and naive_matched use the same immediate-feedback ACI, so the comparison is like-for-like.
- **4. Profile locality:** each block is normalised by its own realised errors.
- **5. Shift gap recovery:** positive only on traffic, against an outlier-inflated constant band.

## File map
`E0/` RESULT, DEFINITIONS, Phase-5 diff · `E1/` RESULT, AGG_E1_*, per-dataset folders, PER_WINDOW_E1.parquet · `E2/` RESULT, SEED_TABLE.md, AGG_E2_*, meta/ · `E3/` RESULT, AGG_E3_predictability/gap_recovery/delta, CRITERIA_E3.json, PER_WINDOW_E3.parquet, cells/ · `E4/` RESULT, AGG_E4_rj_robustness.csv · `E5/` RESULT, AGG_E5a/b/c, E5a_hist.png · `E6/` RESULT, AGG_E6_*, arrays/ · code in `camera_ready/` (leakage test: `python -m camera_ready.test_leakage`).
