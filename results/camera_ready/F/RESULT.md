# F — Answers to follow-up items 1–5

## Provenance
- Git: pre-registration `f5dbe9a` (committed before any F computation); results in the following commit.
- Command: `python -m camera_ready.f_followup`. Items 3 and 4 are code reading only.
- Seeds: ETTh1/ETTh2 0–9; electricity/traffic 0–4. Data: as in E1/E2 (HTTP ETT mirror `c6d1709fd0693b1a` / `c022db5521d23684`; Kaggle-bundle electricity/traffic, local copies byte-identical).
- Hardware: Apple Silicon CPU, no GPU. Wall-clock ≈ 4 min (no training; all from saved arrays and E3/E5 outputs).

---

## 1. Regime gating (H7-A R_Q) on 10 seeds

**What was run, and why it is not literally `h7a_gate.py`.** The stored H7-A run retrains a local SAE (20 000 steps) but reads the cell-B feature IDs from the Kaggle runs' type tables (`v2_confirm/v2_phase1_seed{s}`). Those come from a different SAE, so the feature indices do not refer to the same features. The tables also exist only for seeds 0–2. I used round 2's corrected construction (`p126/r2_wp1.py` L142–163) on the camera-ready pipelines:
- cell-B = the seed's own pool type-B features;
- τ = E_select 90th percentile of A_t (the fraction of cell-B features active);
- D_t = 1[A_t ≥ τ];
- R_Q = Q.90(S | D=1) / Q.90(S | D=0).

Seeds 0–2 reproduce round 2 exactly (|ΔR_Q| ≤ 2e-16, same cell-B sizes; `REPRO_F1_vs_round2.csv`).

| | ETTh1 (10 seeds) | ETTh2 (10 seeds) |
|---|---|---|
| R_Q, E_eval_ID, median [min, max] | 0.95 [0.86, 1.20] | 1.01 [0.97, 1.05] |
| CI excludes 1 from above (pass rule), ID | **0/10** | **0/10** |
| same at block 192 | 0/10 | 0/10 |
| CI excludes 1 from **below** | s7 (ID 0.86 [0.68, 1.00]; shift 0.91 [0.77, 1.00]) | s8 shift 0.96 [0.91, 0.99] |
| pooled 2-level CI, ID / shift | [0.92, 1.03] / [0.93, 1.02] | [0.98, 1.03] / [0.98, 1.02] |
| regime size n₁ (ID) | 200–582 | 255–633 |

**Answer:** 0/20 seed-runs pass. The paper's "only one of six seed-runs passed" becomes **"none of twenty"**; the one original pass (ETTh2 s1) came from the mismatched-SAE construction and does not reproduce (corrected R_Q 1.03 [0.96, 1.11]). Per-seed values: `AGG_F1_regime_gating.csv`.

## 2. Pooled type-B enrichment test, 30 runs

| group | mean enrichment [two-level 95% CI] | Fisher / Stouffer p vs **random draw** (hypergeometric) | seeds above **selection null** (perm-U q95) | mean selection-null enrichment | Fisher / Stouffer p vs **selection null** |
|---|---|---|---|---|---|
| all 30 | 2.52 [2.13, 3.33] | 9.5e-18 / 2.2e-21 | 9/30 | 1.83 | 5.6e-10 / 7.4e-9 |
| ETTh1 (10) | 1.97 [1.49, 2.64] | 1.4e-4 / 9.1e-6 | **0/10** | 1.82 | **0.62 / 0.34** |
| ETTh2 (10) | 2.38 [1.82, 3.26] | 6.3e-8 / 3.6e-9 | 2/10 | 1.64 | 7.5e-4 / 2.3e-4 |
| electricity (5) | 3.52 [2.26, 6.01] | 9.6e-7 / 1.7e-7 | 4/5 | 2.04 | 7.9e-8 / 9.7e-8 |
| traffic (5) | 2.91 [1.56, 5.30] | 1.9e-4 / 9.4e-5 | 3/5 | 2.04 | 9.2e-5 / 8.6e-4 |

**Important for the text.** The hypergeometric test (selected features as a random draw from the pool) is the wrong null. The MVDF rule ranks by a residual of U on P and keeps only monotone curves, and on shuffled U it already produces **≈ 1.8× enrichment on average** (round-2 WP3 perm-U null, now run on all 30 runs). Against that null:
- the pooled population-level effect holds (Fisher p = 5.6e-10, Stouffer p = 7.4e-9 over 30 runs);
- it is carried by ETTh2, electricity and traffic;
- **ETTh1 shows no enrichment beyond what the selection rule produces by itself** (0/10 seeds, Fisher p = 0.62).

Suggested sentence: "Pooled over 30 runs, type-B enrichment exceeds what the selection rule produces from permuted uncertainty effects (Fisher p = 6×10⁻¹⁰), driven by ETTh2, electricity and traffic; on ETTh1 it does not exceed this selection null (p = 0.62)." Per-run values: `AGG_F2_enrichment_tests.csv`; pooled: `AGG_F2_pooled.csv`.

## 3. Do the learned scorers also use immediate-feedback ACI?

**Yes, identically.** The learned scorer is `DifficultyScorer.run` → `FFDCIHeadACI.run` (`calibration/ffdci_head.py:77–91`). Window t's half-width is `q_resid + ĝ(x_t) + a_t`. The global ACI term `a_t` is then updated (`self.aci.update(cov[t])`, line 90) with whether the **whole 96-step** target of window t was covered. naive_matched (`calibration/difficulty_scorer.py:105–126`) uses the same `GlobalACI` class, the same γ = 0.03 and the same 20× clip, updated the same way (line 125). Both see window t's full outcome before window t+1, i.e. 95 steps before it is observable. **The comparison is like-for-like in its feedback; neither is deployable as run.**

Round 2 re-ran the comparison without this feedback (`runs/p126_round2/summary/wp1_continuous_red.csv`, seeds 0–2), with the same result: no learned scorer beats naive_matched consistently. Under static calibration the width reduction vs naive_matched on ETTh shift is −0.088 to +0.003 (dense −0.09 to −0.05, i.e. wider; mvdf ≈ 0); under delayed ACI it is mixed in sign. One sentence in Limitations covers it, e.g. "Both the learned scorers and the constant-width control use the same online ACI correction with immediate feedback, which at H = 96 uses outcomes not yet observed; re-running both with delayed feedback or static calibration gives the same ordering."

## 4. "Profile is block-local"

`validation/errors.collect(model, X, Y, tvar)` computes, **from the windows it is given**:
```
profile_h = mean_t |e_{t,h}| / mean_h mean_t |e_{t,h}|        t over all windows passed in
S_t       = max_h |e_{t,h}| / profile_h
```
(`validation/errors.py:58–61`). Every caller passes one block at a time, so each block is normalised by its own realised errors.

| analysis | block passed to `collect` | windows in the profile (ETTh) | S_t on an eval block uses that block's own errors? |
|---|---|---|---|
| dose-response (P_j, U_j) | E_select | 1872 | n/a (selection block) |
| Phase 3 battery | E_eval_ID, whole | 1966 (fit 1179 + eval 787 together) | **yes**: the eval targets and both splits share one profile from all 1966 windows |
| Phase 4 oracle | E_eval_ID; E_eval_shift | 1966; 1871 | **yes**, each block its own |
| conformal (Task G, CQR, PID) | scorer-train; conformalize; ID width-eval; shift | 983; 589; 394; 1871 | **yes**: calibration scores and test scores use *different* profiles, the test profile comes from the test outcomes, and the band shape `q·profile_h` also uses the test-block profile |

So yes: S_t on an evaluation block is normalised by a profile computed from that same block's realised errors (all windows in it, including later ones). It is a per-horizon-step rescaling (mean-1 shape), not a level. Its effect on the conclusions is small:
- Round 2 found the two definitions rank-correlated at ≈ 0.97 on E_eval_ID, and the Phase-3 conclusion unchanged under the E_select profile.
- E3, E5 and F use the E_select profile throughout (computed before every evaluation block), and reach the same Phase-3 null.

## 5. Shift-block (E_eval_shift) gap recovery

Static split conformal, calibrated on E_eval_ID (first 80% for naive; Mondrian bins on the 30% conformalize block). "Valid" = |coverage − 0.90| ≤ 0.02. GRR = (W_naive − W)/(W_naive − W_oracle). Full table: `AGG_F5_shift_gap.csv`.

| dataset | naive_matched shift coverage | valid cells (all sets × seeds) | GRR on valid cells |
|---|---|---|---|
| ETTh1 | 0.78 (invalid, 0/10) | 48/90 | **all negative**: −0.93 to −2.72 (every set, incl. MVDF −0.97 to −1.21) |
| ETTh2 | 0.91 (valid 8/10) | 18/90 | **all negative**: −0.20 to −0.41 |
| electricity | 0.57 (invalid, 0/5) | 0/45 | none valid (score shift: every static band covers 0.35–0.64) |
| traffic | 1.000 (invalid, 0/5) | 7/45 | **positive**: F3 (forecast stats) +0.28, +0.41, +0.31, +0.36 on 4/5 seeds; F4 +0.41 (1/5); F5 +0.43 (1/5); F7 +0.27 (1/5) |

C3.4 stays **FAIL** on every dataset. The pre-registered "best ex-ante set" is chosen by CV inside the training block, and on traffic that is F6, which covers 1.000.

**The traffic positives need care.** The traffic constant band is inflated by the calibration-block outlier stretch: W_naive ≈ 13.6 against an oracle of ≈ 3.5, i.e. 3.9× the oracle, at coverage 1.000. F3's Mondrian band (W ≈ 9.6–10.9) is narrower partly because its per-bin quantiles isolate the outlier windows. It is still ≈ 3× the oracle. This is better read as "a binned band is less damaged by a calibration outlier" than as "difficulty was predicted". F3's rank signal on traffic is real but modest (ρ ≈ 0.20).

**So the paper sentence should read:** "Under shift, no ex-ante band chosen by cross-validation recovers any of the oracle gap at valid coverage. Every valid ETTh cell is wider than the constant band (GRR −0.20 to −2.72), electricity's static bands all lose coverage (0.35–0.64), and the only positive cases (traffic, forecast statistics, GRR 0.28–0.41 on 4/5 seeds) occur against a constant band inflated by a calibration outlier (coverage 1.000)."

## Check of the planned §5.6 sentence
"The learned scorer never gives a narrower band at valid coverage on any of 30 runs" is **true as stated**: 0 of 120 (run × mode × block) cells have scorer width < naive_matched with coverage in [0.88, 0.92].
- The scorer is narrower in point estimate in 19 cells: ETTh2 ID dense 8/10 and mvdf 6/10, electricity shift mvdf 5/5.
- All 19 are at coverage 0.62–0.85, against naive_matched's 0.85–0.90.
- Note that ETTh ID coverage is below 0.88 for *every* method (naive_matched included, ≈ 0.83–0.85), so on the ETTh ID blocks no width comparison is at valid coverage.

The counts 19/30 (dense) and 15/30 (mvdf) for "naive_matched significantly narrower, shift block" are correct (4+10+0+5 and 0+10+0+5).
