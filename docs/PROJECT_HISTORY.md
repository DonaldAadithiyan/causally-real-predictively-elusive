# Project history

A dated, factual account of the project, written from the stage reports in `docs/reports/` and the
git log. Dates before 2026-11-14 come from the reports, since the project was not under version
control until then.

## 1. v1 — minimum viable experiment (to early September 2026)

- **Question:** can SAE features of a forecaster's hidden state, validated by intervention (MVDF),
  condition conformal intervals better than a dense representation, especially under shift?
  Spec: `docs/reports/PROJECT_REPORT.md`.
- **Runs:** a synthetic control, then five Kaggle runs (T4 GPU) on ETTh1/ETTh2/electricity/traffic.
  - Run 1 was partly invalid; an ablation bug averaged effects over inactive windows.
  - Runs 2–3 fixed SAE stability and the solar loader.
  - Run 4 showed an apparent positive result ("Outcome A").
- **Calibration bug (found after Run 4):**
  - Coverage was scored as "all 96 steps inside the band", but the nonconformity score was
    `mean_h |e_h|`, so the interval undercovered.
  - The global ACI was seeded with `base_q = 0`, so its correction never grew.
  - The Run 4 result was an artefact of the undercoverage.
- **Fix:** the score became `S_t = max_h |e_h| / profile_h`, the smallest half-width multiplier
  covering the whole window.
- **Run 5**, seed-replicated with correct calibration, is the valid v1 result (`results/v1/mve_runs/`).
- **Decision:** move from a single scorer comparison to the v2 mechanistic programme.

## 2. v2 phases 0–2 (2026-09-03 – 09-04)

- **Phase 0, layer sweep:** seven taps, all with weak hidden–error correlation (0.08–0.21). The
  pre-head tap was kept as the default.
- **Phase 1, MVDF with R_j:**
  - Built the dose-response protocol (λ ∈ {0, …, 1.25}) and the specificity score R_j.
  - **H1c:** MVDF was type-B-enriched on 12/12 Kaggle runs, while correlation screening and
    single-point ablation did not find these features.
  - Reports: `V2_PHASES012_REPORT.md`, `V2_CONFIRM_REPORT.md`.
- **Phase 2, invariance I(R) (H3):** MVDF was no more shift-invariant than a correlation-selected
  set, and the metric was degenerate for sparse sets.
- **H6 / Task G, downstream:** a constant-width band (naive_matched) beat the learned scorer
  under shift, 12/12 significant (iid bootstrap, Kaggle copy).
- Final v2 report: `V2_FINAL_REPORT.md`. Frame locked in `PAPER_FRAME.md` (2026-09-04).

## 3. Gates and phases 3–5 (2026-09-04 – 09-09)

| stage | report | outcome |
|---|---|---|
| H7-A regime gating | `H7A_RESULT.md` | negative, 1/6 runs passed |
| SLCP localization | `SLCP_RESULT.md` | stopped: a time-only control explained the apparent structure |
| CQR / PID-conformal baselines | `BASELINE_RESULT.md` | no consistent gain over naive_matched |
| Phase-1 synthetic positive control | `results/v2/phase1_synthetic/PHASE1_RESULT.md` | links 1–3 pass 3/3; link 4 passes 1/3 |
| Phase 3, natural predictability | `results/v2/phase3_real/` | MVDF activation predicts difficulty worse than a constant |
| Phase 4, oracle ceiling | `results/v2/phase4_oracle/` | 24.5–29.2% |
| Phase 5, SAE reproducibility | `results/v2/phase5_sae_repro/` | feature identity does not survive reseeding |
| Phase 2, necessity | `results/v2/phase2_necessity/` | not established |
| TimeMixer, second architecture | `results/v2/arch_gen_timemixer/` | 6/6 on all three gate checks |

Consolidated in `RESULTS_FINAL.md`.

## 4. Submission and acceptance

- **Submission:** a 12-page regular paper to ICAART 2027 (#126), September 2026.
- **Acceptance:** accepted as an 8-page short paper. Reviewers asked for:
  - more seeds;
  - more datasets for the negative results;
  - ex-ante baselines, so the oracle gap is not read as recoverable;
  - robustness of the heuristic specificity score;
  - a separation of artificial interventions from natural mechanisms;
  - clearer definitions.

  Paraphrased in `docs/CAMERA_READY_CHANGES.md`.

## 5. Follow-up rounds 1–2 (2026-11-10 – 11-13)

- **Round 1** (`results/round2/p126/RESULTS.md`, brief `docs/reports/paper126_followup_coding_brief.md`):
  - the negative R² is a level shift;
  - the oracle measures marginal spread, not predictability;
  - a scorer clip bug was found.
- **Round 2** (`results/round2/p126_round2/RESULTS.md`, brief `paper126_round2_coding_brief.md`):
  - built a corrected evaluation path and saved weights and per-window arrays for every model;
  - no dataset has predictable difficulty;
  - enrichment beats a selection null only on electricity/traffic;
  - the synthetic link 4 passes 3/3 on the corrected path but is not MVDF-specific.

## 6. Camera-ready experiments (2026-11-14 – 11-16)

Spec: `docs/reports/CAMERA_READY_EXPERIMENTS.md` (redacted). Every PREREG was committed before its
run (`docs/PREREGISTRATION.md`). Results: `results/camera_ready/`, summary
`results/camera_ready/CAMERA_READY_SUMMARY.md`.

- **E0:**
  - fixed the Phase-5 threshold text;
  - found the Phase-5 shuffle floor is invalid;
  - wrote the definitions audit (`docs/DEFINITIONS.md`).
- **E2 (more seeds):**
  - 10 seeds on ETTh1/ETTh2 and 5 on electricity/traffic (30 runs), 0 unhealthy;
  - enrichment ≥ 1×, the Phase-3 null and the oracle gap hold on 30/30;
  - the "12/12 significant" scorer claim does **not** hold run by run (ETTh1 4/10 dense, 0/10 MVDF).
- **E3 / E1 (ex-ante baselines, four datasets):**
  - nothing available at forecast time predicts difficulty under the pre-registered rule (C3.1 FAIL);
  - on ETTh2 the dense hidden state does (10/10) and MVDF does not (0/10);
  - no ex-ante band recovers the oracle gap at valid coverage (C3.4 FAIL);
  - the oracle is therefore reframed as an upper bound.
- **E4 (R_j robustness):** the null holds under every residualiser; enrichment fails under the
  binned mean (C4.2 FAIL).
- **E5 (dose-response):**
  - clean curves (median fit R² 0.999);
  - natural-range reselection: Jaccard 0.85 (C5b FAIL on one boundary cell);
  - interchange sign agreement 81% (C5c FAIL on ETTh1, 7/10).
- **E6 (TimeMixer):** regenerated exactly; the L1 null holds 6/6.
- **F (follow-up):**
  - regime gating fails on 0/20 runs; the original 1/6 pass came from a mismatched-SAE feature list;
  - enrichment beats the selection null when pooled (p = 6×10⁻¹⁰) but not on ETTh1 (p = 0.62);
  - the learned scorers and naive_matched share the same immediate-feedback ACI;
  - the profile is block-local.
- **G (stopped on request):**
  - G2: the pre-evaluation profile changes no conclusion;
  - G4: on held-out windows MVDF features keep a significant intervention effect no more often
    than the rest of the pool (15/30), while 17–47% of all features have one;
  - G5: static-calibration range −0.09 to +0.004;
  - G1 and G3 are partial.

**Corrections these rounds forced in the paper:**
- §5.6 "12/12 significant" → significant on 19/30 (dense) and 15/30 (MVDF) runs, and never narrower at valid coverage;
- regime gating 1/6 → 0/20;
- the shuffle-floor clause was deleted;
- enrichment is now tested against the selection null;
- MVDF's held-out advantage is 15/30, so the causal claim is stated at the population level.

## 7. Public release (2026-11-16)

The repository was restructured into this layout with history preserved
(`scripts/dev/migrate_layout.py`; no logic changes, verified by exact reproduction). Personal paths
were scrubbed from history, and checks were added that tie every number in the paper to a committed
result. Packaging found two double-rounding errors and one wording mismatch in the camera-ready text
(`docs/KNOWN_ISSUES.md` §A).
