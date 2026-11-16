# What changed from submission to camera-ready

The submitted version was a 12-page regular paper. The accepted version is an 8-page short paper.
Reviewer concerns are paraphrased below; no reviewer text is quoted. Each concern links to the
experiment that addressed it and to the change in the paper.

| concern (paraphrased) | experiment | what changed in the paper |
|---|---|---|
| Heterogeneity in realised errors does not show it can be predicted before forecasting, so the oracle reduction should not be presented as recoverable | E3, E1, F5 | Oracle reframed as an **upper bound** (Abstract, §3.4, §5.4). New ex-ante baselines (past errors, input and forecast statistics, dense state, full SAE code; six estimators chosen by time-series CV): nothing predicts difficulty on ETTh1; on ETTh2 only the dense state and full code do. Mondrian bands recover none of the gap at valid coverage. |
| Negative predictive results rest on a restricted set of estimators | E3 | Ridge and random forests added; estimator chosen by CV inside the fit split; full estimator × feature-set grid reported in `results/camera_ready/E3/RESULT.md`. |
| Negative results come mainly from two datasets | E1 | Phase 3, Phase 4 and E3 extended to electricity and traffic (5 seeds each): the L1 null and the oracle gap hold on all four datasets (§5.2, §5.4, Figure 2). |
| Is the failure specific to the architecture, representation or estimators? | E6, E3 | TimeMixer: L1 null 6/6, oracle 24–30% (§5.5). Dense state and full SAE code tested as alternative representations (Table 3). |
| Three repetitions are too few, and SD over three values is not convincing | E2 | 10 seeds on ETTh1/ETTh2 and 5 on electricity/traffic; per-run values, median [min, max] and k/N, no SDs (§4). Medians did not move beyond their pooled CIs. Per-run significance of the scorer gap weakened (§5.4, Table 4). |
| The causal-specificity score is heuristic | E4 | Alternative residualisers: the null holds under all of them; enrichment depends on the residualiser (§5.5). |
| Separate the effect of artificial interventions from the role of naturally used mechanisms | E5, G4 | Dose-response fit quality logged (median R² 0.999); natural-range interventions (λ ≤ 1) and interchange interventions (§5.5). The main L2 evidence is now **held-out** intervention effects (Table 2), and the claim is stated at the population level (§5.1, §6). |
| Definitions unclear (difficulty score, 90% interval, coverage, width) | E0 | Exact definitions from code (`docs/DEFINITIONS.md`): S_t, block-local profile, coverage = all 96 steps of one variable, width = 2q (§3.1). |

## Corrections not prompted by reviewers (found during the camera-ready work)

- §5.6 (submitted) "naive_matched significantly narrower on 12/12 runs" → significant on 19/30
  (dense) and 15/30 (MVDF), and the learned scorer is never narrower at valid coverage (Table 4).
- Regime gating: the submitted version's single passing run came from a mismatched feature list
  → 0/20 pass (§5.4).
- SAE reproducibility: report text said 0.5 SD (the threshold used is 0.25 SD); the
  "shuffled-signature floor" is invalid and was deleted (§5.5).
- Type-B enrichment is now tested against a selection null (≈ 1.8×): it holds when pooled but
  not on ETTh1 (§5.1, Table 2).
- Limitations add immediate-feedback ACI (both methods) and the static-calibration check (§6).

See `docs/KNOWN_ISSUES.md` §A for three remaining mismatches between `main.tex` and the results.
