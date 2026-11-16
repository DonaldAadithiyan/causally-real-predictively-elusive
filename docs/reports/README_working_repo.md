# Mechanistically Validated Difficulty Features for Robust Conformal Prediction

Implementation of the pipeline in `mechanistic-conformal-project-spec-FINAL.md`.

**Central question (§4.1):** Does causally validating a sparse difficulty feature — checked at
ablation *magnitude* rather than presence — produce conformal calibration that is more robust
under distribution shift than an otherwise-identical pipeline using correlation-only feature
selection?

## What is implemented here (the §7 Minimum Viable Experiment)

- **Synthetic base forecaster + synthetic time series** with a *known injected difficulty feature*
  and a *pre-registered distribution shift*. This lets us test the whole H1/H2/H3 machinery end to
  end before swapping in a real forecaster (iTransformer/FFDCI stack). The synthetic setup is a
  scaffold, not a scientific result — see `NOTES.md`.
- **TopK SAE** (`sae/topk_sae.py`, `sae/train_sae.py`) on the forecaster's pre-head hidden state,
  Dissecting-Chronos recipe (8x expansion, k=64, MSE, dead-feature resampling).
- **Correlation screening** `Corr(z_j, |e|)` and **causal ablation** (zero → decode → patch →
  measure `|Δ_j|`), magnitude-ranked, *not* binary (`validation/`).
- **Three matched-budget feature sets**: `SAE-random`, `SAE-correlated`, `MVDF` (`SAE-validated`).
- **Calibration Variant B** (the paper's contribution): Mondrian bins on the chosen feature +
  a per-bin ACI correction term (`calibration/mondrian_perbin.py`).
- **Calibration Variant A** (FFDCI-style): pinball-loss MLP quantile regressor + a single global
  ACI correction (`calibration/ffdci_head.py`), for the interpretability-cost ablation.
- **Evaluation**: coverage (overall + worst-step), width, MACE (rolling), Winkler, timesteps-per-bin,
  in-distribution and as Δ-degradation under the pre-registered shift (`eval/`).
- **RQ1 diagnostic** (§6.5): Spearman rank correlation between the correlation ranking and the
  ablation-magnitude ranking over a non-correlation-prefiltered feature sample.

## Run

```bash
python -m experiments.mve_synthetic --config configs/mve_synthetic.yaml
```

Outputs land in `configs/../runs/<run_name>/`: `rq1_diagnostic.json`, `h2_in_distribution.csv`,
`h3_shift.csv`, `variant_ab.csv`, `delta_j_distribution.csv`, plus plots.

## Repository structure

```
forecasters/    synthetic_forecaster.py  (real iTransformer/Leddam/SOFTS to be dropped in later)
sae/            topk_sae.py, train_sae.py
validation/     correlation_screen.py, ablation.py
calibration/    ffdci_head.py (Variant A), mondrian_perbin.py (Variant B)
shift/          shift_windows.py  (pre-registered shift definitions + rationale, committed before results)
eval/           metrics.py, plots.py
experiments/    mve_synthetic.py, full_suite.py (stub)
configs/        *.yaml
```

## Pre-registration

`shift/shift_windows.py` and `configs/mve_synthetic.yaml` fix the shift window and all
outcome interpretations (A/B/C, spec §4.4) *before* the main comparison is run. See `NOTES.md`.
