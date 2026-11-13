# Project notes & status

## Where this is (2026-09-03)

The full pipeline from the spec is implemented and runs end-to-end on **real data with
a real forecaster**:

- `data/datasets.py` — ETT auto-downloads; traffic/electricity/weather load from a
  Kaggle-attached dataset (`/kaggle/input/**/<name>.csv`, auto-discovered).
- `forecasters/itransformer.py` + `train_itransformer.py` — real iTransformer
  (ICLR'24), the FFDCI backbone. Test MSE ~0.47 on ETTh1 (matches a linear+RevIN
  baseline; published SOTA is ~0.39 with more capacity/training — not the point,
  spec §5.2).
- `sae/` — TopK SAE, Dissecting-Chronos recipe, 8x / k=64, dead-feature resampling.
  Reconstruction FVU ~1.2%.
- `validation/` — correlation screen, causal ablation (active-window-conditioned
  `Δ_j`), RQ1 rank-divergence diagnostic.
- `calibration/` — Variant B (Mondrian bins + per-bin ACI, the paper's contribution)
  and Variant A (FFDCI-style MLP + global ACI).
- `eval/` — coverage/width/MACE/Winkler, degradation-under-shift, plots.
- `shift/shift_windows.py` — **pre-registered** shift windows per dataset, with
  rationale, committed before the main comparison. KS-checked at run time.
- `experiments/mve_real.py` — the single-dataset MVE. `full_suite.py` — the loop.
- `notebooks/mve_kaggle.ipynb` — self-contained, vendors all of the above, runs the
  full suite on a Kaggle GPU. Regenerate with
  `python notebooks/build_kaggle_notebook.py --full-suite`.

## The open scientific question

On all four ETT datasets, **per-instance difficulty is only weakly readable from the
iTransformer's pre-head hidden state** (`max|corr(hidden PC, |e|)| ≈ 0.08–0.11`).
That is a real property of ETT (near-homoskedastic errors), not a bug — verified by
a linear+RevIN baseline hitting the same MSE, and by the signal being just as weak on
the better-fitting ETTh2/ETTm2.

Consequences on ETT:
- causal ablation `Δ_j` is near float noise; `Δ>0` rate ≈ 0.48 (spec §3.2 reports ~1.0
  on Chronos-Large)
- MVDF selection ≈ random; `|corr ∩ mvdf|` ≈ 1/12
- H3 `Delta_shift` signs disagree across metrics — no result

## Next step (decided: push to Kaggle)

Run `mve_kaggle.ipynb` on **traffic + electricity** (strong regime structure — rush
hours, weekday/weekend — the spec's own headline datasets). The run reports
`corr(hidden, |e|)` per dataset as the first-order screen:

- **high on traffic/electricity + MVDF beats correlated under shift** → outcome A,
  scale to the full FFDCI suite (spec §7 decision rule).
- **weak everywhere** → outcome B ("difficulty weakly represented in competent
  forecasters on LTSF benchmarks; validated features can't help where uncertainty is
  irreducible") — pre-registered as publishable (spec §4.4, §4.5). The synthetic
  control (`experiments/mve_synthetic.py`) then does real work: it shows the machinery
  *does* find and exploit a difficulty feature when one is present by construction.
- **correlated beats MVDF under shift** → outcome C.

All three are written into `configs/mve_etth1.yaml`'s header before results were seen.

## Synthetic control

`experiments/mve_synthetic.py` — NOT a result. A controlled sanity check with known
ground truth (a mechanistic difficulty coordinate the head reads + a spurious one it
ignores that decouples under a pre-registered shift). Used to (a) debug the pipeline,
(b) in the paper, demonstrate the validation step works when a mechanism exists —
which makes a negative result on real data a scope statement, not a failure.
