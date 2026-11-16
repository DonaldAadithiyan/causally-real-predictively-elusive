# Causally Real, Predictively Elusive

Code, results and pre-registrations for **"Causally Real, Predictively Elusive: Mechanistic
Difficulty Representations in Neural Forecasters"** (Donald Aadithiyan, University of Moratuwa;
ICAART 2027, short paper).

Interpretability methods can show, by intervention, that internal features of a neural forecaster
change how large its errors are. This project asks whether such features also *signal*, through
their ordinary activation, when a forecast will be hard, and whether that signal can make
prediction intervals narrower. The central claim is that **causal relevance (L2, intervention),
natural predictive information (L1, association) and downstream usefulness are separate
properties.** Each has to be tested; none follows from another. The headline evidence:

- **Causal relevance holds at the population level.** On held-out windows, 17–47% of sparse-autoencoder
  features (median per dataset) shift forecast difficulty under intervention, against about 5%
  expected with no effect.
- **No natural predictive information.** Across 4 datasets, 2 architectures and 36 trained runs,
  the natural activation of the causally selected features predicts difficulty *worse than a
  constant* on every run (held-out R² < 0, every bootstrap interval below zero).
- **Not recoverable downstream.** Difficulty does vary: a hindsight oracle could narrow 90%
  intervals by 18–39%. That is an upper bound. No predictor using forecast-time information, no
  SAE-based method and no standard adaptive conformal method recovers it consistently.

```bibtex
@inproceedings{aadithiyan2027causally,
  title     = {Causally Real, Predictively Elusive: Mechanistic Difficulty Representations in Neural Forecasters},
  author    = {Aadithiyan, Donald},
  booktitle = {Proceedings of the International Conference on Agents and Artificial Intelligence (ICAART 2027)},
  publisher = {SCITEPRESS},
  year      = {2027}
}
```

**Terms, one line each.**
- **S_t (difficulty):** max over the 96 forecast steps of |error| / mean error at that step.
- **Coverage:** a window is covered only if all 96 future values of the target variable lie inside the band.
- **L1 / L2:** evidence from watching the model run / from intervening on it.
- **MVDF:** the 12 SAE features with the largest difficulty effect beyond their effect on the
  forecast, among those with a monotone six-point dose-response.
- **Oracle:** a non-deployable band that bins windows by their *realised* difficulty; it bounds how much width difficulty-awareness could save.

---

## Quickstart

```bash
git clone https://github.com/DonaldAadithiyan/causally-real-predictively-elusive && cd causally-real-predictively-elusive
python -m venv .venv && source .venv/bin/activate
make install        # pinned requirements + editable package (~3 min)
make data           # downloads ETTh1/ETTh2 and checks SHA-256 (~10 s)
make smoke          # end-to-end pipeline on a tiny config, CPU (~30 s)
make tables         # Tables 2-4 from results/ -> paper/tables/ (~5 s)
make figures        # Figure 2 from results/ -> paper/figures/ (~5 s)
make test           # leakage + every paper number + smoke (~1 min)
```

`make tables`, `make figures` and the paper-number tests need only the committed CSVs. Re-running
analyses needs the large artifacts: `python scripts/fetch_artifacts.py` (≈ 2 GB, SHA-256-checked
against `artifacts/MANIFEST.sha256`).

Environment: Python 3.12.11, torch 2.6.0, numpy 2.4.4, scikit-learn 1.5.2, pandas 3.0.2,
scipy 1.17.1 (`requirements.txt`).

## Paper ↔ code map

Commands run from the repository root. All camera-ready scripts take the frozen settings from
`experiments/v2/phase3_real_predictability.py::CFG` (the paper's configuration) and are driven by the
command-line arguments shown; `scripts/reproduce_all.sh` has every stage. Every number marked ✓ is
recomputed from the listed file by `tests/test_paper_numbers.py` (`paper/paper_numbers.yaml`).

| paper item | what it shows | script + command | output file(s) | runs |
|---|---|---|---|---|
| Table 1 (notation) | evidence level of P_j, U_j, R_j, a_j → S_t | definitions in `docs/DEFINITIONS.md` | — | — |
| Figure 1 (pipeline) | stages and data flow | — (drawn by hand; not included, see Known issues) | — | — |
| §3.1 layer sweep "seven", "0.08–0.21" ✓ | weak hidden–error correlation at every tap | `experiments/v2/phase0_layer_sweep.py` (Kaggle) | `results/v2/v2_runs/v2_phase0/layer_sweep.csv` | 1 |
| §3.1 profile rank agreement "0.95–0.997" ✓ | block-local vs pre-evaluation profile | `python -m experiments.camera_ready.g_analysis g2` | `results/camera_ready/G/AGG_G2_select_profile.csv` | 30 |
| §4 KS shift test "p<0.05" ✓ | E_eval_shift differs from E_select | `experiments/v2/phase1_core_mechanism.py` (Kaggle) | `results/v2/v2_confirm/v2_phase1_seed*/run_log.json` | 4 datasets |
| §4 caveats "0.86", "1.000" ✓ | ETTh ID under-coverage; traffic saturation | `python -m experiments.camera_ready.e2_scorer …` | `results/camera_ready/E2/AGG_E2_scorer_vs_naive.csv` | 30 |
| **Table 2**, §5.1 "17–47%", "15/30", "30/30" ✓ | held-out intervention effects (L2) | `python -m experiments.camera_ready.g_analysis g4` | `results/camera_ready/G/AGG_G4_L2_evidence.csv`, `G4_per_feature_U_ci.csv` | 30 |
| Table 2 enrichment, §5.1 "2.0–3.9×"‡, "all 30" ✓ | type-B enrichment of MVDF | `python -m experiments.camera_ready.report_e2` (from E5a) | `results/camera_ready/E2/AGG_E2_enrichment.csv` | 30 |
| Table 2 selection null, "1.8×", "p = 6×10⁻¹⁰", "0.62" ✓ | enrichment beyond what selection alone produces | `python -m experiments.camera_ready.f_followup` | `results/camera_ready/F/AGG_F2_pooled.csv`, `AGG_F2_enrichment_tests.csv` | 30 |
| §5.1 "0.999" ✓ | dose-response linearity | `python -m experiments.camera_ready.e5_dose a …; …report_e5` | `results/camera_ready/E5/AGG_E5a_diagnostics.csv` | 20 ETTh |
| **Figure 2a**, §5.2 R² per dataset ✓ | natural MVDF activation does not predict S_t (L1) | `python -m experiments.camera_ready.phase34 …`; `make figures` | `results/camera_ready/E2/AGG_E2_phase3.csv`, `paper/figures/fig_dissociation_col.pdf` | 30 |
| §5.2 active-window R² ✓ | null not caused by rare firing | same | `results/camera_ready/E2/AGG_E2_phase3.csv` (`cond_best_r2`) | 30 |
| §5.2 "Spearman −0.10 to +0.13" ✓ | R_j unrelated to predictive score | `experiments/v2/phase3_real_predictability.py` | `results/v2/phase3_real/AGG_phase3.csv` | 6 |
| **Table 3**, §5.2 ex-ante ("10/10", "7/10", "0.33", "1/5", "29 of 30") ✓ | what is predictable at forecast time | `python -m experiments.camera_ready.e3_exante …; …report_e3` | `results/camera_ready/E3/AGG_E3_predictability.csv`, `E1/AGG_E1_E3_predictability.csv`, `*_delta.csv` | 30 |
| §5.3 synthetic control "3/3 ×3, 1/3", "0.26–0.48" ✓ | pipeline detects a built-in signal | `python -m experiments.v2.phase1_synthetic_control` | `results/v2/phase1_synthetic/AGG_phase1.csv` | 3 |
| **Figure 2b**, §5.4 oracle medians, "24.0–37.3%" ✓ | available difficulty variation (upper bound) | `python -m experiments.camera_ready.phase34 …` | `results/camera_ready/E2/AGG_E2_phase4.csv` | 30 |
| §5.4 ex-ante bands "−0.20 to −2.72", "0.35–0.64", traffic "0.28–0.41 on 4/5" ✓ | Mondrian gap recovery under shift | `python -m experiments.camera_ready.f_followup` (from E3) | `results/camera_ready/F/AGG_F5_shift_gap.csv` | 30 |
| **Table 4**, §5.4 "120", "0", "0.89–0.92", "0.62–0.85" ✓ | learned scorer vs naive_matched | `python -m experiments.camera_ready.e2_scorer …` | `results/camera_ready/E2/AGG_E2_scorer_vs_naive.csv` | 30 |
| §5.4 earlier 12 runs "3.89, 3.94, 4.36 vs 3.35", "within 0.02" ✓ | other inputs and calibration budget | `notebooks/v2_h6_topup.ipynb` (Kaggle) | `results/v2/v2_h6_topup/AGG_h6.csv` | 12 |
| §5.4 regime gating "None of 20", "0.86–1.20", "0.97–1.05", "200–633" ✓ | type-B regimes have no heavier tail | `python -m experiments.camera_ready.f_followup` | `results/camera_ready/F/AGG_F1_regime_gating.csv` | 20 |
| §5.4 localization "one third", "0.94 (2/6)", "0.81 (2/6)"‡, "0.73 (4/6)" ✓ | feature-space neighbours | `python -m experiments.v2.slcp_gate` | `results/v2/slcp_gate/SLCP_gate_decision.json` | 6 |
| §5.4 PID "−0.005", CQR "0.71–0.77", "0.98–0.99" ✓ | standard adaptive conformal baselines | `python -m experiments.v2.baseline_comparison` | `results/v2/baseline_comparison/AGG_baselines.csv` | 6 |
| §5.5 R_j alternatives "0.50–0.71", "7/20, 17/20, 5/20, 0/20" ✓ | robustness of the specificity score | `python -m experiments.camera_ready.e4_rj; …report_e4` | `results/camera_ready/E4/AGG_E4_rj_robustness.csv` | 20 ETTh |
| §5.5 natural range "8%", "0.85", "81%" ✓ | in-range and interchange interventions | `python -m experiments.camera_ready.e5_dose a|c …; …report_e5` | `results/camera_ready/E5/AGG_E5a_*.csv`, `AGG_E5b_natural_range.csv`, `AGG_E5c_interchange.csv` | 20 ETTh |
| §5.5 TimeMixer "0.0008–0.0027", "1.2–2.5×, 6/6", "6/6", "24–30%", "−0.49 to −1.17"‡ ✓ | second architecture | `python -m experiments.camera_ready.e6_timemixer …`; `experiments/v2/arch_gen_timemixer.py` | `results/camera_ready/E6/AGG_phase3.csv`, `AGG_phase4.csv`, `AGG_E6_predictability.csv`; `results/v2/arch_gen_timemixer/AGG_arch_gen.csv` | 6 |
| §5.5 SAE reproducibility "8–33% (mean 25%)", "28–43%" ✓ | feature identity across SAE seeds | `python -m experiments.v2.phase5_sae_reproducibility` | `results/v2/phase5_sae_repro/AGG_phase5.csv` | 4 comparisons |
| §5.5 necessity "−0.0004 to −0.0021 on 5 of 6" ✓ | sufficiency, not necessity | `python -m experiments.v2.phase2_necessity_noising` | `results/v2/phase2_necessity/AGG_phase2.csv` | 6 |
| §6 static calibration "−0.09 to +0.004" ✓ | scorer vs naive without ACI feedback | `python -m experiments.camera_ready.g_analysis g5` (reads round 2) | `results/camera_ready/G/AGG_G5_static_shift.csv` | 6 |

‡ The printed value differs from the committed result: TimeMixer −1.17 should be −1.16;
enrichment 2.0–3.9× should be 2.0–3.8×; correlated localization "2/6 below the bar" is 3/6
below 0.80 (2/6 pass the full rule). See `docs/KNOWN_ISSUES.md` §A.

Full per-experiment reports: `results/camera_ready/E0`–`E6`, `F`, `G` (`RESULT.md` in each), and
`results/camera_ready/CAMERA_READY_SUMMARY.md`.

## Pipeline

1. **Forecaster and SAE.** An iTransformer (d_model 256, 3 layers, input 96 → horizon 96) is
   trained per seed. A TopK SAE (2048 features, 64 active) is fitted to the hidden state just
   before the output head (`src/mechconf/forecasters`, `src/mechconf/sae`).
2. **Dose-response.** For 120 candidate features, each feature's code is scaled by
   λ ∈ {0, 0.25, 0.5, 0.75, 1, 1.25} and the SAE reconstruction is patched in. The slopes give P_j
   (effect on the forecast) and U_j (effect on difficulty)
   (`src/mechconf/validation/dose_response.py`).
3. **MVDF selection.** The specificity score is R_j = U_j − Ê[U_j | P_j] (isotonic). The top 12
   features with monotone curves are kept, and features are typed A–D by median splits
   (`src/mechconf/validation/difficulty_specificity.py`).
4. **Natural predictability (L1).** Held-out estimators predict S_t from natural activations
   (MVDF) and from ex-ante feature sets (`experiments/camera_ready/phase34.py`, `e3_exante.py`).
5. **Oracle ceiling.** Leave-one-out quantiles within deciles of realised S_t
   (`experiments/v2/phase4_oracle_ceiling.py`).
6. **Downstream.** Continuous scorer, regime gating, localization, CQR, PID and Mondrian bands,
   each against naive_matched (`src/mechconf/calibration`, `src/mechconf/baselines`,
   `experiments/camera_ready/e2_scorer.py`).

## Reproducing everything

`bash scripts/reproduce_all.sh <stage>|all` (or `make reproduce-<stage>`). Stages, in order, with
measured wall-clock on the original hardware:

| stage | what | time |
|---|---|---|
| `train` | forecaster + SAE + MVDF for seeds 3–9 (ETTh, CPU) and 3–4 (electricity/traffic, MPS) | ~3.5 h |
| `arrays` | traffic seed 0–2 arrays from saved weights | ~1 min |
| `phase34` | Phase 3 battery + Phase 4 oracle from arrays | ~1 min |
| `scorer` | learned scorer vs naive_matched | ~1 min |
| `e3` | ex-ante baselines + gap recovery, 30 cells | ~3 h |
| `e5`, `e4` | dose-response diagnostics; R_j robustness | ~5 min each |
| `e6` | TimeMixer regeneration + analysis | ~1.5 h |
| `f`, `g` | follow-up F; G2/G4/G5 | ~20 min |
| `reports` | E1/E2 reports | ~1 min |

**Hardware:** Apple Silicon (arm64), no CUDA. ETTh was trained on CPU with
`torch.set_num_threads(4)`, the setting under which retrains are bit-exact. Electricity/traffic
were trained on MPS. **Bit-exact:** the ETTh1 seed 0 retrain, all TimeMixer runs, all 30
dose-response pools. **Within ≤ 3×10⁻³:** Phase 3/4 recomputed from float32 arrays. Details:
`docs/PROVENANCE.md`.

Earlier stages (v1 MVE, v2 phases 0–2, H6) ran in Kaggle notebooks (`notebooks/`) against the
pre-restructure layout. They are kept for the record, and their outputs are committed under
`results/v1/` and `results/v2/`.

## Data

`python scripts/download_data.py` fetches ETTh1/ETTh2 from the ETDataset repository and checks SHA-256:
- ETTh1 `f18de3ad…6d29695ee066`
- ETTh2 `a3dc2c59…3f944967dd6b9218b`

The electricity and traffic files are preprocessed versions from a Kaggle data bundle, with no
byte-identical public URL. Place them at `data/raw/electricity.csv` and `data/raw/traffic.csv`,
then run the script to check them:
- electricity `353cbeb1…a659bf48b1570`
- traffic `1e7b7d90…44b84535c9374d7449e23fc5ea902a9db69930d9281`

**Two copies of ETT were used.** The Kaggle bundle copy was used by the v2 Kaggle runs. The HTTP
mirror copy was used by everything from Phase 3 on, including all camera-ready results. See
`docs/PROVENANCE.md`, and `DATA_LICENSES.md` for licences. No data is redistributed here.

## Pre-registration

Every camera-ready experiment's criteria were committed to `PREREG.md` **before** its first run.
`docs/PREREGISTRATION.md` lists each commit and its time. Some pre-registered criteria **failed**,
and they are reported as failures:
- **C3.1:** difficulty is not ex-ante predictable.
- **C3.2:** dissociation (requires C3.1).
- **C3.4:** no recoverable opportunity.
- **C4.2:** enrichment is not robust to the residualiser.
- **C5b:** natural-range enrichment fails on one boundary cell.
- **C5c:** interchange sign agreement fails on ETTh1.
- **C2.1 (partly):** per-run significance of the scorer gap.

## Known issues

`docs/KNOWN_ISSUES.md` lists:
- three mismatches between the camera-ready text and the results (§A);
- the block-local difficulty profile;
- interventions that drop the SAE reconstruction error;
- immediate-feedback ACI in both naive_matched and the learned scorers;
- the traffic calibration outlier;
- the invalid Phase-5 shuffle floor;
- the H7-A mismatched-SAE bug;
- round G stopped early (G1 and G3 partial).

## Project history

`docs/PROJECT_HISTORY.md` gives a dated account from the v1 MVE to the camera-ready, and the
original task specifications are in `docs/reports/`.

## Repository layout

```
src/mechconf/        library: data, forecasters, sae, representation, validation, calibration, baselines, eval
experiments/         entry points: v1_mve/, v2/, round2/, camera_ready/
results/             committed small outputs: v1/, v2/, round2/, camera_ready/{E0..E6,F,G}
paper/               make_fig2.py, make_tables.py, paper_numbers.{py,yaml}, figures/, tables/
scripts/             download_data.py, fetch_artifacts.py, smoke_test.py, reproduce_all.sh, dev/migrate_layout.py
tests/               test_leakage.py, test_paper_numbers.py, test_smoke.py
docs/                PROJECT_HISTORY, PREREGISTRATION, DEFINITIONS, PROVENANCE, KNOWN_ISSUES, CAMERA_READY_CHANGES, reports/
artifacts/           MANIFEST.sha256 for the release assets
```

## Licence and citation

Code: MIT (`LICENSE`). Data: see `DATA_LICENSES.md`. Please cite the paper above (`CITATION.cff`).
