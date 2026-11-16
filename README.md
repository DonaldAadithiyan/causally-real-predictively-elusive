# Causally Real, Predictively Elusive

**Code, results and pre-registrations for "Causally Real, Predictively Elusive: Mechanistic
Difficulty Representations in Neural Forecasters"** — Donald Aadithiyan, Department of Computer
Science and Engineering, University of Moratuwa. ICAART 2027 (short paper).

Release: [`v1.0-camera-ready`](https://github.com/DonaldAadithiyan/causally-real-predictively-elusive/releases/tag/v1.0-camera-ready) ·
Verified: 121 tests pass from a fresh clone ([`docs/VERIFICATION.md`](docs/VERIFICATION.md)) ·
Licence: MIT (code); data is not redistributed (see [Data](#data)).

---

## Contents

1. [In one paragraph](#in-one-paragraph)
2. [Headline results](#headline-results)
3. [Key terms](#key-terms)
4. [Quickstart](#quickstart)
5. [Installation in detail](#installation-in-detail)
6. [Data](#data)
7. [Large artifacts](#large-artifacts)
8. [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
9. [Results in detail](#results-in-detail)
10. [Paper ↔ code map](#paper--code-map)
11. [Running the experiments](#running-the-experiments)
12. [Tests](#tests)
13. [What is in `results/`](#what-is-in-results)
14. [Pre-registration](#pre-registration)
15. [Known issues and limitations](#known-issues-and-limitations)
16. [Project history](#project-history)
17. [Repository layout](#repository-layout)
18. [Troubleshooting](#troubleshooting)
19. [Citation and licence](#citation-and-licence)

---

## In one paragraph

Interpretability methods can show, by intervention, that internal features of a neural
forecaster change how large its errors are. It is tempting to conclude that such a feature is
also a useful *monitor*: that its ordinary activation warns us when a hard input arrives, and that
it can therefore make prediction intervals narrower where forecasts are easy and wider where they
are hard. This project tests that assumption directly. Using sparse autoencoders (SAEs) on an
iTransformer forecaster and a six-point dose-response intervention, we select features whose
intervention changes forecast *difficulty* more than the forecast itself. We then ask, separately,
(i) whether interventions on such features really change difficulty, (ii) whether their natural
activation predicts difficulty on new data, and (iii) whether any of this narrows a valid 90%
interval. **The answer for these forecasters: (i) yes, at the population level; (ii) no; (iii) no.
Causal relevance, natural predictive information and downstream usefulness are separate
properties, and each must be tested rather than assumed from another.**

## Headline results

| claim | evidence | runs |
|---|---|---|
| Interventions on SAE features change difficulty | on **held-out** windows, **17–47%** of candidate features (median per dataset) have an intervention effect on difficulty whose 95% interval excludes 0, against ~5% expected if interventions did nothing | 30 iTransformer runs |
| …but the *selected* features are not special out of sample | MVDF features keep a significant effect more often than the rest of the pool on only **15/30** runs (30/30 on the selection windows, i.e. selection-inflated) | 30 |
| Natural activation does **not** predict difficulty | held-out R² of the best of four estimators is **below 0 on every one of 36 runs** (4 datasets, 2 architectures), with every bootstrap interval below zero | 36 |
| Difficulty does vary | a hindsight oracle could narrow 90% intervals by **18–39%** (median per dataset) — an *upper bound*, not a recoverable opportunity | 30 |
| Nothing recovers that variation | no forecast-time (ex-ante) predictor, SAE-based method or standard adaptive conformal method is consistently narrower than a constant-width band at valid coverage; the learned scorer is narrower at valid coverage in **0 of 120** run × input × block combinations | 30 |
| The measurement can find a signal when one exists | on a synthetic regime-switching process, the same battery finds held-out R² **0.26–0.48** | 3 |

## Key terms

| term | meaning |
|---|---|
| **S_t** (difficulty) | for forecast window t: the largest absolute error over the 96 forecast steps, each step's error divided by the typical error at that step (`profile_h`). The smallest scalar half-width that would have covered the whole window. |
| **profile_h** | mean absolute error at forecast step h over the windows of the block being scored, rescaled to mean 1 over the 96 steps. |
| **90% interval / coverage / width** | every method outputs a half-width q_t; the band is ŷ ± q_t·profile_h. A window is *covered* only if all 96 future values of the target variable lie inside it (S_t ≤ q_t). Target: 90% of windows. Width = 2q_t in normalised target units. |
| **L1 / L2 / L3** | levels of evidence about model internals: L1 = association (watching the model run); L2 = intervention (changing a feature and seeing the output change); L3 = counterfactual (not claimed here). |
| **SAE** | TopK sparse autoencoder (2048 features, 64 active per input) trained on the forecaster's hidden state just before its output head. |
| **dose-response** | scale one SAE feature by λ ∈ {0, 0.25, 0.5, 0.75, 1, 1.25}, patch the SAE reconstruction into the forecaster, and record the forecast loss and S_t at each λ. |
| **P_j / U_j** | least-squares slopes over the six doses of forecast loss (P_j, effect on the forecast) and of S_t (U_j, effect on difficulty). |
| **R_j** (specificity) | U_j − Ê[U_j \| P_j], the difficulty effect beyond what the forecast effect explains (isotonic fit over a 120-feature pool). A heuristic residual, not a mediation estimate. |
| **types A–D** | median split on \|P_j\| and \|U_j\|: A forecast-only, **B difficulty-only**, C both, D neither. |
| **MVDF** | Multi-point Validated Difficulty Features: the 12 pool features with the highest R_j among those whose difficulty curve is monotone in λ. |
| **E_select / E_eval_ID / E_eval_shift** | the test windows split by time into thirds: feature selection / in-distribution evaluation / later, shifted evaluation. |
| **ex-ante** | information available when the forecast is made: past errors whose targets are already observed, input-window statistics, forecast statistics, the hidden state. |
| **naive_matched** | the main control: a constant-width conformal band calibrated on the same data the learned methods use. |
| **oracle** | a non-deployable band that bins windows by their *realised* S_t; its width saving bounds how much difficulty-awareness could help. |
| **GRR** (gap recovery ratio) | (W_naive − W_method) / (W_naive − W_oracle): the fraction of the oracle's width saving a deployable method recovers. Interpreted only at valid coverage (0.88–0.92). |

## Quickstart

```bash
git clone https://github.com/DonaldAadithiyan/causally-real-predictively-elusive
cd causally-real-predictively-elusive
python -m venv .venv && source .venv/bin/activate
make install        # pinned requirements + editable package          (~3 min)
make data           # download ETTh1/ETTh2 and check SHA-256            (~10 s)
make smoke          # whole pipeline on a tiny config, CPU               (~30-60 s)
make tables         # Tables 2-4 from results/ -> paper/tables/          (~5 s)
make figures        # Figure 2 from results/ -> paper/figures/           (~5 s)
make test           # leakage + every paper number + smoke               (~1 min)
```

Expected end of `make smoke`:

```
[smoke] dose-response + MVDF selection     ok
[smoke] Phase 3 battery (best R2 ...)      ok
[smoke] Phase 4 oracle (gap ..%)           ok
[smoke] learned scorer vs naive_matched    ok
[smoke] E3 cell (F0, F2) + leakage check   ok
[smoke] PASS in 32s
```

The smoke test checks that every stage *runs* on a deliberately tiny model; its numbers are not
the paper's. The paper's numbers are checked by `make test` against the committed results.

## Installation in detail

- **Python 3.11+** (results were produced with 3.12.11). `requirements.txt` pins the exact
  versions used: torch 2.6.0, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.5.2,
  statsmodels 0.14.0, matplotlib 3.10.3, pyarrow 21.0.0, PyYAML 6.0.2, pytest 8.3.4.
- `pip install -e .` installs the library as `mechconf` (from `src/mechconf/`). Experiment entry
  points live in `experiments/` and are run from the repository root with `python -m …`.
- **CPU is enough** for everything except re-training electricity/traffic in reasonable time
  (those used Apple MPS). For a CPU-only torch wheel: `pip install torch==2.6.0 --index-url
  https://download.pytorch.org/whl/cpu` before `make install` (this is what CI does).
- **Bit-exact reproduction of the ETTh models requires CPU with 4 torch threads**
  (`torch.set_num_threads(4)`, set by the scripts). A different thread count changes the order of
  floating-point reductions and therefore the trained weights.
- `mechconf.paths.REPO_ROOT` locates `data/`, `configs/` and `results/`; override with the
  environment variable `MECHCONF_ROOT` if you run from elsewhere.

## Data

| dataset | file | how to get it | SHA-256 |
|---|---|---|---|
| ETTh1 | `data/raw/ETTh1.csv` | `python scripts/download_data.py` (ETDataset GitHub) | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| ETTh2 | `data/raw/ETTh2.csv` | same | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| electricity | `data/raw/electricity.csv` | place by hand (370 clients × 35 000 hourly rows, preprocessed from UCI ElectricityLoadDiagrams20112014) | `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570` |
| traffic | `data/raw/traffic.csv` | place by hand (862 sensors × 17 544 hourly rows, Caltrans PeMS) | `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281` |

`scripts/download_data.py` downloads what it can, checks the SHA-256 of every file present, and
exits non-zero on a mismatch (`--require-all` also fails if electricity/traffic are missing). The
electricity and traffic files are the preprocessed versions from the Kaggle data bundle the
paper's models were trained on; no public URL serves them byte-identically.

**Splits.**
- Rows: ETTh train/validation/test = 8640 / 2880 / 5900; other datasets 70/10/20%.
- Windows: input 96 → horizon 96, stride 1.
- Test windows are split into thirds by time: E_select / E_eval_ID / E_eval_shift (ETTh: 1872 / 1966 / 1871).
- Predictability models: first 60% of E_eval_ID for fitting, last 40% for evaluation.
- Conformal methods: 50/30/20 of E_eval_ID (training / calibration / width evaluation).
- All variables are inputs; errors and intervals use one target variable (oil temperature for ETTh, the last column otherwise).

**Two copies of ETT.** The v2 Kaggle notebooks used the Kaggle bundle copy (edge fingerprints
`f56e…`/`1132…`); everything from Phase 3 on, including all camera-ready results, used the HTTP
mirror copy (`c6d1709fd0693b1a`/`c022db5521d23684`) — same split sizes, slightly different values.
Details: [`docs/PROVENANCE.md`](docs/PROVENANCE.md). Licences: [`DATA_LICENSES.md`](DATA_LICENSES.md).

## Large artifacts

Trained weights, per-window arrays and per-window parquet files (1.996 GB, 395 files) are not in
git. They are published as three release bundles, each file checked against
[`artifacts/MANIFEST.sha256`](artifacts/MANIFEST.sha256):

| bundle | contents | size | needed for |
|---|---|---|---|
| `artifacts_camera_ready.tar` | `results/camera_ready/**` weights (seeds 3–9), arrays, E3 per-window parquet, E6 TimeMixer arrays, G quantile forecasts | 0.62 GB | re-running E1–E6, F, G analyses on new seeds |
| `artifacts_round2_p126_round2.tar` | `results/round2/p126_round2/**` weights and arrays of seeds 0–2, synthetic dial runs | 1.22 GB | everything on seeds 0–2; round-2 analyses |
| `artifacts_round2_p126.tar` | `results/round2/p126/**` round-1 arrays | 0.16 GB | round-1 follow-up |

```bash
python scripts/fetch_artifacts.py                        # all bundles, extract, verify
python scripts/fetch_artifacts.py --bundles camera_ready
python scripts/fetch_artifacts.py --verify               # verify files already in place
```

**You do not need them** to regenerate the paper's tables, Figure 2 or to run the paper-number
tests — those use only the committed CSVs.

## The pipeline, stage by stage

```
 forecaster (iTransformer) ──► hidden state h_t ──► TopK SAE ──► features a_j(h_t)
                                                                    │
         ┌──────────────────────────────────────────────────────────┘
         ▼
 dose-response on E_select ──► P_j, U_j, R_j ──► MVDF (12 features), types A–D       [L2]
         │
         ├──► natural predictability on E_eval_ID: a_j(h_t) → S_t ?                   [L1]
         ├──► ex-ante baselines: past errors, input/forecast stats, dense state → S_t ?
         ├──► oracle ceiling: how much width could difficulty-awareness save?
         └──► downstream: continuous scorer, regime gate, localization, CQR, PID,
              Mondrian bands — vs naive_matched, at valid coverage
```

| stage | input → output | code | key settings |
|---|---|---|---|
| 1. Forecaster | windows → ŷ, hidden state | `src/mechconf/forecasters/itransformer.py`, `train_itransformer.py` | d_model 256, 3 layers, 8 heads, d_ff 512, 20 epochs, patience 5 |
| 2. SAE | pre-head target token → 2048 sparse codes | `src/mechconf/sae/` | TopK k = 64, 12 000 steps, dead-feature resampling |
| 3. Dose-response | 120 random eligible features × 6 doses → P_j, U_j | `src/mechconf/validation/dose_response.py` | active windows only; reconstruction error not added back |
| 4. Selection | P_j, U_j → R_j, MVDF, types | `src/mechconf/validation/difficulty_specificity.py` | isotonic residual, monotone-curve gate, top 12 |
| 5. Natural predictability | MVDF activations → S_t (held out) | `experiments/camera_ready/phase34.py` (from `experiments/v2/phase3_real_predictability.py`) | linear, isotonic, gradient boosting, pinball MLP; all windows and active windows |
| 6. Ex-ante baselines | 9 feature sets → S_t; Mondrian bands | `experiments/camera_ready/e3_exante.py`, `features.py` | + ridge and random forest; estimator by time-series CV on the fit split; leakage unit test |
| 7. Oracle | realised S_t deciles → LOO quantiles | `experiments/v2/phase4_oracle_ceiling.py`, `experiments/camera_ready/phase34.py` | α = 0.10, 10 deciles |
| 8. Downstream | features → intervals | `src/mechconf/calibration/`, `src/mechconf/baselines/`, `experiments/camera_ready/e2_scorer.py` | ACI γ = 0.03; CQR; PID-conformal |

Exact formulas with line references: [`docs/DEFINITIONS.md`](docs/DEFINITIONS.md).

## Results in detail

All numbers below are recomputed from committed files by `tests/test_paper_numbers.py`.

### 1. Intervention effects on difficulty are real at the population level (Table 2)

Share (%) of features whose intervention effect on S_t has a 95% interval excluding 0 on
**held-out** windows, median [min, max] over runs, then type-B enrichment of the MVDF set:

| | ETTh1 | ETTh2 | electricity | traffic |
|---|---|---|---|---|
| all pool features | 19 [13,23] | 22 [15,28] | 47 [34,53] | 17 [15,22] |
| MVDF features | 17 [8,50] | 25 [8,50] | 58 [17,67] | 8 [0,17] |
| enrichment (median) | 2.04× | 2.28× | 3.85× | 3.33× |
| runs above selection null | 0/10 | 2/10 | 4/5 | 3/5 |

The dose-response curves are clean (median linear-fit R² 0.999 on ETTh). The selection rule alone
produces ≈ 1.8× enrichment on permuted U_j; beyond that, enrichment holds when pooled over 30 runs
(Fisher p = 6×10⁻¹⁰) but not on ETTh1 (p = 0.62). Selected features also keep neither their
held-out advantage (15/30) nor their identity across SAE retraining (8–33% matched vs 28–43% for
arbitrary features). Hence the claim is about the population of SAE features, not individual ones.

### 2. Natural activation does not predict difficulty (Figure 2a, Table 3)

Best held-out R² (four estimators) from natural MVDF activation, median [min, max]:
ETTh1 −0.60 [−0.72, −0.52], ETTh2 −0.89 [−0.94, −0.77], electricity −0.29 [−0.33, −0.29],
traffic −0.70 [−1.33, −0.46]; the same holds on windows where an MVDF feature fires, and on
TimeMixer (−0.49 to −1.16). Is anything predictable at forecast time? Median Spearman ρ and runs
where both the R² and ρ intervals lie above zero:

| feature set | ETTh1 | ETTh2 |
|---|---|---|
| MVDF activations | −0.01 (0/10) | +0.00 (0/10) |
| best ex-ante set (ETTh1: past errors + input/forecast stats; ETTh2: input stats) | −0.02 (1/10) | +0.32 (6/10) |
| dense hidden state | −0.01 (0/10) | +0.38 (10/10) |
| full SAE code | −0.02 (0/10) | +0.26 (7/10) |

On ETTh2 a weak rank signal exists in the representation, but not in the causally selected
features; adding MVDF to all ex-ante features does not improve held-out R² on 29 of 30 runs.

### 3. Positive control

On a regime-switching synthetic process with a built-in difficulty signal, the unchanged battery
finds held-out R² 0.26–0.48 (links: causal sign 3/3, enrichment 3/3, predictability 3/3,
downstream gain 1/3), so the real-data null is a property of the representation, not of the
measurement.

### 4. Difficulty varies, but no method recovers it (Figure 2b, Table 4)

- **Oracle width reduction, in-distribution** (median [min, max]): ETTh1 25.2% [23.6, 25.9],
  ETTh2 30.0% [28.2, 30.6], electricity 18.1% [17.3, 18.3], traffic 39.1% [39.0, 39.6].
  Under shift the medians are 24.0–37.3%.
- **Ex-ante Mondrian bands:** under shift every valid ETTh band is *wider* than the constant band
  (GRR −0.20 to −2.72).
- **Continuous scorer:** the learned scorer minus naive_matched width under shift, with the number
  of runs where naive_matched is significantly narrower in brackets:

| input | ETTh1 | ETTh2 | electricity | traffic |
|---|---|---|---|---|
| dense hidden state | +0.14 (4/10) | +0.38 (10/10) | +0.10 (0/5) | +3.49 (5/5) |
| MVDF | +0.06 (0/10) | +0.15 (10/10) | −0.41* (0/5) | +2.56 (5/5) |

\* narrower only because it under-covers (0.62–0.85); naive_matched covers 0.89–0.92.

- **Regime gating** passes on none of 20 runs.
- **Localization:** a time-only neighbour control explains most of the apparent feature-space
  structure.
- **PID-conformal** ties naive_matched on average.
- **CQR** under- or over-covers.

### 5. Robustness

| check | result |
|---|---|
| Alternative specificity scores (binned mean, spline, rank residual) | the null holds under every definition |
| Interventions within the natural range (λ ≤ 1) | reselect largely the same features (Jaccard 0.85) |
| Setting features to values they take on other inputs | moves S_t in the predicted direction for 81% of features |
| A second architecture (TimeMixer) | replicates the null |

## Paper ↔ code map

Commands run from the repository root. Camera-ready scripts take the frozen settings from
`experiments/v2/phase3_real_predictability.py::CFG` (the paper's configuration) and are driven by
the command-line arguments shown; `scripts/reproduce_all.sh` lists every stage with its exact
arguments. ✓ = recomputed by `tests/test_paper_numbers.py` from the listed file
(`paper/paper_numbers.yaml`).

| paper item | what it shows | script + command | output file(s) | runs |
|---|---|---|---|---|
| Table 1 (notation) | evidence level of P_j, U_j, R_j, a_j → S_t | definitions in `docs/DEFINITIONS.md` | — | — |
| Figure 1 (pipeline) | stages and data flow | — (drawn by hand; see the diagram above) | — | — |
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
| §5.5 natural range "8%", "0.85", "81%" ✓ | in-range and interchange interventions | `python -m experiments.camera_ready.e5_dose a\|c …; …report_e5` | `results/camera_ready/E5/AGG_E5a_*.csv`, `AGG_E5b_natural_range.csv`, `AGG_E5c_interchange.csv` | 20 ETTh |
| §5.5 TimeMixer "0.0008–0.0027", "1.2–2.5×, 6/6", "6/6", "24–30%", "−0.49 to −1.17"‡ ✓ | second architecture | `python -m experiments.camera_ready.e6_timemixer …`; `experiments/v2/arch_gen_timemixer.py` | `results/camera_ready/E6/AGG_phase3.csv`, `AGG_phase4.csv`, `AGG_E6_predictability.csv`; `results/v2/arch_gen_timemixer/AGG_arch_gen.csv` | 6 |
| §5.5 SAE reproducibility "8–33% (mean 25%)", "28–43%" ✓ | feature identity across SAE seeds | `python -m experiments.v2.phase5_sae_reproducibility` | `results/v2/phase5_sae_repro/AGG_phase5.csv` | 4 comparisons |
| §5.5 necessity "−0.0004 to −0.0021 on 5 of 6" ✓ | sufficiency, not necessity | `python -m experiments.v2.phase2_necessity_noising` | `results/v2/phase2_necessity/AGG_phase2.csv` | 6 |
| §6 static calibration "−0.09 to +0.004" ✓ | scorer vs naive without ACI feedback | `python -m experiments.camera_ready.g_analysis g5` (reads round 2) | `results/camera_ready/G/AGG_G5_static_shift.csv` | 6 |

‡ The printed value differs from the committed result: TimeMixer −1.17 should be −1.16;
enrichment 2.0–3.9× should be 2.0–3.8×; correlated localization "2/6 below the bar" is 3/6 below
0.80 (2/6 pass the full rule). See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) §A. These three
are pinned as strict expected failures in the tests.

## Running the experiments

Everything runs from the repository root. The camera-ready drivers share one loader
(`experiments/camera_ready/common.py`) that reads per-window arrays and metadata for a
(dataset, seed) from `results/camera_ready/arrays/` or `results/round2/p126_round2/arrays/`.

**Train a new seed** (forecaster → SAE → MVDF selection; ETTh on CPU for bit-exactness):

```bash
python -m experiments.camera_ready.train_seeds --datasets ETTh1 --seeds 10 --device cpu
python -m experiments.camera_ready.train_seeds --datasets electricity --seeds 5 --device mps
```

Outputs: `results/camera_ready/weights/<ds>_seed<s>_{forecaster,sae}.pt`,
`results/camera_ready/arrays/<ds>_seed<s>.npz`, `results/camera_ready/E2/meta/<ds>_seed<s>_{meta.json,candidates.csv}`.

**Analyse runs** (each takes `--datasets` and `--seeds`):

```bash
python -m experiments.camera_ready.phase34   --datasets ETTh1 --seeds 0 1 2 --out /tmp/p34   # Phase 3 + 4
python -m experiments.camera_ready.e3_exante --datasets ETTh1 --seeds 0 --out /tmp/e3 --sets F0 F2 F4
python -m experiments.camera_ready.e5_dose a --datasets ETTh1 --seeds 0                       # dose-response diagnostics
python -m experiments.camera_ready.e2_scorer --datasets ETTh1 --seeds 0 --out /tmp/scorer.csv
python -m experiments.camera_ready.test_leakage ETTh1 0                                       # leakage rule
```

**Analyse a different set of arrays** (e.g. TimeMixer or the smoke test): set
`CR_ARR_DIR=<dir>` to a folder holding `<ds>_seed<s>.npz` and `<ds>_seed<s>_meta.json`.

**Full re-run:** `bash scripts/reproduce_all.sh all` or one stage at a time (`make reproduce-<stage>`):

| stage | what | measured time |
|---|---|---|
| `train` | seeds 3–9 (ETTh, CPU) and 3–4 (electricity/traffic, MPS) | ~3.5 h |
| `arrays` | traffic seed 0–2 arrays from saved weights | ~1 min |
| `phase34` | Phase 3 battery + Phase 4 oracle from arrays | ~1 min |
| `scorer` | learned scorer vs naive_matched | ~1 min |
| `e3` | ex-ante baselines + gap recovery, 30 cells | ~3 h |
| `e5`, `e4` | dose-response diagnostics; R_j robustness | ~5 min each |
| `e6` | TimeMixer regeneration + analysis | ~1.5 h |
| `f`, `g` | follow-up F; G2/G4/G5 | ~20 min |
| `reports` | E1/E2 reports | ~1 min |

Hardware used: Apple Silicon (arm64, 10 cores, 16 GB), no CUDA. Earlier stages (v1 MVE, v2
phases 0–2, H6) ran in Kaggle notebooks (`notebooks/`) against the pre-restructure layout; they are
kept for the record and their outputs are committed under `results/v1/` and `results/v2/`.

## Tests

| test | what it checks | needs |
|---|---|---|
| `tests/test_paper_numbers.py::test_paper_number` (120 cases) | every number printed in the paper equals the value recomputed from `results/` (`paper/paper_numbers.yaml`); known mismatches are strict expected failures, so a silent fix or regression fails CI | committed CSVs only |
| `tests/test_paper_numbers.py::test_config_constants` | the paper's configuration (d_model 256, 3 layers, 96→96, 2048 / 64, λ grid, pool 120, 12 features, α 0.10, block 96) matches the code | — |
| `tests/test_leakage.py` | no ex-ante feature for window t uses a target with timestamp > t (synthetic data always; a deliberately leaky feature must be caught; real arrays when present) | release assets for the real-data case |
| `tests/test_smoke.py` | the end-to-end smoke run passes | `data/raw/ETTh1.csv` |

CI (`.github/workflows/ci.yml`) installs CPU torch, lints for syntax errors and undefined names,
downloads ETTh1/ETTh2 and runs the full test suite. It is **manual-only** (Actions tab → "ci" →
"Run workflow"), so it never marks a commit as failed when Actions is unavailable. The same
checks, run from a fresh clone, are recorded in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## What is in `results/`

| folder | contents |
|---|---|
| `results/camera_ready/E0` | Phase-5 report fix and diff; `DEFINITIONS.md` (original line references) |
| `results/camera_ready/E1` | electricity/traffic: Phase 3/4 (`AGG_phase3.csv`, `AGG_phase4.csv`), E3 tables (`AGG_E1_E3_*`), per-dataset folders |
| `results/camera_ready/E2` | 30-run core metrics (`AGG_E2_phase3/phase4/enrichment/scorer_vs_naive/health.csv`), `SEED_TABLE.md`, per-seed `meta/` |
| `results/camera_ready/E3` | ex-ante predictability (`AGG_E3_predictability.csv`: dataset, seed, split, feature_set, estimator, n_fit, n_eval, R2 [CI], rho [CI]), gap recovery, ΔR², per-cell outputs |
| `results/camera_ready/E4` | R_j robustness under six residualisers |
| `results/camera_ready/E5` | per-feature dose-response diagnostics, natural-range reselection, interchange, histogram |
| `results/camera_ready/E6` | TimeMixer Phase 3/4 and ex-ante subset |
| `results/camera_ready/F` | regime gating (10 seeds), pooled enrichment vs selection null, shift gap recovery |
| `results/camera_ready/G` | G2 pre-evaluation profile, G4 held-out L2 evidence, G5 static check; `STATUS.md` (G1, G3 partial) |
| `results/camera_ready/CAMERA_READY_SUMMARY.md` | every pre-registered criterion with PASS/FAIL, "for the paper" text, numbers that changed |
| `results/v2/` | submission-era outputs: phases 0–5, H7-A, SLCP, baselines, TimeMixer, synthetic control, Kaggle v2 runs, legacy figures |
| `results/round2/` | follow-up rounds 1 (`p126/`) and 2 (`p126_round2/`) |
| `results/v1/` | v1 MVE (Run 5) outputs |

Every `RESULT.md` starts with provenance (git hash, command, seeds, data fingerprints, hardware,
wall-clock), then the pre-registered criteria, results, a verdict per criterion, a plain-English
summary, anomalies, and a "For the paper" section.

## Pre-registration

Every camera-ready experiment's pass/fail criteria were committed to `PREREG.md` **before** its first
run; [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) lists each commit and its time, and the
changes made afterwards (all disclosed). Some pre-registered criteria **failed**, and they are
reported as failures:

- **C3.1**: difficulty is not ex-ante predictable under the pre-registered rule.
- **C3.2**: the dissociation criterion (requires C3.1).
- **C3.4**: no recoverable opportunity.
- **C4.2**: enrichment is not robust to the residualiser.
- **C5b**: natural-range enrichment fails on one boundary cell.
- **C5c**: interchange sign agreement fails on ETTh1.
- **C2.1** (partly): per-run significance of the scorer gap.

## Known issues and limitations

Full list with effects on the paper: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). In brief:

- **Three printed numbers differ from the results** (marked ‡ above).
- **Block-local profile:** S_t is normalised by the scored block's own error profile. A
  pre-evaluation profile changes no conclusion (rank agreement 0.95–0.997).
- **Interventions drop the SAE reconstruction error**, also at λ = 1.
- **Immediate-feedback ACI:** naive_matched and the learned scorers share the same ACI correction,
  which uses outcomes not yet observable at a 96-step horizon. The comparison is like-for-like,
  and a static-calibration check agrees.
- **Traffic:** in-distribution coverage saturates at 1.000 because of an outlier stretch in the
  calibration data.
- **ETTh in-distribution blocks** under-cover for every method, so the shift block is the primary
  comparison.
- **Single-step activations only:** predictors use activation at one time step. The paper
  establishes an instantaneous dissociation, not that no temporal model could predict difficulty.
- **Two earlier-stage bugs** found during the camera-ready work: an invalid Phase-5
  "shuffle floor", and H7-A reading a different SAE's feature list.
- **Round G** was stopped early; G1 and G3 are partial.

## Project history

| period | stage |
|---|---|
| to early Sep 2026 | v1 MVE: synthetic control, five Kaggle runs, a calibration bug found and fixed (Run 5 is the valid result) |
| Sep 3–4 | v2 phases 0–2: layer sweep, MVDF with R_j (12/12 type-B enrichment), invariance, downstream Task G |
| Sep 4–9 | gates and phases 3–5: regime gating, localization, CQR/PID, synthetic control, natural predictability, oracle, SAE reproducibility, necessity, TimeMixer |
| Sep 2026 | submitted as a 12-page regular paper; accepted as an 8-page short paper |
| Nov 10–13 | follow-up rounds 1–2 (corrected evaluation path, saved weights and arrays) |
| Nov 14–16 | camera-ready experiments E0–E6, F, G (pre-registered) |
| Nov 16 | this public release |

Details: [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md). The original task specifications
are in [`docs/reports/`](docs/reports/); what changed for the camera-ready is in
[`docs/CAMERA_READY_CHANGES.md`](docs/CAMERA_READY_CHANGES.md).

## Repository layout

```
src/mechconf/             importable library (pip install -e .)
  data/                   loaders, windows, test-split carve-out, fingerprints, synthetic DGP
  forecasters/            iTransformer, TimeMixer, training loops
  sae/                    TopK SAE and training
  representation/         hidden-state taps
  validation/             errors and S_t, correlation screen, dose-response, specificity, selection
  calibration/            ACI, Mondrian, learned difficulty scorer
  baselines/              CQR (quantile iTransformer), PID-conformal
  eval/                   metrics, consolidation
experiments/              entry points (python -m experiments.<stage>.<script>)
  v1_mve/                 v1 minimum viable experiment
  v2/                     submission-era phases, gates, baselines, TimeMixer, synthetic control
  round2/                 follow-up rounds 1 and 2
  camera_ready/           E0–E6, F, G drivers, feature builder, leakage test, report generators
results/                  committed small outputs (CSV, JSON, Markdown, PNG)
paper/                    make_fig2.py, make_tables.py, paper_numbers.{py,yaml}, figures/, tables/
scripts/                  download_data.py, fetch_artifacts.py, smoke_test.py, reproduce_all.sh, dev/
tests/                    test_paper_numbers.py, test_leakage.py, test_smoke.py
docs/                     history, pre-registration, definitions, provenance, known issues, changes, reports/
artifacts/                MANIFEST.sha256 for the release assets
configs/                  YAML configs of the v2 Kaggle stages, data fingerprints, synthetic DGP
notebooks/                Kaggle notebooks used for v1 and v2 (pre-restructure layout)
kaggle_ext/               runbook for the Kaggle extension to electricity/traffic
```

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `FileNotFoundError: data/raw/...csv` | run `python scripts/download_data.py`; place electricity/traffic by hand |
| `[MISMATCH] ... sha256` | a different copy of the dataset; results will not match the paper |
| `FileNotFoundError: results/.../arrays/<tag>.npz` | large artifacts not downloaded: `python scripts/fetch_artifacts.py` |
| retrained ETTh model differs from the committed one | run on CPU with 4 torch threads (the scripts set this); other thread counts change reduction order |
| electricity/traffic retrain differs slightly | trained on Apple MPS; MPS kernels are not guaranteed bit-reproducible across machines |
| `ModuleNotFoundError: paper` / `experiments` | run from the repository root with `python -m …` |
| a paper-number test fails after editing a result | the YAML pins printed values; update `paper/paper_numbers.yaml` only if the paper text changes too |

## Citation and licence

```bibtex
@inproceedings{aadithiyan2027causally,
  title     = {Causally Real, Predictively Elusive: Mechanistic Difficulty Representations in Neural Forecasters},
  author    = {Aadithiyan, Donald},
  booktitle = {Proceedings of the International Conference on Agents and Artificial Intelligence (ICAART 2027)},
  publisher = {SCITEPRESS},
  year      = {2027}
}
```

Code: MIT ([`LICENSE`](LICENSE)). Data: not redistributed; see [`DATA_LICENSES.md`](DATA_LICENSES.md).
Citation metadata: [`CITATION.cff`](CITATION.cff).
