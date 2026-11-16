# Provenance: data, splits, hardware, determinism

## Datasets and the two data copies

| dataset | raw file (data/raw/) | SHA-256 of the raw file | edge fingerprint | source |
|---|---|---|---|---|
| ETTh1 | `ETTh1.csv` | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` | `c6d1709fd0693b1a` | ETDataset GitHub (HTTP mirror); `scripts/download_data.py` fetches it byte-identically |
| ETTh2 | `ETTh2.csv` | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` | `c022db5521d23684` | same |
| electricity | `electricity.csv` | `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570` | `cc6af3b5973c14af` | preprocessed file of the Kaggle data bundle (370 clients × 35 000 hourly rows, derived from UCI ElectricityLoadDiagrams20112014). Place manually |
| traffic | `traffic.csv` | `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281` | `a93d2ec2b5815355` | preprocessed file of the Kaggle data bundle (862 sensors × 17 544 hourly rows, Caltrans PeMS). Place manually |

"Edge fingerprint" is the hash of the split edges the code logs (`mechconf.data.fingerprint`). The
raw-file SHA-256 is what `scripts/download_data.py` checks.

**Two copies of ETT exist in this project:**
- The **Kaggle bundle copy** (edge fingerprints `f56e…` / `1132…`) was used by the v2 Kaggle notebooks
  (`results/v2/v2_runs/`, `v2_confirm/`, `v2_h6_topup/`). These produced the submitted paper's H1c
  (12/12) and Task-G / "earlier 12 runs" scorer results, which the camera-ready still quotes
  (§5.4, widths 3.89/3.94/4.36 vs 3.35).
- The **HTTP mirror copy** (`c6d1…` / `c022…`) was used locally from Phase 3 onward, and by every
  camera-ready iTransformer and TimeMixer result. It has the same split sizes as the Kaggle copy but
  slightly different edge values.

The electricity/traffic local files are byte-identical to the Kaggle bundle's.
`kaggle_ext/KAGGLE_README.md` documents how they were recovered.

## Splits

- Rows: ETTh train [0, 8640), validation [8640, 11520), test [11520, 17420). Other datasets: 70/10/20%.
- Windows: input 96, horizon 96, stride 1; window index = forecast origin.
- Test windows are split into thirds by window end index: E_select (selection) / E_eval_ID / E_eval_shift.
  ETTh: 1872 / 1966 / 1871 windows.
- Phase-3 predictability: first 60% of E_eval_ID for fitting, last 40% for evaluation (ETTh 1179 / 787).
- Conformal methods: 50/30/20 of E_eval_ID (scorer training / calibration / width evaluation;
  ETTh 983 / 589 / 394). The shift block is scored with the same calibration.
- Exact definitions with code references: `docs/DEFINITIONS.md`.

## Seeds and determinism

- Seed per run: `experiments._seed.derive_seed(seed, dataset)` seeds the forecaster, SAE, candidate
  pool draw and dose windows. Bootstrap RNG seed 0. CQR quantile model seed `derive_seed + 31337`.
- ETTh runs: CPU with `torch.set_num_threads(4)`. Under this setting retrains are **bit-exact**:
  the ETTh1 seed 0 retrain through the camera-ready driver reproduced the paper's forecaster, SAE,
  MVDF set and every saved array exactly. A first attempt with 3 threads did not reproduce,
  because thread count changes reduction order.
- Electricity/traffic forecasters and SAEs: Apple MPS (not guaranteed bit-reproducible across machines).
- TimeMixer: CPU, 4 threads; all 6 runs reproduce the stored val MSE, SAE FVU and MVDF set exactly.
- Everything after training (dose-response, batteries, conformal) runs on CPU from saved arrays.

## Hardware and measured wall-clock

Apple Silicon (arm64, 10 cores, 16 GB), macOS, no CUDA GPU. Python 3.12.11, torch 2.6.0,
numpy 2.4.4, scikit-learn 1.5.2, pandas 3.0.2, scipy 1.17.1.

| stage | wall-clock (measured) |
|---|---|
| ETTh seed retrain (forecaster + SAE + selection), CPU | 6–15 min per seed (contended) |
| electricity / traffic seed retrain, MPS | 18–21 / 12 min per seed |
| traffic arrays from saved weights | ~1 min for 3 seeds |
| Phase 3 + Phase 4 from arrays, 20 runs | ~1 min |
| E3 ex-ante cell (9 feature sets × 7 estimators + CV + gap recovery) | 5–8 min per cell |
| E5a dose-response diagnostics | ~9 s per cell |
| E4 R_j robustness, all cells | ~4 min |
| TimeMixer regeneration | ~8–14 min per run |
| F follow-up | ~4 min |
| G4 (direct L2 evidence, 30 runs) | ~15 min |
| smoke test (`make smoke`) | ~30 s |

Per-seed training times are logged in `results/camera_ready/E2/meta/*_meta.json` (`wall_clock_s`).

## Large artifacts

Forecaster/SAE/quantile weights, per-window arrays (`*.npz`) and per-window parquet files are not
in git. They are published as release assets with SHA-256 checksums in `artifacts/MANIFEST.sha256`.
Place them at the paths listed there (relative to the repository root).
