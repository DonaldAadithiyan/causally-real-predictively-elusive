# Extending Sections 5.3–5.11 to electricity + traffic — Kaggle GPU runbook

**Goal:** lift the ETTh1/ETTh2-only scope limit by running the extended-wrap-up
experiments on the two remaining datasets, under the *identical* protocol, splits,
and seeds already documented for ETTh1/ETTh2.

**Why this is now possible:** the raw electricity/traffic data — believed lost — was
found in `~/Downloads/results (3)/mechanistic_conformal/data/raw/`. It is the
adapter-materialised cache from the original Kaggle run and reproduces the Section
5.1/5.2 fingerprints **bit-for-bit** (`electricity` edge-hash `cc6af3b5973c14af`,
`traffic` `a93d2ec2b5815355`). So it is the byte-exact source, not the mismatched
LTSF-standard substitute the paper's limitation warns against.

---

## What actually needs to run — 5 jobs, not 6

| § | script | status | action |
|---|---|---|---|
| 5.3 Phase 3 dissociation | `phase3_real_predictability.py` | ETTh1/2 only | **run** |
| 5.5 Oracle ceiling | `phase4_oracle_ceiling.py` | ETTh1/2 only | **run** |
| 5.6 Task G / H6 | `phase3_scorer_h6.py` → `v2_h6_topup/` | **already has all 4 datasets** (96 rows in `AGG_h6.csv`, confirmed in `RESULTS_FINAL.md`) | nothing — or `--with-taskg` to regenerate |
| 5.9 CQR + PID baseline | `baseline_comparison.py` | ETTh1/2 only | **run** |
| 5.10 Arch-gen (TimeMixer) | `arch_gen_timemixer.py` | ETTh1/2 only | **run** |
| 5.11 SAE reproducibility | `phase5_sae_reproducibility.py` | ETTh1/2 only | **run** |
| 5.4 synthetic control | — | dataset-independent | nothing |

Jobs 5.3 / 5.5 / 5.10 / 5.11 each run their **own** Phase 1 internally (train
iTransformer → SAE → dose-response → `select_mvdf_v2`) — fully self-contained, so
`--datasets electricity traffic` just works. 5.9 uses no SAE at all.

This matches exactly how ETTh1/ETTh2 appear in these sections: per the Phase 3 header,
those were **retrained locally** ("Phase 1 retrained locally for self-consistent MVDF
selections", different edge fingerprint from the 5.1 Kaggle run). Adding
electricity/traffic under the same local-retrain protocol is self-consistent with the
rest of 5.3–5.11.

---

## Setup on Kaggle

### 1. Upload the repo + raw data as a Kaggle Dataset

From your Mac:

```bash
cd "/Users/.../RecessionRadar/mechanistic-conformal"
mkdir -p /tmp/mc_kaggle && rsync -a --exclude '__pycache__' --exclude '.git' \
  --exclude 'mve_runs' --exclude 'notebooks' \
  ./ /tmp/mc_kaggle/mechanistic-conformal/
# add the recovered raw files
cp "/Users/.../Downloads/results (3)/mechanistic_conformal/data/raw/electricity.csv" \
   "/Users/.../Downloads/results (3)/mechanistic_conformal/data/raw/traffic.csv" \
   /tmp/mc_kaggle/mechanistic-conformal/data/raw/
# add this package
cp kaggle_pkg/* /tmp/mc_kaggle/mechanistic-conformal/
```

Zip `/tmp/mc_kaggle/` and create a Kaggle Dataset from it (e.g. `mc-icaart-ext`).
`electricity.csv` is 110 MB and `traffic.csv` 104 MB — well within Kaggle's limits.

### 2. New Kaggle Notebook, GPU on (T4 ×2 or P100), internet **off**

Add the dataset. Then:

```python
import os, shutil, subprocess, sys

SRC = "/kaggle/input/mc-icaart-ext/mechanistic-conformal"
WORK = "/kaggle/working/mechanistic-conformal"
shutil.copytree(SRC, WORK)          # writable copy
os.chdir(WORK)

# pin the fingerprint manifest so strict_fingerprint doesn't need to write
shutil.copy("data_fingerprints_ext.json", "configs/data_fingerprints.json")

# keep a pristine copy of experiments/ so the GPU patch is reversible
shutil.copytree("experiments", "/kaggle/working/experiments_orig")

# GPU patch (device strings only — no scientific-path edits)
os.environ["MC_DEVICE"] = "cuda"
print(subprocess.run(["bash", "patch_gpu.sh"], capture_output=True, text=True).stdout)

# sanity: data loads + fingerprints match 5.1/5.2
import importlib; sys.path.insert(0, WORK)
from data.datasets import get_splits
from data.fingerprint import data_fingerprint
for ds, want in [("electricity","cc6af3b5973c14af"), ("traffic","a93d2ec2b5815355")]:
    fp = data_fingerprint(ds, get_splits(ds, max_channels=48))
    assert fp["edge_hash"] == want, (ds, fp["edge_hash"], want)
    print(f"{ds}: fingerprint OK ({fp['edge_hash']}, {fp['rows']} rows, {fp['channels']} ch)")

import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### 3. Run

```python
!MC_DEVICE=cuda python run_extension.py --all
```

or one job at a time (safer against the 9 h / 12 h Kaggle wall — checkpoint by
committing between runs):

```python
!MC_DEVICE=cuda python run_extension.py --only phase3
!MC_DEVICE=cuda python run_extension.py --only oracle
!MC_DEVICE=cuda python run_extension.py --only baseline
!MC_DEVICE=cuda python run_extension.py --only archgen
!MC_DEVICE=cuda python run_extension.py --only phase5
```

Rough GPU wall-clock (T4): phase3 ~35 min, oracle ~25 min, baseline ~50 min
(trains a quantile iTransformer too), archgen ~30 min, phase5 ~45 min → **~3.5 h
total** for electricity + traffic × 3 seeds. Comfortably one session; still,
`--only` + commit-per-job is the robust path.

### 4. Collect outputs

Each job appends the two datasets to the SAME dirs and regenerates its `*_RESULT.md`:

```
runs/phase3_real/        AGG_phase3.csv   PHASE3_RESULT.md      (now 12 rows)
runs/phase4_oracle/      AGG_oracle.csv   PHASE4_RESULT.md      (now 12 rows)
runs/baseline_comparison/AGG_baselines.csv AGG_baselines_tests.csv
runs/arch_gen_timemixer/ AGG_arch_gen.csv ARCH_GEN_RESULT.md    (now 12 rows)
runs/phase5_sae_repro/   AGG_phase5.csv   PHASE5_RESULT.md      (now 4 datasets)
```

Zip `runs/` from `/kaggle/working/` as the notebook output. Back on the Mac, drop
those over the local `runs/` dirs and regenerate the consolidated tables:

```bash
python eval/consolidate_results.py       # RESULTS_FINAL.md, picks up the new rows
```

---

## Reading the new numbers into the paper

- **If electricity + traffic land in the same direction as ETTh1/ETTh2** (expected —
  they did for Task G, which already has all 4): fold them into the existing tables
  in each section, and **rewrite Section 4.1 + Section 7 Limitations**: the scope
  limit is lifted for 5.3–5.11, keep only whatever residual caveat the numbers
  actually justify.
- **If either dataset diverges:** keep it as a reported result with the divergence
  called out — that is itself informative (e.g. traffic's known ID-coverage anomaly,
  RESULTS_FINAL.md Task A). Do **not** suppress it.
- Either way the current wording ("raw files unavailable") must change — that is now
  factually wrong.

---

## Undo the GPU patch (if running locally again later)

```bash
rm -rf experiments && mv /kaggle/working/experiments_orig experiments
git checkout configs/v2_phase3_h6.yaml configs/data_fingerprints.json   # if git
```

Files in this package:

| file | purpose |
|---|---|
| `KAGGLE_README.md` | this file |
| `patch_gpu.sh` | rewrites the hard-coded `"cpu"` device strings → `${MC_DEVICE:-cuda}` |
| `run_extension.py` | driver — runs the 5 jobs on electricity + traffic, seeds 0/1/2 |
| `data_fingerprints_ext.json` | fingerprint manifest incl. electricity/traffic (pin over `configs/data_fingerprints.json`) |
