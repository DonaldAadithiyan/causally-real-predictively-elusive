"""Generate a self-contained Kaggle/Colab notebook that vendors the whole
`mechanistic-conformal` package and runs the MVE (or the full suite) on GPU.

Usage:
    python notebooks/build_kaggle_notebook.py --out notebooks/mve_kaggle.ipynb

The notebook:
  1. writes every package file into /kaggle/working/mechanistic_conformal/
  2. pip-installs deps not already on the Kaggle image
  3. runs experiments.mve_real for ETTh1 + ETTh2 (device auto -> cuda)
  4. displays the RQ1 / H2 / H3 tables and the plots inline
  5. zips runs/ for download

No internet is needed for code (vendored); ETT data is downloaded from the public
mirror in cell 3 (enable "Internet" in the Kaggle notebook settings), or upload the
ETT CSVs as a Kaggle Dataset and set data/raw/ paths.
"""
from __future__ import annotations

import argparse
import json
import os

PKG_FILES = [
    "__init__.py",
    "data/__init__.py", "data/datasets.py", "data/probe_inputs.py",
    "forecasters/__init__.py", "forecasters/itransformer.py",
    "forecasters/train_itransformer.py", "forecasters/synthetic_forecaster.py",
    "forecasters/train_forecaster.py",
    "sae/__init__.py", "sae/topk_sae.py", "sae/train_sae.py",
    "validation/__init__.py", "validation/errors.py", "validation/correlation_screen.py",
    "validation/ablation.py", "validation/rq1.py",
    "calibration/__init__.py", "calibration/aci.py", "calibration/mondrian_perbin.py",
    "calibration/ffdci_head.py",
    "shift/__init__.py", "shift/shift_windows.py",
    "eval/__init__.py", "eval/metrics.py", "eval/plots.py",
    "experiments/__init__.py", "experiments/mve_real.py", "experiments/full_suite.py",
    "configs/mve_etth1.yaml",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build(full_suite: bool, datasets=None, seeds=None):
    cells = []
    cells.append(md(
        "# Mechanistically Validated Difficulty Features — MVE\n\n"
        "Self-contained. Vendors the `mechanistic-conformal` package, loads the LTSF "
        "benchmark data, trains iTransformer + TopK SAE, runs correlation screening + "
        "causal ablation, builds the three matched-budget feature sets (SAE-random / "
        "SAE-correlated / MVDF), calibrates with Variant B (Mondrian + per-bin ACI) and "
        "Variant A (opaque MLP), and reports the RQ1 diagnostic, the first-order "
        "`corr(hidden, |e|)` screen, and the H2 / H3 tables per dataset.\n\n"
        "**Kaggle setup:**\n"
        "1. Sidebar → Accelerator → **GPU** (T4 or P100).\n"
        "2. Sidebar → Internet → **On** (for the ETT download).\n"
        "3. Sidebar → **Add Input** → attach the multivariate-TS bundle you have "
        "(the one with `archive/ETTh.csv`, `ETTm.csv`, `LD2011_2014.csv`, `PeMS.csv`, "
        "`mpi_roof.csv`, `solar_Alabama.csv`, `exchange.csv`, `QPS.csv`). The adapters "
        "in `data/datasets.py` know these filenames/formats and normalize each to the "
        "standard layout (raw electricity/weather are resampled to hourly). Any "
        "already-standard `traffic.csv` / `electricity.csv` / `weather.csv` is used "
        "directly if present. File discovery walks all of `/kaggle/input` recursively.\n"
    ))

    # cell: write package files
    payload = {}
    for rel in PKG_FILES:
        with open(os.path.join(ROOT, rel), "r") as f:
            payload[rel] = f.read()
    writer = (
        "import os, base64, json\n"
        "PKG = 'mechanistic_conformal'\n"
        "FILES = json.loads(base64.b64decode('%s').decode())\n"
        "for rel, content in FILES.items():\n"
        "    path = os.path.join(PKG, rel)\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "    open(path, 'w').write(content)\n"
        "print('wrote', len(FILES), 'files into', PKG)\n"
        % __import__("base64").b64encode(json.dumps(payload).encode()).decode()
    )
    cells.append(md("## 1. Vendor the package"))
    cells.append(code(writer))

    cells.append(md("## 2. Dependencies"))
    cells.append(code(
        "import importlib, subprocess, sys\n"
        "for pkg in ['pyyaml', 'scipy', 'scikit-learn']:\n"
        "    name = {'pyyaml':'yaml','scikit-learn':'sklearn'}.get(pkg, pkg)\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)\n"
        "import torch\n"
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
    ))

    cells.append(md("## 2b. Probe the attached inputs\n\n"
                    "Confirms which datasets the adapters can build before the long run."))
    cells.append(code(
        "import sys, os\n"
        "sys.path.insert(0, 'mechanistic_conformal')\n"
        "os.chdir('mechanistic_conformal')\n"
        "from data.probe_inputs import probe\n"
        "probe()\n"
        "os.chdir('..')\n"
    ))

    if seeds is None:
        seeds = [0]
    run_targets = (
        "['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'traffic', 'electricity', 'weather', 'solar', 'exchange']"
        if full_suite else str(list(datasets or ["ETTh1", "ETTh2", "traffic", "electricity"]))
    )
    seed_note = (
        f"Each dataset is run for seeds {seeds} and results aggregated (mean +/- sd across "
        "seeds) so H3 effects can be told from single-seed noise."
        if len(seeds) > 1 else "Single seed."
    )
    cells.append(md(
        "## 3. Configure & run\n\n"
        f"`device: cuda` if a GPU is attached. {seed_note} Full training budget "
        "(20-epoch early-stopped iTransformer, 20k SAE steps, 400-feature ablation "
        "sweep on up to 2048 windows).\n\n"
        "Datasets whose CSVs are not found under `/kaggle/input` (ETT excepted — HTTP) "
        "print `SKIP` and are passed over."
    ))
    cells.append(code(
        "import os, sys, yaml, torch\n"
        "if 'mechanistic_conformal' not in sys.path: sys.path.insert(0, 'mechanistic_conformal')\n"
        "if os.path.basename(os.getcwd()) != 'mechanistic_conformal': os.chdir('mechanistic_conformal')\n"
        "from data.datasets import download\n"
        "\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "base = yaml.safe_load(open('configs/mve_etth1.yaml'))\n"
        "base['device'] = dev\n"
        "\n"
        f"targets = {run_targets}\n"
        f"seeds = {list(seeds)}\n"
        "available = []\n"
        "for ds in targets:\n"
        "    try:\n"
        "        download(ds); available.append(ds)\n"
        "    except Exception as e:\n"
        "        print(f'SKIP {ds}: {e}')\n"
        "print('will run:', available, 'x seeds', seeds)\n"
        "\n"
        "import runpy\n"
        "for ds in available:\n"
        "    for sd in seeds:\n"
        "        cfg = dict(base)\n"
        "        cfg['dataset'] = dict(base['dataset'], name=ds)\n"
        "        cfg['seed'] = sd\n"
        "        cfg['run_name'] = f'mve_{ds}_seed{sd}' if len(seeds) > 1 else f'mve_{ds}'\n"
        "        cpath = f'configs/_run_{ds}_s{sd}.yaml'\n"
        "        yaml.safe_dump(cfg, open(cpath, 'w'))\n"
        "        print('=' * 70, f'\\n {ds}  seed {sd}\\n', '=' * 70)\n"
        "        sys.argv = ['mve_real', '--config', cpath]\n"
        "        try:\n"
        "            runpy.run_module('experiments.mve_real', run_name='__main__')\n"
        "        except SystemExit:\n"
        "            pass\n"
        "        except Exception as e:\n"
        "            print(f'RUN FAILED {ds} seed {sd}: {type(e).__name__}: {e}')\n"
    ))

    cells.append(md("## 4. Show results + aggregate across seeds"))
    cells.append(code(
        "import json, glob, re, os\n"
        "import numpy as np, pandas as pd\n"
        "from IPython.display import display, Image\n"
        "\n"
        "rows = []\n"
        "for rd in sorted(glob.glob('runs/mve_*')):\n"
        "    try:\n"
        "        log = json.load(open(f'{rd}/run_log.json'))\n"
        "        rq1 = json.load(open(f'{rd}/rq1_diagnostic.json'))\n"
        "    except FileNotFoundError:\n"
        "        continue\n"
        "    m = re.match(r'runs/mve_(.+?)(?:_seed(\\d+))?$', rd)\n"
        "    ds, seed = m.group(1), int(m.group(2) or 0)\n"
        "    h3 = log.get('H3_delta_shift_correlated_minus_mvdf', {})\n"
        "    hd = log.get('hidden_difficulty_signal', {})\n"
        "    rows.append(dict(dataset=ds, seed=seed,\n"
        "        fc_val_mse=log.get('forecaster_val_mse'),\n"
        "        sae_fvu=(log.get('sae_reconstruction') or {}).get('fvu_standardized'),\n"
        "        corr_hidden_e=hd.get('max_abs_corr_hiddenPC_vs_score'),\n"
        "        rq1_rho=rq1.get('spearman_rho'), rq1_jaccard=rq1.get('top_set_jaccard'),\n"
        "        delta_pos_rate=log.get('ablation_positive_rate'),\n"
        "        h3_d_coverage=h3.get('d_coverage'), h3_d_width=h3.get('d_width'),\n"
        "        h3_d_winkler=h3.get('d_winkler'), h3_d_mace=h3.get('d_mace')))\n"
        "raw = pd.DataFrame(rows).sort_values(['dataset','seed'])\n"
        "print('=== per-run ==='); display(raw)\n"
        "\n"
        "agg = raw.groupby('dataset').agg(['mean','std']).round(4)\n"
        "print('=== aggregated across seeds (mean / std) ==='); display(agg)\n"
        "raw.to_csv('runs/AGG_per_run.csv', index=False)\n"
        "agg.to_csv('runs/AGG_by_dataset.csv')\n"
        "\n"
        "# H3 read: positive d_coverage AND negative d_width => MVDF both covers better and\n"
        "# stays tighter under shift (outcome A). Flag datasets where the sign is stable.\n"
        "for ds, g in raw.groupby('dataset'):\n"
        "    cov = g['h3_d_coverage']; wid = g['h3_d_width']\n"
        "    stable_cov = (cov > 0).all() or (cov < 0).all()\n"
        "    print(f'{ds:12s} H3 d_coverage {cov.mean():+.4f}+-{cov.std():.4f} '\n"
        "          f'd_width {wid.mean():+.4f}+-{wid.std():.4f}  '\n"
        "          f'{\"[stable sign]\" if stable_cov else \"[sign flips across seeds]\"}')\n"
        "\n"
        "for rd in sorted(glob.glob('runs/mve_*')):\n"
        "    p = f'{rd}/coverage_width_over_time.png'\n"
        "    if os.path.exists(p):\n"
        "        print(rd); display(Image(p))\n"
    ))

    cells.append(md("## 5. Zip for download"))
    cells.append(code(
        "import shutil\n"
        "shutil.make_archive('/kaggle/working/mve_runs', 'zip', 'runs')\n"
        "print('runs -> /kaggle/working/mve_runs.zip')\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "mve_kaggle.ipynb"))
    ap.add_argument("--full-suite", action="store_true",
                    help="all 9 datasets (overrides --datasets)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of datasets to run (ignored with --full-suite)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seeds per dataset; >1 triggers cross-seed aggregation")
    args = ap.parse_args()
    nb = build(full_suite=args.full_suite, datasets=args.datasets, seeds=args.seeds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)} bytes)")
