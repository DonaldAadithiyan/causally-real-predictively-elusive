"""Generate a self-contained Kaggle notebook that runs V2 Phases 0 -> 1 -> 2, gated.

  python notebooks/build_v2_notebook.py --out notebooks/v2_phases012.ipynb

The notebook:
  1. vendors the whole mechanistic-conformal package (V2 modules included)
  2. runs Phase 0 (layer sweep, all datasets) and reads the GO / NO_GO gate
  3. runs Phase 1 (dose-response, R_j, feature-type table, MVDF_v2 selection) at the
     winning taps (or pre_head fallback on NO_GO)
  4. runs Phase 2 (cross-environment invariance) for every feature-set condition
  5. displays the H1c enrichment table and the flagship AGG_invariance.csv
  6. zips runs/ for download

Kaggle: GPU + Internet on, attach the uom220001c multivariate-TS bundle. Run All.
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import argparse
import base64
import json
import os

ROOT = REPO_ROOT

PKG_FILES = [
    "__init__.py",
    "data/__init__.py", "data/datasets.py", "data/probe_inputs.py",
    "forecasters/__init__.py", "forecasters/itransformer.py",
    "forecasters/train_itransformer.py",
    "sae/__init__.py", "sae/topk_sae.py", "sae/train_sae.py",
    "validation/__init__.py", "validation/errors.py", "validation/correlation_screen.py",
    "validation/ablation.py", "validation/rq1.py",
    "validation/importance_and_difficulty.py", "validation/dose_response.py",
    "validation/difficulty_specificity.py", "validation/invariance.py",
    "calibration/__init__.py", "calibration/aci.py", "calibration/mondrian_perbin.py",
    "calibration/ffdci_head.py",
    "representation/__init__.py", "representation/layer_sweep.py",
    "shift/__init__.py", "shift/shift_windows.py",
    "eval/__init__.py", "eval/metrics.py", "eval/plots.py",
    "experiments/__init__.py", "experiments/mve_real.py",
    "experiments/phase0_layer_sweep.py", "experiments/phase1_core_mechanism.py",
    "experiments/phase2_invariance.py",
    "configs/mve_etth1.yaml", "configs/v2_phase0.yaml", "configs/v2_phase1.yaml",
    "configs/v2_phase2.yaml", "configs/environment_splits/README.md",
]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build(datasets, fast: bool):
    ds_list = datasets or ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "traffic", "electricity",
                           "weather", "solar"]
    cells = []
    cells.append(md(
        "# Mechanistic Uncertainty-Causal CP — V2 Phases 0-2 (gated)\n\n"
        "Self-contained. Closes the gap between the v1 spec and Run 5's simplified pipeline:\n"
        "- **Phase 0** — layer sweep (spec §4.5): is difficulty encoded somewhere other than "
        "the pre-head layer? Cheap `max|corr(activation PC, S_t)|` across 7 taps, all datasets. "
        "Pre-registered GO threshold 0.30 (Run 5's pre-head ceiling was 0.09-0.22).\n"
        "- **Phase 1** — dose-response `P_j`/`U_j` from one intervention, residual `R_j = U_j - "
        "E[U_j|P_j]`, the A/B/C/D feature-type table, and `MVDF_v2` selection "
        "(|U_j| gate + top-k by R_j). H1/H1b/H1c tables.\n"
        "- **Phase 2** — cross-environment invariance `I(R)` (the measurement Run 5 skipped) "
        "for every feature-set condition, on E_eval_ID + shift sub-blocks, with the hard "
        "leakage assertion (E_select never re-touched).\n\n"
        "**Kaggle:** GPU + Internet on; attach the `uom220001c` multivariate-TS bundle. Run All.\n\n"
        "**Pre-registered outcomes (spec §7):** `I(MVDF_v2) < I(correlated)` across datasets → "
        "positive/row 1-2; `I(MVDF_v2) ≈ I(correlated)` → row 3, a genuine negative on the "
        "invariance hypothesis (distinct from Run 5, which never measured invariance)."
    ))

    payload = {}
    for rel in PKG_FILES:
        with open(os.path.join(ROOT, rel)) as f:
            payload[rel] = f.read()
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    cells.append(md("## 1. Vendor the package"))
    cells.append(code(
        "import os, base64, json\n"
        "PKG='mechanistic_conformal'\n"
        f"FILES=json.loads(base64.b64decode('{blob}').decode())\n"
        "for rel,content in FILES.items():\n"
        "    p=os.path.join(PKG,rel); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(content)\n"
        "print('wrote',len(FILES),'files')\n"
    ))

    cells.append(md("## 2. Deps + probe inputs"))
    cells.append(code(
        "import importlib,subprocess,sys,os\n"
        "for pkg in ['pyyaml','scipy','scikit-learn','statsmodels']:\n"
        "    n={'pyyaml':'yaml','scikit-learn':'sklearn'}.get(pkg,pkg)\n"
        "    try: importlib.import_module(n)\n"
        "    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True)\n"
        "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())\n"
        "sys.path.insert(0,'mechanistic_conformal'); os.chdir('mechanistic_conformal')\n"
        "from data.probe_inputs import probe; probe()\n"
    ))

    fastflag = "'--fast'," if fast else ""
    dss = json.dumps(ds_list)

    cells.append(md("## 3. Phase 0 — layer sweep (gate)"))
    cells.append(code(
        "import runpy, sys, torch, json\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "import yaml\n"
        "for cf in ['configs/v2_phase0.yaml','configs/v2_phase1.yaml','configs/v2_phase2.yaml']:\n"
        "    c=yaml.safe_load(open(cf)); c['device']=dev; yaml.safe_dump(c,open(cf,'w'))\n"
        f"sys.argv=['p0','--config','configs/v2_phase0.yaml','--datasets',*{dss},{fastflag}]\n"
        "try: runpy.run_module('experiments.v2.phase0_layer_sweep', run_name='__main__')\n"
        "except SystemExit: pass\n"
        "p0=json.load(open('results/v2/v2_phase0/layer_sweep_summary.json'))\n"
        "print('\\nPHASE 0:', p0['result'], '| winning:', p0['winning_layers'])\n"
    ))

    cells.append(md("## 4. Phase 1 — dose-response, R_j, feature-type table, MVDF_v2"))
    cells.append(code(
        "import runpy, sys\n"
        f"sys.argv=['p1','--config','configs/v2_phase1.yaml','--datasets',*{dss},{fastflag}]\n"
        "try: runpy.run_module('experiments.v2.phase1_core_mechanism', run_name='__main__')\n"
        "except SystemExit: pass\n"
        "import pandas as pd\n"
        "print('\\n=== H1c enrichment (type-B) across runs ===')\n"
        "display(pd.read_csv('results/v2/v2_phase1/AGG_h1c.csv'))\n"
    ))

    cells.append(md("## 5. Phase 2 — cross-environment invariance (flagship)"))
    cells.append(code(
        "import runpy, sys\n"
        f"sys.argv=['p2','--config','configs/v2_phase2.yaml','--datasets',*{dss},{fastflag}]\n"
        "try: runpy.run_module('experiments.v2.phase2_invariance', run_name='__main__')\n"
        "except SystemExit: pass\n"
        "import pandas as pd\n"
        "agg=pd.read_csv('results/v2/v2_phase2/AGG_invariance.csv')\n"
        "print('=== AGG_invariance.csv ==='); display(agg)\n"
        "cols=[c for c in agg.columns if c.startswith('I_') and c not in ('I_MVDFv2_lt_correlated','I_MVDFv2_lt_MVDFv1')]\n"
        "print('\\nMean I by condition (lower = more invariant):')\n"
        "print(agg[cols].mean().round(4).to_string())\n"
        "print('\\nI(MVDF_v2) < I(correlated):', agg['I_MVDFv2_lt_correlated'].tolist())\n"
        "print('I(MVDF_v2) < I(MVDF_v1)  :', agg['I_MVDFv2_lt_MVDFv1'].tolist())\n"
        "n=agg['I_MVDFv2_lt_correlated'].sum(); tot=agg['I_MVDFv2_lt_correlated'].notna().sum()\n"
        "print(f'\\nMVDF_v2 more invariant than correlated on {n}/{tot} (dataset,tap) runs')\n"
    ))

    cells.append(md("## 6. Zip"))
    cells.append(code(
        "import shutil; shutil.make_archive('/kaggle/working/v2_runs','zip','results', 'v2')\n"
        "print('runs -> /kaggle/working/v2_runs.zip')\n"
    ))

    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "v2_phases012.ipynb"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    nb = build(a.datasets, a.fast)
    with open(a.out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)} bytes)")
