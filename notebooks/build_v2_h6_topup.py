"""Top-up notebook: aggregation diagnostic + H6 ONLY, reusing Phase 1 outputs from a
prior v2_confirm run (its mvdf_v2_selection.json / candidates.csv files).

  python notebooks/build_v2_h6_topup.py --out notebooks/v2_h6_topup.ipynb

Phase 1 + Phase 2 already completed in the previous run. This skips them (~45 min
saved) and runs only:
  - the aggregation diagnostic (does any aggregation give MVDF_v2/correlated a real
    signal? — the H3-is-structural check, now on 4 datasets)
  - the 6-mode split-conformal H6 (dense / mvdf / dense+mvdf / correlated / random_k /
    pca_k), the one result still needed before write-up

The forecaster + SAE are still retrained per (dataset, seed) — deterministic from
ds_seed, so identical to the previous run — but the Phase 1 dose-response sweep is
NOT re-run (its output is read from the uploaded zip).

Kaggle setup:
  1. GPU + Internet on, attach the `uom220001c` bundle.
  2. **Also attach the previous run's output**: download `v2_confirm.zip` from the
     prior run's output, upload it as a new Kaggle Dataset, attach it here. The
     notebook auto-finds `v2_phase1_seed*/` inside any attached input.
  3. Run All. ~20-25 min.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKG_FILES = [
    "__init__.py",
    "data/__init__.py", "data/datasets.py", "data/probe_inputs.py", "data/fingerprint.py",
    "forecasters/__init__.py", "forecasters/itransformer.py", "forecasters/train_itransformer.py",
    "sae/__init__.py", "sae/topk_sae.py", "sae/train_sae.py",
    "validation/__init__.py", "validation/errors.py", "validation/correlation_screen.py",
    "validation/ablation.py", "validation/rq1.py",
    "validation/importance_and_difficulty.py", "validation/dose_response.py",
    "validation/difficulty_specificity.py", "validation/invariance.py",
    "calibration/__init__.py", "calibration/aci.py", "calibration/mondrian_perbin.py",
    "calibration/ffdci_head.py", "calibration/difficulty_scorer.py",
    "representation/__init__.py", "representation/layer_sweep.py",
    "shift/__init__.py", "shift/shift_windows.py",
    "eval/__init__.py", "eval/metrics.py", "eval/plots.py",
    "experiments/__init__.py", "experiments/_seed.py", "experiments/mve_real.py",
    "experiments/phase3_scorer_h6.py", "experiments/aggregation_diagnostic.py",
    "experiments/h6_and_diagnostic.py",
    "configs/mve_etth1.yaml", "configs/v2_phase2.yaml", "configs/v2_phase3_h6.yaml",
]

DATASETS = ["ETTh1", "ETTh2", "electricity", "traffic"]
SEEDS = [0, 1, 2]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build():
    cells = []
    cells.append(md(
        "# V2 top-up — H6 + agg diagnostic + Task G matched-budget check\n\n"
        "**Purpose of this run:** the previous H6 pass found `naive_no_scorer` beats "
        "every learned scorer, but on an UNMATCHED calibration budget (learned modes get "
        "80% of E_eval_ID, naive got 30%). This run adds `naive_matched` (80%, same data "
        "as the learned modes) so the comparison is budget-matched. If `naive_matched` "
        "still wins on width AND coverage -> the H6 conclusion holds and the paper frame "
        "locks (see `PAPER_FRAME.md`).\n\n"
        "Phase 1 + Phase 2 already completed. This reuses their outputs (from the "
        "previous run's `v2_confirm.zip`, attached as a Kaggle input) and runs only:\n"
        "- **aggregation diagnostic** — does mean / max / L2 / active-count aggregation "
        "give `MVDF_v2` / `correlated` a signal with ≥4 populated quantile bins? (the "
        "check for whether H3 is structurally unmeasurable, now on 4 datasets)\n"
        "- **H6 four-way** — `dense` / `mvdf` / `dense+mvdf` / `correlated` / `random_k` "
        "/ `pca_k`, split-conformal calibration, α=0.10, ID + shift, paired bootstrap.\n\n"
        "**Attach BOTH**: the `uom220001c` data bundle AND the previous run's output "
        "(upload `v2_confirm.zip` as a dataset). GPU + Internet on. Run All. ~20-25 min."
    ))

    payload = {}
    for rel in PKG_FILES:
        with open(os.path.join(ROOT, rel)) as f:
            payload[rel] = f.read()
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    cells.append(md("## 1. Vendor + deps + locate the prior Phase 1 outputs"))
    cells.append(code(
        "import os, base64, json, sys, subprocess, importlib, glob, shutil, zipfile\n"
        f"FILES=json.loads(base64.b64decode('{blob}').decode())\n"
        "for rel,c in FILES.items():\n"
        "    p=os.path.join('mechanistic_conformal',rel); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(c)\n"
        "print('wrote',len(FILES),'files')\n"
        "for pkg in ['pyyaml','scipy','scikit-learn','statsmodels']:\n"
        "    n={'pyyaml':'yaml','scikit-learn':'sklearn'}.get(pkg,pkg)\n"
        "    try: importlib.import_module(n)\n"
        "    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True)\n"
        "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())\n"
        "sys.path.insert(0,'mechanistic_conformal'); os.chdir('mechanistic_conformal')\n"
        "\n"
        "# find v2_phase1_seed* dirs anywhere under /kaggle/input, or inside a zip there\n"
        "os.makedirs('runs', exist_ok=True)\n"
        "found=[]\n"
        "for root,dirs,files in os.walk('/kaggle/input'):\n"
        "    for d in dirs:\n"
        "        if d.startswith('v2_phase1_seed'):\n"
        "            src=os.path.join(root,d); dst=os.path.join('runs',d)\n"
        "            if not os.path.exists(dst): shutil.copytree(src,dst); found.append(d)\n"
        "    for f in files:\n"
        "        if f.endswith('.zip'):\n"
        "            try:\n"
        "                with zipfile.ZipFile(os.path.join(root,f)) as z:\n"
        "                    names=[n for n in z.namelist() if 'v2_phase1_seed' in n]\n"
        "                    if names:\n"
        "                        z.extractall('runs/_unzip'); print('extracted', f)\n"
        "                        for p in glob.glob('runs/_unzip/**/v2_phase1_seed*', recursive=True):\n"
        "                            b=os.path.basename(p); dst=os.path.join('runs',b)\n"
        "                            if os.path.isdir(p) and not os.path.exists(dst):\n"
        "                                shutil.copytree(p,dst); found.append(b)\n"
        "            except zipfile.BadZipFile: pass\n"
        "print('Phase 1 dirs available:', sorted(set(os.path.basename(p) for p in glob.glob('runs/v2_phase1_seed*'))))\n"
        "assert glob.glob('runs/v2_phase1_seed*/*/mvdf_v2_selection.json'), \\\n"
        "    'No Phase 1 selections found. Attach the previous run output (v2_confirm.zip as a dataset).'\n"
        "from data.probe_inputs import probe; probe()\n"
    ))

    dss = json.dumps(DATASETS); sds = json.dumps(SEEDS)
    cells.append(md("## 2. Configure"))
    cells.append(code(
        "import yaml, torch\n"
        "dev='cuda' if torch.cuda.is_available() else 'cpu'\n"
        "for cf in ['configs/v2_phase2.yaml','configs/v2_phase3_h6.yaml']:\n"
        "    c=yaml.safe_load(open(cf)); c['device']=dev; c['strict_fingerprint']=True\n"
        f"    c['datasets']={dss}\n"
        "    if 'seeds' in c: c['seeds']=" + sds + "\n"
        "    yaml.safe_dump(c,open(cf,'w'))\n"
        "print('configured: device',dev)\n"
    ))

    cells.append(md(
        "## 3. Aggregation diagnostic + H6  (one forecaster/SAE per dataset-seed)\n\n"
        "`h6_and_diagnostic.py` trains each deterministic iTransformer + SAE **once** and "
        "runs both the aggregation diagnostic and the 6-mode split-conformal H6 from it "
        "(the earlier top-up design would have retrained twice). ~20-25 min for 4×3."
    ))
    cells.append(code(
        "import runpy, sys\n"
        "sys.argv=['hd','--datasets']+[str(x) for x in " + dss + "]"
        "+['--seeds']+[str(x) for x in " + sds + "]"
        "+['--phase1_root_tmpl','runs/v2_phase1_seed{seed}']\n"
        "try: runpy.run_module('experiments.h6_and_diagnostic', run_name='__main__')\n"
        "except SystemExit: pass\n"
    ))

    cells.append(md("## 4. Read the results"))
    cells.append(code(
        "import pandas as pd\n"
        "diag=pd.read_csv('runs/h6_and_diag/AGG_diag.csv')\n"
        "print('=== AGG DIAGNOSTIC: bins_populated by condition x agg ===')\n"
        "print(diag.pivot_table(index='condition',columns='agg',values='bins_populated').round(2).to_string())\n"
        "print('\\n=== per-feature activation rate (median) ===')\n"
        "print(diag.groupby('condition')['median_per_feat_active'].mean().round(3).to_string())\n"
        "print('  (MVDF_v2/correlated staying ~2-3 bins under ALL aggs -> H3 structurally unmeasurable)')\n"
        "\n"
        "h6=pd.read_csv('runs/h6_and_diag/AGG_h6.csv')\n"
        "print('\\n=== H6 coverage sanity (split-conformal -> ~0.90 ID) ===')\n"
        "print(h6.groupby('mode')[['id_coverage','shift_coverage']].mean().round(3).to_string())\n"
        "print('\\n=== per-seed shift width by mode ===')\n"
        "print(h6.pivot_table(index=['dataset','seed'],columns='mode',values='shift_width').round(3).to_string())\n"
        "if 'naive_no_scorer' in set(h6['mode']):\n"
        "    print('\\n=== Task G: parameter-free constant-width CP vs the learned-scorer family ===')\n"
        "    sh=['naive_matched','naive_no_scorer','dense','mvdf'] if 'naive_matched' in set(h6['mode']) else ['naive_no_scorer','dense','mvdf']\n"
        "    g=h6[h6['mode'].isin(sh)]\n"
        "    print(g.groupby('mode')[['id_coverage','id_width','shift_coverage','shift_width']].mean().reindex(sh).round(3).to_string())\n"
        "    print('  naive_no_scorer = 30% conformalize block only; naive_matched = train U conf (80%, matched to learned modes)')\n"
        "    print('  DECISION: does naive_matched still beat every learned mode on width AND coverage? -> H6 conclusion holds at matched budget')\n"
        "    if 'naive_matched' in set(h6['mode']):\n"
        "        em=h6[h6.dataset=='electricity'].pivot_table(index='mode',columns='seed',values='shift_coverage').reindex(sh).round(3)\n"
        "        print('\\n  electricity shift coverage by seed (cf mvdf 0.827/0.630/0.811):'); print(em.to_string())\n"
        "\n"
        "t=pd.read_csv('runs/h6_and_diag/AGG_h6_tests.csv')\n"
        "summ=t.groupby(['comparison','env']).agg(mean_diff=('mean_diff','mean'),"
        "p_a_lt_b=('p_a_lt_b','mean'),n_sig=('significant','sum'),n=('significant','size')).round(4)\n"
        "print('\\n=== paired bootstrap by comparison x env (mean over 12 dataset-seed runs) ===')\n"
        "print(summ.to_string())\n"
        "def sig(comp,env):\n"
        "    try: r=summ.loc[(comp,env)]; return f'{r.mean_diff:+.3f} (sig {int(r.n_sig)}/{int(r.n)})'\n"
        "    except KeyError: return 'n/a'\n"
        "print('\\n=== FOUR-WAY READ ===')\n"
        "for env in ['ID','shift']:\n"
        "    print(f'  [{env}] mvdf-dense {sig(\"mvdf_minus_dense\",env)}  mvdf-correlated {sig(\"mvdf_minus_correlated\",env)}  '\n"
        "          f'mvdf-random_k {sig(\"mvdf_minus_random_k\",env)}  mvdf-pca_k {sig(\"mvdf_minus_pca_k\",env)}  '\n"
        "          f'pca_k-dense {sig(\"pca_k_minus_dense\",env)}')\n"
        "print()\n"
        "print('  mvdf<dense but mvdf~correlated~random_k     -> any k-dim SAE selection helps, not causal content')\n"
        "print('  mvdf<dense AND mvdf<correlated              -> dose-response/R_j content specifically matters')\n"
        "print('  mvdf~random_k<dense BUT pca_k~dense         -> SAE dictionary basis specifically (not any compression)')\n"
    ))

    cells.append(md("## 5. Zip"))
    cells.append(code(
        "import shutil\n"
        "shutil.make_archive('/kaggle/working/v2_h6_topup','zip','runs/h6_and_diag')\n"
        "print('runs/h6_and_diag -> /kaggle/working/v2_h6_topup.zip')\n"
    ))

    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "v2_h6_topup.ipynb"))
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(build(), f, indent=1)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)} bytes)")
