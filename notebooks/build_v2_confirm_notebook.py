"""Generate the V2 confirm-run notebook: multi-seed Phase 1 + Phase 2 (fixed
intervention_only) + Phase 3 H6, on ETTh1 + ETTh2 + electricity, seeds 0/1/2.

  python notebooks/build_v2_confirm_notebook.py --out notebooks/v2_confirm.ipynb

Purpose (per the review):
  1. FIX intervention_only (top-|U_j|, no R_j) so Phase 2 can test whether the R_j
     gate adds invariance beyond dose-response alone.
  2. Multi-seed error bars on the flagship I(MVDF_v2) vs I(correlated) — the single
     "win" (ETTh1, margin 0.053) is thin enough seed noise could flip it.
  3. Close H6: g_MVDF vs g_dense interval width, the one downstream test v1's causal
     chain says to run regardless of H3's outcome.

All runs carry a DATA FINGERPRINT ASSERT (data/fingerprint.py): the pipeline refuses
to run if the resolved dataset shape/hash doesn't match the recorded canonical
(uom220001c bundle). Per-dataset seeding (experiments/_seed.py) makes a 3-dataset run
byte-identical to picking those datasets out of an 8-dataset run.

Kaggle: GPU + Internet on, attach uom220001c. Run All. ~2-3 hr.
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
    "experiments/phase0_layer_sweep.py", "experiments/phase1_core_mechanism.py",
    "experiments/phase2_invariance.py", "experiments/phase3_scorer_h6.py",
    "experiments/aggregation_diagnostic.py",
    "configs/mve_etth1.yaml", "configs/v2_phase0.yaml", "configs/v2_phase1.yaml",
    "configs/v2_phase2.yaml", "configs/v2_phase3_h6.yaml",
    "configs/environment_splits/README.md",
]

DATASETS = ["ETTh1", "ETTh2", "electricity", "traffic"]  # traffic added so the
#   H6 SAE-basis-vs-PCA distinction isn't an ETTh2 idiosyncrasy
SEEDS = [0, 1, 2]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build():
    cells = []
    cells.append(md(
        "# V2 confirm run v2 — multi-seed Phase 1 + 2 + widened H6\n\n"
        "Fixes from two review rounds:\n"
        "1. **`intervention_only`** = top-k by `|U_j|` (no `R_j`) — Phase 2 tests whether "
        "the `R_j` gate adds invariance beyond dose-response alone.\n"
        "2. **Multi-seed** (0/1/2) on ETTh1 + ETTh2 + electricity — error bars on "
        "`I(MVDF_v2)` vs `I(correlated)`.\n"
        "3. **Invariance degeneracy guard** — a condition whose feature signal populates "
        "<3 quantile bins now yields `I = NaN` (was `~0`, which spuriously read as "
        "'perfectly invariant' — this inflated the ETTh1 seed sd).\n"
        "4. **H6 widened to a four-way decomposition** with a proper split-conformal "
        "calibration (50% MLP-train / 30% conformalize / 20% ID-width eval of "
        "`E_eval_ID`), so ID coverage is ~1-α by construction:\n"
        "   - `dense` (FFDCI as-is) vs `mvdf` vs `dense+mvdf`\n"
        "   - **`correlated`** — top-k correlation-selected features, same k, same scorer. "
        "*The sharpest test: does dose-response/`R_j` earn anything over the cheap baseline?*\n"
        "   - **`random_k`** — k random SAE features (dimensionality control)\n"
        "   - **`pca_k`** — top-k PCA of the dense vector (sensible-compression control)\n"
        "   Read: `mvdf < dense` but `mvdf ≈ correlated` → any selection helps, not causal "
        "content. `mvdf < dense AND < correlated` → the `R_j` content matters. "
        "`mvdf ≈ random_k ≈ pca_k < dense` → dimensionality reduction alone.\n\n"
        "**Guards:** data-fingerprint assert (channels exact for ETT/weather, ±1 for "
        "capped sets); per-dataset deterministic seeding.\n\n"
        "Kaggle: GPU + Internet on, attach `uom220001c`. Run All. ~2-3 hr."
    ))

    payload = {}
    for rel in PKG_FILES:
        with open(os.path.join(ROOT, rel)) as f:
            payload[rel] = f.read()
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    cells.append(md("## 1. Vendor + deps + probe"))
    cells.append(code(
        "import os, base64, json, sys, subprocess, importlib\n"
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
        "from data.probe_inputs import probe; probe()\n"
    ))

    dss = json.dumps(DATASETS); sds = json.dumps(SEEDS)
    cells.append(md("## 2. Set device + strict fingerprint on all configs"))
    cells.append(code(
        "import yaml, torch\n"
        "dev='cuda' if torch.cuda.is_available() else 'cpu'\n"
        "for cf in ['configs/v2_phase1.yaml','configs/v2_phase2.yaml','configs/v2_phase3_h6.yaml']:\n"
        "    c=yaml.safe_load(open(cf)); c['device']=dev; c['strict_fingerprint']=True\n"
        f"    c['datasets']={dss}\n"
        "    yaml.safe_dump(c,open(cf,'w'))\n"
        "print('configs set: device',dev,'strict_fingerprint True, datasets',"+dss+")\n"
    ))

    cells.append(md("## 3. Phase 1 per seed  (dose-response, R_j, MVDF_v2, feature-type table)"))
    cells.append(code(
        "import runpy, sys, shutil\n"
        f"for sd in {sds}:\n"
        "    c=yaml.safe_load(open('configs/v2_phase1.yaml')); c['seed']=sd; c['run_name']=f'v2_phase1_seed{sd}'\n"
        "    yaml.safe_dump(c,open('configs/_p1.yaml','w'))\n"
        "    print('='*70,f'\\n PHASE 1  seed {sd}\\n','='*70)\n"
        "    sys.argv=['p1','--config','configs/_p1.yaml','--taps','pre_head']\n"
        "    try: runpy.run_module('experiments.v2.phase1_core_mechanism', run_name='__main__')\n"
        "    except SystemExit: pass\n"
        "import pandas as pd, glob\n"
        "for f in sorted(glob.glob('results/v2/v2_phase1_seed*/AGG_h1c.csv')):\n"
        "    print(f); display(pd.read_csv(f))\n"
    ))

    cells.append(md(
        "## 4. Phase 2 per seed  +  aggregation diagnostic\n\n"
        "**Phase 2's `I` is known-unreliable for `MVDF_v2` and `correlated`.** The "
        "bin-count audit showed their mean-aggregated feature signals populate only "
        "~2-3 of 5 quantile bins (features fire on ~2% of windows by design — that's "
        "what makes them difficulty-*specific*). The aggregation diagnostic below tests "
        "whether max / L2 / active-count aggregation rescues the signal; the local run "
        "showed it does not — no aggregation of a ~2%-active 12-feature bundle varies "
        "enough. So H3 is reported as **not cleanly measurable, and the reason is "
        "structural** (the H1c selection criterion and the H3 metric pull against each "
        "other). H1c + H6 carry the paper; Phase 2 `I` is context, not a headline."
    ))
    cells.append(code(
        "import runpy, sys\n"
        f"for sd in {sds}:\n"
        "    c=yaml.safe_load(open('configs/v2_phase2.yaml')); c['seed']=sd; c['run_name']=f'v2_phase2_seed{sd}'\n"
        "    yaml.safe_dump(c,open('configs/_p2.yaml','w'))\n"
        "    print('='*70,f'\\n PHASE 2  seed {sd}\\n','='*70)\n"
        "    sys.argv=['p2','--config','configs/_p2.yaml','--phase1_root',f'results/v2/v2_phase1_seed{sd}','--taps','pre_head']\n"
        "    try: runpy.run_module('experiments.v2.phase2_invariance', run_name='__main__')\n"
        "    except SystemExit: pass\n"
        "\n"
        "import pandas as pd, glob\n"
        "frames=[]\n"
        "for f in sorted(glob.glob('results/v2/v2_phase2_seed*/AGG_invariance.csv')):\n"
        "    d=pd.read_csv(f); d['seed']=int(f.split('seed')[1].split('/')[0]); frames.append(d)\n"
        "inv=pd.concat(frames, ignore_index=True)\n"
        "inv.to_csv('results/v2/AGG_invariance_multiseed.csv', index=False)\n"
        "deg_cols=[c for c in inv.columns if c.startswith('degenerate_')]\n"
        "print('=== degenerate-I flags (min shared bins < 3) by condition ===')\n"
        "if deg_cols: print(inv[deg_cols].sum().to_string())\n"
        "print('\\n=== I by condition across seeds (report WITH the degeneracy caveat) ===')\n"
        "for ds,g in inv.groupby('dataset'):\n"
        "    ic=g['I_correlated']; iv2=g['I_MVDF_v2']; iio=g['I_intervention_only']; iv1=g['I_MVDF_v1']\n"
        "    print(f'{ds:12s} I_corr {ic.mean():.3f}+-{ic.std():.3f}  I_MVDFv2 {iv2.mean():.3f}+-{iv2.std():.3f}'\n"
        "          f'  I_intOnly {iio.mean():.3f}+-{iio.std():.3f}  I_MVDFv1 {iv1.mean():.3f}+-{iv1.std():.3f}')\n"
        "\n"
        "# aggregation diagnostic — does ANY aggregation give MVDF_v2/correlated a real signal?\n"
        + "sys.argv=['ad','--datasets']+[str(x) for x in " + dss
        + "]+['--seeds']+[str(x) for x in " + sds
        + "]+['--phase1_root_tmpl','results/v2/v2_phase1_seed{seed}']\n"
        + "try: runpy.run_module('experiments.v2.aggregation_diagnostic', run_name='__main__')\n"
        "except SystemExit: pass\n"
        "except Exception as e: print('agg diagnostic failed (non-fatal):', e)\n"
    ))

    cells.append(md("## 5. Phase 3 — H6 four-way scorer decomposition (split-conformal)"))
    cells.append(code(
        "import runpy, sys\n"
        "c=yaml.safe_load(open('configs/v2_phase3_h6.yaml')); c['device']=dev; c['strict_fingerprint']=True\n"
        f"c['datasets']={dss}; c['seeds']={sds}\n"
        "yaml.safe_dump(c,open('configs/v2_phase3_h6.yaml','w'))\n"
        "sys.argv=['p3','--config','configs/v2_phase3_h6.yaml','--phase1_root_tmpl','results/v2/v2_phase1_seed{seed}']\n"
        "try: runpy.run_module('experiments.v2.phase3_scorer_h6', run_name='__main__')\n"
        "except SystemExit: pass\n"
        "import pandas as pd\n"
        "h6=pd.read_csv('results/v2/v2_phase3_h6/AGG_h6.csv')\n"
        "print('=== coverage sanity (split-conformal -> should be ~0.90 ID) ===')\n"
        "print(h6.groupby('mode')[['id_coverage','shift_coverage']].mean().round(3).to_string())\n"
        "print('\\n=== per-seed shift width by mode ===')\n"
        "print(h6.pivot_table(index=['dataset','seed'],columns='mode',values='shift_width').round(3).to_string())\n"
        "t=pd.read_csv('results/v2/v2_phase3_h6/AGG_h6_tests.csv')\n"
        "print('\\n=== paired bootstrap by comparison x env (mean over 9 dataset-seed runs) ===')\n"
        "summ=t.groupby(['comparison','env']).agg(mean_diff=('mean_diff','mean'),"
        "p_a_lt_b=('p_a_lt_b','mean'),n_sig=('significant','sum'),n=('significant','size')).round(4)\n"
        "print(summ.to_string())\n"
        "print('\\n=== FOUR-WAY READ ===')\n"
        "def sig(comp,env): r=summ.loc[(comp,env)]; return f\"{r.mean_diff:+.3f} (sig {int(r.n_sig)}/{int(r.n)})\"\n"
        "for env in ['ID','shift']:\n"
        "    print(f'  [{env}] mvdf-dense {sig(\"mvdf_minus_dense\",env)}  |  mvdf-correlated {sig(\"mvdf_minus_correlated\",env)}  '\n"
        "          f'|  mvdf-random_k {sig(\"mvdf_minus_random_k\",env)}  |  mvdf-pca_k {sig(\"mvdf_minus_pca_k\",env)}')\n"
        "print('\\n  mvdf<dense but mvdf~correlated       -> any sensible selection helps (not causal content)')\n"
        "print('  mvdf<dense AND mvdf<correlated       -> dose-response/R_j content specifically matters')\n"
        "print('  mvdf~random_k~pca_k, all < dense     -> dimensionality reduction alone explains it')\n"
    ))

    cells.append(md("## 6. Zip"))
    cells.append(code(
        "import shutil; shutil.make_archive('/kaggle/working/v2_confirm','zip','results', 'v2')\n"
        "print('runs -> /kaggle/working/v2_confirm.zip')\n"
    ))

    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "v2_confirm.ipynb"))
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(build(), f, indent=1)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)} bytes)")
