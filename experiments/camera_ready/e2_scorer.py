"""E2 metric 4: continuous learned scorer vs naive_matched, the v2_h6_topup protocol
(experiments/phase3_scorer_h6.py lines 97-188), from saved arrays, modes dense + mvdf.

Unchanged protocol details: E_eval_ID split 50/30/20; every block (train, conformalize,
ID width-eval, shift) gets its own collect()-style profile and score; DifficultyScorer
fit/conformalize/run with global ACI gamma 0.03 (configs/v2_phase3_h6.yaml); naive_matched =
constant quantile of train U conformalize scores + the same global ACI.
Added: moving-block bootstrap CI (block 96) on the paired per-window width difference.
  python -m experiments.camera_ready.e2_scorer --datasets ETTh1 ETTh2 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

from experiments.camera_ready.common import Run, CR, ALPHA, boot_mean, profile_of, score_with, conformal_split
from experiments._seed import derive_seed
from mechconf.calibration.difficulty_scorer import DifficultyScorer, build_features, naive_no_scorer_run
from mechconf.eval import metrics

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
GAMMA = 0.03
BOOTS = {}


def run_cell(ds, seed):
    run = Run(ds, seed, load_X=False)
    ds_seed = derive_seed(seed, ds)
    tr, cf, te = conformal_split(run)
    sh = run.idx_sh

    def bundle(idx):
        ae = run.abs_err[idx]
        prof = profile_of(ae)
        return dict(hidden=run.hidden[idx], z=run.z_dense(idx), score=score_with(ae, prof),
                    yhat=run.yhat[idx].astype(np.float64), y=run.y[idx].astype(np.float64), prof=prof)
    b = {k: bundle(i) for k, i in (("tr", tr), ("cf", cf), ("te", te), ("sh", sh))}
    res, widths = [], {}
    for mode in ("dense", "mvdf"):
        f = {k: build_features(mode, b[k]["hidden"], b[k]["z"], run.mvdf) for k in b}
        sc = DifficultyScorer(mode, hidden_dim=f["tr"].shape[1]).fit(f["tr"], b["tr"]["score"],
                                                                    alpha=ALPHA, seed=ds_seed)
        sc.conformalize(f["cf"], b["cf"]["score"])
        for blk in ("te", "sh"):
            r = sc.run(f[blk], b[blk]["yhat"], b[blk]["y"], b[blk]["prof"], ALPHA, gamma=GAMMA)
            m = metrics.summarize(r, b[blk]["y"], ALPHA)
            widths[(mode, blk)] = np.asarray(r["width"])
            res.append(dict(mode=mode, block=blk, coverage=m["coverage"], width=m["mean_width"]))
    s80 = np.concatenate([b["tr"]["score"], b["cf"]["score"]])
    for blk in ("te", "sh"):
        r = naive_no_scorer_run(s80, b[blk]["yhat"], b[blk]["y"], b[blk]["prof"], ALPHA, gamma=GAMMA)
        m = metrics.summarize(r, b[blk]["y"], ALPHA)
        widths[("naive_matched", blk)] = np.asarray(r["width"])
        res.append(dict(mode="naive_matched", block=blk, coverage=m["coverage"], width=m["mean_width"]))
    for r in res:
        r.update(dataset=ds, seed=seed, n_test=len(te) if r["block"] == "te" else len(sh),
                 block={"te": "ID", "sh": "shift"}[r["block"]])
    out = []
    for r in res:
        if r["mode"] != "naive_matched":
            k = "te" if r["block"] == "ID" else "sh"
            d = widths[(r["mode"], k)] - widths[("naive_matched", k)]
            mu, lo, hi, bb = boot_mean(d)
            BOOTS[(ds, seed, r["mode"], r["block"])] = bb
            nm = [x for x in res if x["mode"] == "naive_matched" and x["block"] == r["block"]][0]
            r.update(width_minus_naive=mu, wmn_lo=lo, wmn_hi=hi,
                     coverage_minus_naive=r["coverage"] - nm["coverage"],
                     naive_coverage=nm["coverage"], naive_width=nm["width"])
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(CR, "E2", "AGG_E2_scorer_vs_naive.csv"))
    a = ap.parse_args()
    rows, t0 = [], time.time()
    for ds in a.datasets:
        for s in a.seeds:
            rr = run_cell(ds, s)
            rows += rr
            sh = [r for r in rr if r["block"] == "shift" and r["mode"] != "naive_matched"]
            print(f"[{ds} s{s}] " + "  ".join(f"{r['mode']} dW_shift {r['width_minus_naive']:+.3f} "
                                              f"[{r['wmn_lo']:+.3f},{r['wmn_hi']:+.3f}] cov {r['coverage']:.3f}"
                                              for r in sh), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.to_pickle(BOOTS, a.out.replace(".csv", "_boot.pkl"))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
