"""G1: Table 7 on the local data copy. Every downstream mode, original protocols unchanged,
from saved arrays + G/quantile. See runs/camera_ready/G/PREREG.md.
  python -m camera_ready.g1_table7
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from camera_ready.common import (Run, CR, ALPHA, boot_mean, profile_of, score_with, conformal_split,
                                 available_seeds)
from experiments._seed import derive_seed
from calibration.difficulty_scorer import DifficultyScorer, build_features, naive_no_scorer_run
from calibration.aci import empirical_quantile
from baselines.cqr import cqr_calibrate, cqr_run
from baselines.pid_conformal import pid_conformal_run
from eval import metrics

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
GAMMA = 0.03
OUT = os.path.join(CR, "G")
SCORER_MODES = ["dense", "mvdf", "dense+mvdf", "correlated", "random_k", "pca_k"]


def run_cell(ds, seed):
    r = Run(ds, seed, load_X=False)
    q = np.load(os.path.join(OUT, "quantile", f"{r.tag}.npz"))
    ds_seed = derive_seed(seed, ds)
    tr, cf, te = conformal_split(r)
    sh = r.idx_sh

    def bundle(idx):
        ae = r.abs_err[idx]; prof = profile_of(ae)
        return dict(hidden=r.hidden[idx], z=r.z_dense(idx), score=score_with(ae, prof),
                    yhat=r.yhat[idx].astype(np.float64), y=r.y[idx].astype(np.float64), prof=prof,
                    lo=q["q_lo"][idx].astype(np.float64), hi=q["q_hi"][idx].astype(np.float64))
    b = {k: bundle(i) for k, i in (("tr", tr), ("cf", cf), ("te", te), ("sh", sh))}
    k = len(r.mvdf)
    rng = np.random.default_rng(ds_seed + 777)
    elig = np.where((b["tr"]["z"] != 0).mean(0) >= 0.02)[0]
    rand_idx = np.sort(rng.choice(elig, size=min(k, len(elig)), replace=False))
    pca = PCA(n_components=k).fit(b["tr"]["hidden"].astype(np.float64))
    fkw = dict(corr_idx=r.correlated, rand_idx=rand_idx, pca_model=pca, k=k)
    res = {}
    for mode in SCORER_MODES:
        f = {kk: build_features(mode, b[kk]["hidden"], b[kk]["z"], r.mvdf, **fkw) for kk in b}
        sc = DifficultyScorer(mode, hidden_dim=f["tr"].shape[1]).fit(f["tr"], b["tr"]["score"], alpha=ALPHA, seed=ds_seed)
        sc.conformalize(f["cf"], b["cf"]["score"])
        for blk in ("te", "sh"):
            res[(mode, blk)] = sc.run(f[blk], b[blk]["yhat"], b[blk]["y"], b[blk]["prof"], ALPHA, gamma=GAMMA)
    s80 = np.concatenate([b["tr"]["score"], b["cf"]["score"]])
    for nm, scal in (("naive_matched", s80), ("naive_no_scorer", b["cf"]["score"])):
        for blk in ("te", "sh"):
            res[(nm, blk)] = naive_no_scorer_run(scal, b[blk]["yhat"], b[blk]["y"], b[blk]["prof"], ALPHA, gamma=GAMMA)
    qh = cqr_calibrate(b["cf"]["lo"], b["cf"]["hi"], b["cf"]["y"], ALPHA, step_profile=b["cf"]["prof"])
    q0 = empirical_quantile(s80, 1 - ALPHA)
    for blk in ("te", "sh"):
        res[("cqr", blk)] = cqr_run(b[blk]["lo"], b[blk]["hi"], b[blk]["y"], qh, step_profile=b[blk]["prof"])
        res[("pid_conformal", blk)] = pid_conformal_run(q0, b[blk]["yhat"], b[blk]["y"], b[blk]["prof"], ALPHA, eta=GAMMA)
    rows = []
    for (mode, blk), rr in res.items():
        m = metrics.summarize(rr, b[blk]["y"], ALPHA)
        nm = res[("naive_matched", blk)]
        d = np.asarray(rr["width"]) - np.asarray(nm["width"])
        mu, lo, hi, _ = boot_mean(d)
        rows.append(dict(dataset=ds, seed=seed, mode=mode, block={"te": "ID", "sh": "shift"}[blk],
                         n_test=len(b[blk]["y"]), coverage=m["coverage"], width=m["mean_width"],
                         width_minus_naive=mu, wmn_lo=lo, wmn_hi=hi))
    return rows


def main():
    t0 = time.time()
    rows = []
    for ds in ["ETTh1", "ETTh2", "electricity", "traffic"]:
        for s in available_seeds(ds):
            if not os.path.exists(os.path.join(OUT, "quantile", f"{ds}_seed{s}.npz")):
                print(f"[G1] missing quantile for {ds} s{s}", flush=True); continue
            rows += run_cell(ds, s)
            print(f"[G1 {ds} s{s}] done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["cov_valid"] = (df.coverage - 0.9).abs() <= 0.02
    nmv = df[df["mode"] == "naive_matched"][["dataset", "seed", "block", "coverage"]].rename(columns={"coverage": "naive_coverage"})
    df = df.merge(nmv, on=["dataset", "seed", "block"])
    df["narrower_valid"] = (df.width_minus_naive < 0) & df.cov_valid & (df["mode"] != "naive_matched")
    df["narrower_sig"] = (df.wmn_hi < 0) & (df["mode"] != "naive_matched")
    df.to_csv(os.path.join(OUT, "AGG_G1_table7_local.csv"), index=False)
    print(f"G1 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
