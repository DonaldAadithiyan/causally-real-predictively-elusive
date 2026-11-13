"""Round 2 WP5: activation history. Activations are functions of the input window, so
the activation of window t-k is known at the origin of window t (no H delay).
Features per SAE feature j (from trailing windows, current window included):
  current activation; lags {H/4, H/2, H}; trailing mean / max over {H/4, H, 2H};
  count of active windows over {H/4, H, 2H}; windows since last activation (cap 2H).
Evaluated with round-1 item-A metrics under blocked 5-fold CV inside E_eval_ID
(embargo L+H), GBM and ridge. Outputs runs/p126_round2/wp5/."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
HERE0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("P126_ARR", os.path.join(HERE0, "runs", "p126_round2", "arrays"))

from p126.common import Run, REAL, HERE, spearman, auroc, circ_null, dump  # noqa: E402
from p126.work_a import feature_sets  # noqa: E402
from p126.work_b import causal_score_feats, input_feats  # noqa: E402
from p126.r2_wp4 import folds, partial_spearman  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402

OUT = os.path.join(HERE, "runs", "p126_round2", "wp5")
os.makedirs(OUT, exist_ok=True)


def trailing(a, w, fn):
    """fn over a[t-w+1 .. t] for every t (all test windows, time order)."""
    n = len(a)
    out = np.empty(n)
    for t in range(n):
        out[t] = fn(a[max(0, t - w + 1):t + 1])
    return out


def history_feats(A, H):
    """A: (n_all, k) activations over ALL test windows in time order."""
    cols, names = [], []
    q, h2 = max(1, H // 4), 2 * H
    for j in range(A.shape[1]):
        a = A[:, j]
        on = (a != 0).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(a)]); cn = np.concatenate([[0.0], np.cumsum(on)])
        idx = np.arange(len(a))
        cols.append(a); names.append(f"f{j}_cur")
        for lag in (q, H // 2, H):
            l = np.r_[np.zeros(lag), a[:-lag]]; cols.append(l); names.append(f"f{j}_lag{lag}")
        for w in (q, H, h2):
            lo = np.maximum(0, idx - w + 1)
            cols.append((cs[idx + 1] - cs[lo]) / (idx + 1 - lo)); names.append(f"f{j}_mean{w}")
            cols.append((cn[idx + 1] - cn[lo])); names.append(f"f{j}_cnt{w}")
            from numpy.lib.stride_tricks import sliding_window_view
            pad = np.r_[np.zeros(w - 1), a]
            cols.append(sliding_window_view(pad, w).max(1)); names.append(f"f{j}_max{w}")
        last = -10 ** 9; tsl = np.empty(len(a))
        for t in range(len(a)):
            if on[t]:
                last = t
            tsl[t] = min(t - last, h2)
        cols.append(tsl); names.append(f"f{j}_since")
    return np.column_stack(cols), names


def cv_metrics(F, y, ctrl, emb, seed, model="gbm"):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    fl = folds(n, 5, emb)
    oof = np.full(n, np.nan)
    for tr, te in fl:
        if model == "gbm":
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=seed).fit(F[tr], y[tr])
            oof[te] = m.predict(F[te])
        else:
            sc = StandardScaler().fit(F[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(sc.transform(F[tr]), y[tr])
            oof[te] = m.predict(sc.transform(F[te]))

    def within(p):
        return float(np.mean([spearman(p[te], y[te]) for _, te in fl]))
    fold_r2 = [1 - np.sum((y[te] - oof[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2) for _, te in fl]
    _, p = circ_null(lambda xx, _y: within(xx), oof, y, emb, 300, seed)
    return dict(spearman_within=within(oof),
                spearman_folds=[spearman(oof[te], y[te]) for _, te in fl],
                partial_within=float(np.mean([partial_spearman(oof[te], y[te], ctrl[te]) for _, te in fl])),
                r2_fold_mean=float(np.mean(fold_r2)),
                auroc_within=float(np.nanmean([auroc(oof[te], y[te] >= np.quantile(y[te], .9)) for _, te in fl])),
                circ_p=p), oof


def run(ds, seed):
    r = Run(ds, seed)
    H, L = r.H, r.L
    emb = L + H
    sseed = derive_seed(seed, ds) if not ds.startswith("synth") else seed
    id_ix = r.idx_id
    y = r.S[id_ix]
    n = len(id_ix)
    fs = feature_sets(r, id_ix[:int(0.5 * n)], sseed)
    G = causal_score_feats(r.S, H)
    inp = input_feats(r)
    simple_names = ["last_obs", "ema", "roll_q90"]
    simple = np.column_stack([G[k] for k in simple_names] +
                             [inp[k] for k in inp])
    simple = np.where(np.isnan(simple), np.nanmean(simple[id_ix], 0), simple)
    ctrl = simple[id_ix][:, :6]
    rows = []
    res, _ = cv_metrics(simple[id_ix], y, ctrl, emb, sseed)
    rows.append(dict(dataset=ds, seed=seed, rep="simple", feats="simple", n_feat=simple.shape[1], **res))
    for rep in ("mvdf", "correlated", "random_k"):
        A = fs[rep]
        Hf, names = history_feats(A, H)
        sets = {"current": A[id_ix], "history": Hf[id_ix],
                "simple+current": np.column_stack([simple[id_ix], A[id_ix]]),
                "simple+history": np.column_stack([simple[id_ix], Hf[id_ix]])}
        for nm, F in sets.items():
            for mdl in ("gbm", "ridge"):
                res, _ = cv_metrics(F, y, ctrl, emb, sseed, mdl)
                rows.append(dict(dataset=ds, seed=seed, rep=rep, feats=nm, model=mdl,
                                 n_feat=F.shape[1], **res))
        print(" ", ds, seed, rep, flush=True)
    for rr in rows:
        rr.setdefault("model", "gbm")
    return rows


if __name__ == "__main__":
    runs = REAL if len(sys.argv) < 2 else [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    rows = []
    for ds, s in runs:
        print("WP5", ds, s, flush=True)
        rows += run(ds, s)
        suf = "" if len(sys.argv) < 2 else "_" + "_".join(a.replace(":", "") for a in sys.argv[1:])
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"history{suf}.csv"), index=False)
    print("WP5 done")
