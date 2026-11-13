"""Work item A — is the negative held-out R^2 a real null? (brief §6)

Uses the regenerated arrays (runs/p126/arrays). Outputs runs/p126/A/.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from p126.common import (Run, REAL, SYN, OUT, boot_ci, circ_null, spearman, kendall,  # noqa
                         auroc, dump)
from experiments._seed import derive_seed  # noqa: E402

AOUT = os.path.join(OUT, "A")
os.makedirs(AOUT, exist_ok=True)
torch.set_num_threads(4)
NB = 2000


# ------------------------------------------------------------------ estimators
def fit_predict(Ftr, ytr, Fte, seed, which=("linear", "isotonic", "gbm", "mlp")):
    """Same four estimators and hyper-parameters as
    experiments/phase1_synthetic_control._fit_eval_estimators, but returning predictions."""
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from calibration.ffdci_head import train_quantile_mlp
    out = {}
    if "linear" in which:
        out["linear"] = LinearRegression().fit(Ftr, ytr).predict(Fte)
    if "isotonic" in which:
        if Ftr.shape[1] == 1:
            j = 0
        else:
            cs = [abs(np.corrcoef(Ftr[:, j], ytr)[0, 1]) if np.std(Ftr[:, j]) > 0 else 0
                  for j in range(Ftr.shape[1])]
            j = int(np.argmax(cs))
        out["isotonic"] = IsotonicRegression(out_of_bounds="clip").fit(Ftr[:, j], ytr).predict(Fte[:, j])
    if "gbm" in which:
        out["gbm"] = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, random_state=seed).fit(Ftr, ytr).predict(Fte)
    if "mlp" in which:
        m = train_quantile_mlp(Ftr.astype(np.float32), ytr.astype(np.float32), alpha_target=0.5,
                               epochs=300, lr=1e-3, seed=seed)
        with torch.no_grad():
            out["mlp"] = m(torch.from_numpy(Fte.astype(np.float32))).numpy().astype(float)
    return out


def r2_decomp(pred, y_eval, y_fit):
    sse = float(np.sum((y_eval - pred) ** 2))
    tot_eval = float(np.sum((y_eval - y_eval.mean()) ** 2))
    tot_fit = float(np.sum((y_eval - y_fit.mean()) ** 2))
    mse_const = tot_fit / len(y_eval)
    return dict(R2_test=1 - sse / tot_eval, R2_train=1 - sse / tot_fit,
                R2_const=1 - tot_fit / tot_eval, skill=1 - (sse / len(y_eval)) / mse_const)


def dist_stats(s):
    return dict(mean=float(np.mean(s)), median=float(np.median(s)), sd=float(np.std(s)),
                q50=float(np.quantile(s, .5)), q90=float(np.quantile(s, .9)), n=int(len(s)))


def rank_metrics(x, S, thr_eval, thr_fit, block, emb, seed, active=None, do_ci=True):
    """Shift-robust metrics of predictor x against S on a contiguous segment. If `active`
    is given, metrics are computed on that subset (resampling/shifting is still done on the
    full contiguous segment, then restricted)."""
    n = len(S)
    act = np.ones(n, bool) if active is None else active

    def sp(ix=None, xx=None):
        ix = np.arange(n) if ix is None else ix
        xx = x if xx is None else xx
        m = act[ix]
        return spearman(xx[ix][m], S[ix][m])

    def au(ix=None, xx=None, thr=thr_eval):
        ix = np.arange(n) if ix is None else ix
        xx = x if xx is None else xx
        m = act[ix]
        return auroc(xx[ix][m], S[ix][m] >= thr)

    res = dict(n=int(act.sum()), spearman=sp(), kendall=kendall(x[act], S[act]),
               auroc_top10_evalthr=au(), auroc_top10_fitthr=au(thr=thr_fit))
    if do_ci:
        lo, hi, nbk = boot_ci(lambda ix: sp(ix), n, block, NB, seed)
        res.update(spearman_ci=[lo, hi], n_blocks=nbk)
        lo, hi, _ = boot_ci(lambda ix: au(ix), n, block, NB, seed + 1)
        res.update(auroc_ci=[lo, hi])
        _, p = circ_null(lambda xx, yy: sp(None, xx), x, S, emb, 1000, seed)
        res["spearman_p_circ"] = p
        _, p = circ_null(lambda xx, yy: au(None, xx) - 0.5, x, S, emb, 1000, seed + 2)
        res["auroc_p_circ"] = p
    return res


def act_split_stats(a, S):
    on = a != 0
    if on.sum() < 5 or (~on).sum() < 5:
        return dict(n_active=int(on.sum()))
    return dict(n_active=int(on.sum()),
                d_median=float(np.median(S[on]) - np.median(S[~on])),
                d_q90=float(np.quantile(S[on], .9) - np.quantile(S[~on], .9)))


# ------------------------------------------------------------------ blocked CV
def blocked_cv_oof(F, y, K, emb, seed, which):
    n = len(y)
    edges = np.linspace(0, n, K + 1).astype(int)
    oof = {w: np.full(n, np.nan) for w in which}
    for k in range(K):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.where((np.arange(n) < edges[k] - emb) | (np.arange(n) >= edges[k + 1] + emb))[0]
        p = fit_predict(F[tr], y[tr], F[te], seed, which)
        for w in which:
            oof[w][te] = p[w]
    return oof, edges


def feature_sets(r, fit_ix, seed):
    """MVDF + matched-budget comparison representations (k = |MVDF|)."""
    from sklearn.decomposition import PCA
    k = len(r.mvdf)
    # random: k SAE features active on >=2% of fit windows (same rule as phase3_scorer_h6)
    act_rate = np.zeros(int(r.d_sae))
    zi = r.z_idx[fit_ix]
    for c in np.unique(zi[zi >= 0]):
        act_rate[c] = (zi == c).any(1).mean()
    elig = np.where(act_rate >= 0.02)[0]
    rng = np.random.default_rng(seed + 777)
    rand = np.sort(rng.choice(elig, size=min(k, len(elig)), replace=False))
    if hasattr(r, "random") and len(getattr(r, "random", [])) == k:
        rand = np.asarray(r.random, int)          # synthetic run saved its random set
    pca = PCA(n_components=k, random_state=0).fit(r.hidden[fit_ix].astype(np.float64))
    return {
        "mvdf": r.z_cols(r.mvdf),
        "correlated": r.z_cols(r.correlated),
        "random_k": r.z_cols(rand),
        "pca_k": pca.transform(r.hidden.astype(np.float64)),
        "dense": r.hidden.astype(np.float64),
    }


# ------------------------------------------------------------------ per run
def run(ds, seed):
    r = Run(ds, seed)
    sseed = derive_seed(seed, ds) if not ds.startswith("synth") else seed
    H, L = r.H, r.L
    emb = L + H
    block = 2 * H
    id_ix = r.idx_id
    n = len(id_ix); cpt = int(0.6 * n)
    fit_ix, ev_ix = id_ix[:cpt], id_ix[cpt:]
    S_orig, _ = r.S_local(id_ix)          # the original code's S_t (ID-block profile)
    S_cal = r.S[id_ix]                    # E_select profile
    fs = feature_sets(r, fit_ix, sseed)
    F = fs["mvdf"][id_ix]
    active = (F != 0).any(1)
    out = dict(dataset=ds, seed=seed, n_id=n, n_fit=cpt, n_eval=n - cpt,
               corr_S_orig_S_cal=spearman(S_orig, S_cal),
               mvdf_any_active_rate_id=float(active.mean()),
               mvdf_per_feature_active_rate_id=float((F != 0).mean()))
    rows_dec, rows_rank = [], []

    # --- task 1+2: reproduce + decompose, both S definitions, all/active subsets
    for sname, S in (("S_orig", S_orig), ("S_cal", S_cal)):
        yf, ye = S[:cpt], S[cpt:]
        out[f"{sname}_dist_fit"] = dist_stats(yf)
        out[f"{sname}_dist_eval"] = dist_stats(ye)
        out[f"{sname}_dist_fit_active"] = dist_stats(yf[active[:cpt]])
        out[f"{sname}_dist_eval_active"] = dist_stats(ye[active[cpt:]])
        for subset in ("all", "active"):
            if subset == "all":
                mtr, mte = np.ones(cpt, bool), np.ones(n - cpt, bool)
            else:
                mtr, mte = active[:cpt], active[cpt:]
            preds = fit_predict(F[:cpt][mtr], yf[mtr], F[cpt:][mte], sseed)
            for est, p in preds.items():
                d = r2_decomp(p, ye[mte], yf[mtr])
                # share of squared error from active vs inactive eval windows (all-subset only)
                if subset == "all":
                    se = (ye - p) ** 2
                    d["sse_share_active"] = float(se[active[cpt:]].sum() / se.sum())
                    d["eval_active_frac"] = float(active[cpt:].mean())
                rows_dec.append(dict(dataset=ds, seed=seed, S=sname, subset=subset, estimator=est,
                                     n_fit=int(mtr.sum()), n_eval=int(mte.sum()), **d))
                # task 3 on primary target only
                if sname == "S_orig":
                    thr_e = np.quantile(ye, .9); thr_f = np.quantile(yf, .9)
                    full = np.full(n - cpt, np.nan); full[mte] = p
                    x = np.where(np.isnan(full), 0.0, full)
                    rm = rank_metrics(x, ye, thr_e, thr_f, block, emb, sseed,
                                      active=None if subset == "all" else active[cpt:])
                    rows_rank.append(dict(dataset=ds, seed=seed, predictor=f"est:{est}",
                                          subset=subset, **rm))
    # single-feature raw activations (S_orig, eval segment)
    ye, yf = S_orig[cpt:], S_orig[:cpt]
    thr_e, thr_f = np.quantile(ye, .9), np.quantile(yf, .9)
    for j, fid in enumerate(r.mvdf):
        a = F[cpt:, j]
        for subset in ("all", "active"):
            rm = rank_metrics(a, ye, thr_e, thr_f, block, emb, sseed + j,
                              active=None if subset == "all" else active[cpt:],
                              do_ci=(subset == "all"))
            rm.update(act_split_stats(a if subset == "all" else a[active[cpt:]],
                                      ye if subset == "all" else ye[active[cpt:]]))
            rows_rank.append(dict(dataset=ds, seed=seed, predictor=f"feat:{int(fid)}",
                                  subset=subset, **rm))
    # block-length sensitivity for the GBM Spearman
    pg = fit_predict(F[:cpt], yf, F[cpt:], sseed, ("gbm",))["gbm"]
    out["gbm_spearman_ci_by_block"] = {
        str(b): boot_ci(lambda ix: spearman(pg[ix], ye[ix]), len(ye), b, NB, sseed)[:2]
        for b in (H, 2 * H, 4 * H)}

    # --- task 4 + 5: blocked 5-fold CV inside ID (embargo L+H), targets x representations
    S_hist = r.S                               # all windows, E_select profile
    # causal baseline: trailing mean of the last H fully-observed scores (window t-H and older)
    W = H
    cs = np.concatenate([[0.0], np.cumsum(S_hist)])
    base = np.full(len(S_hist), np.nan)
    for t in range(H + W - 1, len(S_hist)):
        lo_, hi_ = t - H - W + 1, t - H + 1
        base[t] = (cs[hi_] - cs[lo_]) / W
    targets = {"raw": S_cal, "log": np.log(S_cal),
               "minus_causal_baseline": S_cal - base[id_ix]}
    cv_rows = []
    for rep_name, Fall in fs.items():
        Fr = Fall[id_ix]
        which = ("linear", "isotonic", "gbm", "mlp") if rep_name in ("mvdf",) else ("linear", "gbm")
        if rep_name == "dense":
            which = ("gbm",)
        for tname, yt in targets.items():
            if rep_name != "mvdf" and tname != "raw":
                continue
            ok = np.isfinite(yt)
            oof, edges = blocked_cv_oof(Fr[ok], yt[ok], 5, emb, sseed, which)
            for est, p in oof.items():
                yy = yt[ok]
                # pooled OOF R^2 (ID mean) + mean of per-fold R^2 (fold-eval-mean denominators)
                pooled = 1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2)
                fold_r2 = []
                for k in range(5):
                    sl = slice(edges[k], edges[k + 1])
                    fold_r2.append(1 - np.sum((yy[sl] - p[sl]) ** 2) /
                                   np.sum((yy[sl] - yy[sl].mean()) ** 2))
                thr = np.quantile(yy, .9)
                rho = spearman(p, yy)   # pooled: contaminated by leave-fold-out level effect

                def within(xx):
                    # mean within-fold Spearman: immune to fold-level differences
                    return float(np.mean([spearman(xx[edges[k]:edges[k + 1]],
                                                   yy[edges[k]:edges[k + 1]]) for k in range(5)]))
                rho_w = within(p)
                fold_rho = [spearman(p[edges[k]:edges[k + 1]], yy[edges[k]:edges[k + 1]])
                            for k in range(5)]
                _, pc = circ_null(lambda xx, _y: within(xx), p, yy, emb, 1000, sseed)
                au_w = float(np.nanmean([auroc(p[edges[k]:edges[k + 1]],
                                               yy[edges[k]:edges[k + 1]] >=
                                               np.quantile(yy[edges[k]:edges[k + 1]], .9))
                                         for k in range(5)]))
                lo, hi = np.nan, np.nan
                cv_rows.append(dict(dataset=ds, seed=seed, representation=rep_name, target=tname,
                                    estimator=est, n=int(ok.sum()), r2_pooled=float(pooled),
                                    r2_fold_mean=float(np.mean(fold_r2)),
                                    r2_fold_min=float(np.min(fold_r2)),
                                    r2_fold_max=float(np.max(fold_r2)),
                                    spearman_pooled=rho, spearman_within=rho_w,
                                    spearman_folds=fold_rho, spearman_within_p_circ=pc,
                                    auroc_top10_within=au_w,
                                    auroc_top10_pooled=auroc(p, yy >= thr)))
    # save raw arrays used
    np.savez_compressed(os.path.join(AOUT, f"{r.tag}_arrays.npz"), id_ix=id_ix, S_orig=S_orig,
                        S_cal=S_cal, mvdf_act=F, active=active, causal_base=base[id_ix],
                        gbm_eval_pred=pg)
    return out, rows_dec, rows_rank, cv_rows


def feature_level(ds, seed):
    """Task 7: R_j vs predictiveness (|Spearman|, AUROC) across the 120-feature pool."""
    r = Run(ds, seed)
    from p126.common import HERE
    cand = pd.read_csv(os.path.join(HERE, f"runs/phase3_real/{r.tag}/rj_vs_predictive.csv"))
    id_ix = r.idx_id
    S = r.S[id_ix]
    Z = r.z_cols(cand.feature.values)[id_ix]
    thr = np.quantile(S, .9)
    cand["abs_spearman"] = [abs(spearman(Z[:, i], S)) for i in range(Z.shape[1])]
    cand["auroc_top10"] = [auroc(Z[:, i], S >= thr) for i in range(Z.shape[1])]
    out = {}
    rng = np.random.default_rng(seed)
    for col in ("abs_spearman", "auroc_top10", "best_natural_predictive_r2"):
        ok = cand[col].notna() & cand.R_j.notna()
        x, y = cand.R_j[ok].values, cand[col][ok].values
        rho = spearman(x, y)
        bs = []
        for _ in range(NB):
            ix = rng.integers(0, len(x), len(x))
            bs.append(spearman(x[ix], y[ix]))
        out[col] = dict(rho=rho, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        n_features=int(ok.sum()))
    out["pool_max_abs_spearman"] = float(cand.abs_spearman.max())
    out["mvdf_mean_abs_spearman"] = float(cand[cand.in_mvdf].abs_spearman.mean())
    out["pool_mean_abs_spearman"] = float(cand.abs_spearman.mean())
    cand.to_csv(os.path.join(AOUT, f"{r.tag}_feature_level.csv"), index=False)
    return dict(dataset=ds, seed=seed, **out)


if __name__ == "__main__":
    runs = REAL
    if len(sys.argv) > 1:
        runs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    summ, dec, rank, cv, fl = [], [], [], [], []
    for ds, s in runs:
        if not os.path.exists(os.path.join(OUT, "arrays", f"{ds}_seed{s}.npz")):
            print("missing", ds, s); continue
        print("A:", ds, s, flush=True)
        o, d, rk, c = run(ds, s)
        summ.append(o); dec += d; rank += rk; cv += c
        if not ds.startswith("synth"):
            fl.append(feature_level(ds, s))
    suffix = "" if len(sys.argv) == 1 else "_" + "_".join(a.replace(":", "") for a in sys.argv[1:])
    dump(summ, os.path.join(AOUT, f"summary{suffix}.json"))
    pd.DataFrame(dec).to_csv(os.path.join(AOUT, f"r2_decomposition{suffix}.csv"), index=False)
    pd.DataFrame(rank).to_csv(os.path.join(AOUT, f"rank_metrics{suffix}.csv"), index=False)
    pd.DataFrame(cv).to_csv(os.path.join(AOUT, f"blocked_cv{suffix}.csv"), index=False)
    if fl:
        dump(fl, os.path.join(AOUT, f"feature_level{suffix}.json"))
    print("A done")
