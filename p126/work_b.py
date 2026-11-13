"""Work item B — a ceiling for PREDICTABLE difficulty (brief §7). Outputs runs/p126/B/."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from p126.common import (Run, REAL, SYN, OUT, ALPHA, boot_ci, spearman, acf, conf_q,  # noqa
                         matched_width, dump, block_bootstrap)
from experiments._seed import derive_seed  # noqa: E402
from experiments.phase4_oracle_ceiling import _oracle_widths  # noqa: E402  (unchanged)
from calibration.aci import empirical_quantile  # noqa: E402

BOUT = os.path.join(OUT, "B")
os.makedirs(BOUT, exist_ok=True)
NB = 2000


# ------------------------------------------------------------------ task 1
def oracle_ratio(s, alpha=ALPHA, nb=10):
    """Existing Phase-4 oracle, unchanged: LOO per-true-decile quantile vs block quantile."""
    qn = empirical_quantile(s, 1 - alpha)
    hw, _ = _oracle_widths(s, alpha, nb)
    return float((2 * qn - np.mean(2 * hw)) / (2 * qn)), float(np.mean(s <= hw))


def task1():
    rows = []
    for ds, sd in REAL:
        r = Run(ds, sd)
        for blk, ix in (("ID", r.idx_id), ("shift", r.idx_sh)):
            s_orig, _ = r.S_local(ix)
            rr, cov = oracle_ratio(s_orig)
            perm = np.random.default_rng(sd).permutation(s_orig)
            rp, _ = oracle_ratio(perm)
            rows.append(dict(case="real", dataset=ds, seed=sd, block=blk, n=len(ix),
                             oracle_ratio=rr, oracle_cov=cov, oracle_ratio_permuted=rp,
                             identical=bool(abs(rr - rp) < 1e-12)))
    rng = np.random.default_rng(0)
    for name, gen in (("abs_normal", lambda n: np.abs(rng.standard_normal(n))),
                      ("exponential", lambda n: rng.exponential(1.0, n)),
                      ("lognormal_sd0.4", lambda n: rng.lognormal(0, 0.4, n))):
        vals = [oracle_ratio(gen(1966))[0] for _ in range(50)]
        rows.append(dict(case=f"iid_{name}", n=1966, oracle_ratio=float(np.mean(vals)),
                         oracle_ratio_sd=float(np.std(vals))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BOUT, "task1_oracle_property.csv"), index=False)
    return df


# ------------------------------------------------------------------ task 2
def task2():
    rows = []
    for ds, sd in REAL:
        r = Run(ds, sd)
        H = r.H
        for blk, ix in (("select", r.idx_sel), ("ID", r.idx_id), ("shift", r.idx_sh)):
            for tname, x in (("S", r.S[ix]), ("logS", np.log(r.S[ix]))):
                lags = [H, int(1.5 * H), 2 * H, 3 * H, 4 * H]
                lags = [l for l in lags if l < len(x) - 10]
                a = acf(x, max(lags))
                for l in lags:
                    lo, hi, nbk = boot_ci(lambda ii: acf(x[ii], l)[l], len(x), 2 * H, 500, sd)
                    rows.append(dict(dataset=ds, seed=sd, block=blk, target=tname, lag=l,
                                     lag_over_H=l / H, acf=float(a[l]), ci_lo=lo, ci_hi=hi,
                                     n=len(x), n_blocks=nbk))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BOUT, "task2_acf.csv"), index=False)
    return df


# ------------------------------------------------------------------ predictors
def causal_score_feats(S, H):
    """Past-score predictors. At window t only windows <= t-H are fully observed."""
    n = len(S)
    last = np.full(n, np.nan); last[H:] = S[:-H]
    ema = np.full(n, np.nan)
    a = 2.0 / (H + 1)
    e = np.nan
    for t in range(n):
        if t - H >= 0:
            v = S[t - H]
            e = v if np.isnan(e) else a * v + (1 - a) * e
            ema[t] = e
    rq = np.full(n, np.nan)
    W = 2 * H
    for t in range(H + W - 1, n):
        rq[t] = np.quantile(S[t - H - W + 1:t - H + 1], 0.9)
    return dict(last_obs=last, ema=ema, roll_q90=rq)


def input_feats(r):
    tv = int(r.target_var)
    x = r.X[:, :, tv].astype(np.float64)
    dx = np.diff(x, axis=1)
    yh = r.yhat.astype(np.float64)
    f = dict(in_std=x.std(1), in_diff_std=dx.std(1), in_range=x.max(1) - x.min(1),
             in_level=x.mean(1), in_last=x[:, -1],
             in_allch_std=r.X.astype(np.float64).std(1).mean(1),
             fc_mag=np.abs(yh).mean(1), fc_rough=np.diff(yh, axis=1).std(1),
             fc_range=yh.max(1) - yh.min(1))
    if hasattr(r, "origin_ts"):
        ts = r.origin_ts.astype(np.int64)
        f["hour"] = ((ts // 3600) % 24).astype(float)
        f["dow"] = (((ts // 86400) + 4) % 7).astype(float)
    return f


def knn_predict(Xq, tq, Xp, tp, yp, k, emb):
    """kNN regression excluding pool points within `emb` windows of the query in time."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(len(Xp), k + 2 * emb + 5)).fit(Xp)
    _, ind = nn.kneighbors(Xq)
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        cand = ind[i][np.abs(tp[ind[i]] - tq[i]) > emb][:k]
        out[i] = np.mean(yp[cand]) if len(cand) else np.mean(yp)
    return out


def gbm_fit_predict(Ftr, ytr, Fall, seed, inner_emb):
    """GBM on log S; (depth, n_estimators) chosen on an embargoed inner validation split
    of the FIT data only."""
    from sklearn.ensemble import GradientBoostingRegressor
    n = len(ytr)
    cut = int(0.7 * n)
    itr, iva = np.arange(0, cut - inner_emb), np.arange(cut, n)
    if len(itr) < 100 or len(iva) < 50:
        # round 2: fitting segment too short for an embargoed inner split (traffic):
        # use fixed defaults, no selection
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=seed).fit(Ftr, ytr)
        return m.predict(Fall), ("default_no_inner_split", 200)
    best, bl = None, np.inf
    for d in (2, 3):
        for ne in (100, 300):
            m = GradientBoostingRegressor(n_estimators=ne, max_depth=d, learning_rate=0.05,
                                          subsample=0.8, random_state=seed)
            m.fit(Ftr[itr], ytr[itr])
            l = np.mean((m.predict(Ftr[iva]) - ytr[iva]) ** 2)
            if l < bl:
                bl, best = l, (d, ne)
    m = GradientBoostingRegressor(n_estimators=best[1], max_depth=best[0], learning_rate=0.05,
                                  subsample=0.8, random_state=seed).fit(Ftr, ytr)
    return m.predict(Fall), best


# ------------------------------------------------------------------ interval constructions
def binned_hw(g_cal, S_cal, g_te, K, alpha):
    edges = np.quantile(g_cal, np.linspace(0, 1, K + 1)[1:-1])
    bc, bt = np.digitize(g_cal, edges), np.digitize(g_te, edges)
    qs = np.array([conf_q(S_cal[bc == k], 1 - alpha) if (bc == k).sum() else np.inf
                   for k in range(K)])
    return qs[bt]


def normalized_hw(g_cal, S_cal, g_te, alpha):
    return conf_q(S_cal / g_cal, 1 - alpha) * g_te


def evaluate(g_fit, g_cal, g_te, S_fit, S_cal, S_te, base_hw, block, seed):
    """All B metrics for one predictor on one test segment. base_hw = naive_matched const."""
    base_cov = float(np.mean(S_te <= base_hw))
    W_b = 2 * base_hw
    out = dict(spearman=spearman(g_te, S_te))
    # skill: linear calibration of g on cal, vs constant mean(S fit u cal)
    A = np.vstack([np.ones_like(g_cal), g_cal]).T
    coef = np.linalg.lstsq(A, S_cal, rcond=None)[0]
    pred = coef[0] + coef[1] * g_te
    const = np.mean(np.concatenate([S_fit, S_cal]))
    out["skill"] = float(1 - np.mean((S_te - pred) ** 2) / np.mean((S_te - const) ** 2))
    n = len(S_te)
    for cname, fn in (("bin5", lambda al: binned_hw(g_cal, S_cal, g_te, 5, al)),
                      ("bin10", lambda al: binned_hw(g_cal, S_cal, g_te, 10, al)),
                      ("norm", lambda al: normalized_hw(g_cal, S_cal, g_te, al))):
        if cname == "norm" and np.any(g_cal <= 0):
            continue
        hw = fn(ALPHA)
        cov = float(np.mean(S_te <= hw)); w = float(np.mean(2 * hw))
        Wm, al_m, _ = matched_width(fn, S_te, base_cov)
        red = (W_b - Wm) / W_b
        hw_m = fn(al_m)

        def red_fn(ix):
            return (W_b - np.mean(2 * hw_m[ix])) / W_b
        lo, hi, nbk = boot_ci(red_fn, n, block, NB, seed)
        # supplementary: BOTH matched to 0.90 empirical test coverage (the oracle's
        # operating point); baseline at 0.90 = 2 * empirical 0.9-quantile of S_te
        W9b = 2 * np.quantile(S_te, 0.9)
        W9m, _, _ = matched_width(fn, S_te, 0.90)
        out[f"{cname}_red_at90"] = float((W9b - W9m) / W9b)
        out[f"{cname}_cov"] = cov
        out[f"{cname}_width"] = w
        out[f"{cname}_matched_alpha"] = al_m
        out[f"{cname}_red"] = float(red)
        out[f"{cname}_red_ci"] = [lo, hi]
    out["base_cov"] = base_cov
    out["base_width"] = float(W_b)
    lo, hi, nbk = boot_ci(lambda ix: np.mean(S_te[ix] <= base_hw), n, block, NB, seed + 5)
    out["base_cov_ci"] = [lo, hi]
    out["n_test"], out["n_blocks_test"] = n, nbk
    return out


# ------------------------------------------------------------------ per run
def run(ds, seed, embargo=True):
    r = Run(ds, seed)
    sseed = derive_seed(seed, ds) if not ds.startswith("synth") else seed
    H, L = r.H, r.L
    emb = L + H if embargo else 0
    S_all = r.S
    id_ix = r.idx_id
    n = len(id_ix); c1, c2 = int(0.5 * n), int(0.8 * n)
    fit = id_ix[:c1 - (0 if not embargo else 0)]
    cal = id_ix[c1 + emb:c2]
    te_id = id_ix[c2 + emb:]
    te_sh = r.idx_sh
    seg = dict(fit=fit, cal=cal, test_ID=te_id, test_shift=te_sh)

    # ---------- predictors over all test windows -------------------------------
    G = {}
    G.update(causal_score_feats(S_all, H))
    inp = input_feats(r)
    G.update({k: v for k, v in inp.items() if k in ("in_std", "in_diff_std", "in_range",
                                                    "in_allch_std", "fc_rough", "fc_range")})
    # kNN on PCA-reduced input window (target channel), pool = fit segment
    from sklearn.decomposition import PCA
    tv = int(r.target_var)
    Xin = r.X[:, :, tv].astype(np.float64)
    pca = PCA(n_components=10, random_state=0).fit(Xin[fit])
    Z = pca.transform(Xin)
    tidx = np.arange(len(S_all))
    G["knn"] = np.exp(knn_predict(Z, tidx, Z[fit], tidx[fit], np.log(S_all[fit]), 50, L + H))

    # union GBM (non-representation)
    union_cols = ["last_obs", "ema", "roll_q90"] + list(inp.keys())
    U = np.column_stack([G[c] if c in G else inp[c] for c in union_cols])
    U = np.where(np.isnan(U), np.nanmean(U[fit], 0), U)
    pr, hp_u = gbm_fit_predict(U[fit], np.log(S_all[fit]), U, sseed, L + H)
    G["gbm_union"] = np.exp(pr)

    # representations
    from p126.work_a import feature_sets
    fs = feature_sets(r, fit, sseed)
    for name, F in fs.items():
        pr, _ = gbm_fit_predict(F[fit], np.log(S_all[fit]), F, sseed, L + H)
        G[f"rep_{name}"] = np.exp(pr)

    # select best non-representation predictor on an inner validation fold INSIDE fit
    cut = int(0.7 * len(fit))
    iva = fit[cut:]
    nonrep = [k for k in G if not k.startswith("rep_")]
    sel_scores = {}
    for k in nonrep:
        g = G[k][iva]; ok = np.isfinite(g)
        sel_scores[k] = spearman(g[ok], S_all[iva][ok]) if ok.sum() > 20 else -np.inf
    # (note: knn/gbm_union were fit on all of `fit`, incl. iva -> their inner score is
    # optimistic; refit them on fit[:cut] for an honest selection score)
    fi = fit[:max(cut - (L + H), 100)]   # round 2: floor for short ID blocks (traffic)
    pca_i = PCA(n_components=10, random_state=0).fit(Xin[fi]); Zi = pca_i.transform(Xin)
    g_knn_i = np.exp(knn_predict(Zi[iva], tidx[iva], Zi[fi], tidx[fi], np.log(S_all[fi]), 50, L + H))
    sel_scores["knn"] = spearman(g_knn_i, S_all[iva])
    pr_i, _ = gbm_fit_predict(U[fi], np.log(S_all[fi]), U[iva], sseed, L + H)
    sel_scores["gbm_union"] = spearman(pr_i, S_all[iva])
    best = max(sel_scores, key=sel_scores.get)

    # combination: best non-rep predictor + MVDF activations
    comb = np.column_stack([np.where(np.isnan(G[best]), np.nanmean(G[best][fit]), G[best]),
                            fs["mvdf"]])
    pr, _ = gbm_fit_predict(comb[fit], np.log(S_all[fit]), comb, sseed, L + H)
    G["combo_best+mvdf"] = np.exp(pr)
    # best predictor alone refit through the same GBM (fair incremental comparison)
    pr, _ = gbm_fit_predict(comb[fit][:, :1], np.log(S_all[fit]), comb[:, :1], sseed, L + H)
    G["combo_best_alone"] = np.exp(pr)

    # synthetic: true-scale ceiling predictors (not deployable, known only in the DGP)
    if ds.startswith("synth"):
        G["TRUE_mean_sigma"] = r.sigma_h.mean(1).astype(float)
        G["TRUE_regime_at_origin"] = 1.0 + r.regime_win.astype(float)

    # ---------- evaluate --------------------------------------------------------
    S_fit, S_cal = S_all[fit], S_all[cal]
    q_base = conf_q(np.concatenate([S_fit, S_cal]), 1 - ALPHA)   # naive_matched (80% budget)
    rows = []
    for tname in ("test_ID", "test_shift"):
        te = seg[tname]
        S_te = S_all[te]
        # outcome-binned oracle (existing, unchanged) on this test segment
        rr, oc = oracle_ratio(S_te)
        rows.append(dict(dataset=ds, seed=seed, segment=tname, predictor="ORACLE_outcome_binned",
                         kind="oracle", oracle_ratio=rr, oracle_cov=oc, n_test=len(te)))
        for k, g in G.items():
            gf, gc, gt = g[fit], g[cal], g[te]
            if np.any(~np.isfinite(gc)) or np.any(~np.isfinite(gt)):
                okc = np.isfinite(gc)
                gc = np.where(okc, gc, np.nanmedian(gc))
                gt = np.where(np.isfinite(gt), gt, np.nanmedian(gc))
            ev = evaluate(gf, gc, gt, S_fit, S_cal, S_te, q_base, 2 * H, sseed)
            kind = ("representation" if k.startswith("rep_") else
                    "true_scale" if k.startswith("TRUE") else
                    "combination" if k.startswith("combo") else "forecast_time")
            rows.append(dict(dataset=ds, seed=seed, segment=tname, predictor=k, kind=kind,
                             selected_best=(k == best), **ev))
    meta = dict(dataset=ds, seed=seed, embargo=emb, n_fit=len(fit), n_cal=len(cal),
                n_test_ID=len(te_id), n_test_shift=len(te_sh), best_nonrep=best,
                inner_selection_spearman=sel_scores, gbm_union_hparams=list(hp_u),
                q_base=q_base)
    np.savez_compressed(os.path.join(BOUT, f"{r.tag}_{'emb' if embargo else 'noemb'}_preds.npz"),
                        S=S_all, fit=fit, cal=cal, te_id=te_id, te_sh=te_sh,
                        **{f"g_{k}": v for k, v in G.items()})
    return rows, meta


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "t1"):
        print(task1().to_string())
    if which in ("all", "t2"):
        task2()
    if which in ("all", "main") or ":" in which:
        runs = REAL
        if ":" in which:
            runs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
        for embargo in (True, False):
            rows, metas = [], []
            for ds, s in runs:
                print("B:", ds, s, "embargo" if embargo else "no-embargo", flush=True)
                rr, m = run(ds, s, embargo)
                rows += rr; metas.append(m)
            suf = ("emb" if embargo else "noemb") + ("" if ":" not in which else
                                                      "_" + "_".join(a.replace(":", "") for a in sys.argv[1:]))
            pd.DataFrame(rows).to_csv(os.path.join(BOUT, f"main_{suf}.csv"), index=False)
            dump(metas, os.path.join(BOUT, f"meta_{suf}.json"))
    print("B done")
