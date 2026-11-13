"""E3: ex-ante difficulty baselines + Mondrian gap recovery. See runs/camera_ready/E3/PREREG.md.

Per (dataset, seed) writes to <out>/cells/<tag>/:
  pred.csv       one row per (split, feature_set, estimator)
  gap.csv        one row per (split, feature_set)
  delta.csv      C3.3 paired dR2 (F7 - F6) with F6's CV-selected estimator
  iso_single.csv per-single-feature isotonic (side table; not eligible as best)
  per_window.parquet
  boot.pkl       bootstrap replicates of the CV-selected rows (for pooled CIs)
Usage: python -m camera_ready.e3_exante --datasets ETTh1 ETTh2 --seeds 0 1 2 --out runs/camera_ready/E3
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch

from camera_ready.common import (Run, ALPHA, boot_r2_rho, boot_mean, ci, conf_q,
                                 phase3_split, conformal_split, spearman, r2_vs)
from camera_ready.features import build_exante, MIN_T
from camera_ready.test_leakage import check as leakage_check
from experiments._seed import derive_seed
from experiments.phase4_oracle_ceiling import _oracle_widths

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
EST = ["ols", "iso_col", "iso_ols", "gbm", "mlp", "ridge", "rf"]
SETS = ["F0", "F1", "F1p", "F2", "F3", "F4", "F5", "F6", "F7"]
EXANTE = ["F1", "F1p", "F2", "F3", "F6"]


# ------------------------------------------------------------------ estimators
def _most_corr_col(X, y):
    c = [abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 0 else 0 for j in range(X.shape[1])]
    return int(np.nanargmax(c))


def fit_predict(name, Xtr, ytr, Xtes, seed):
    """Fit one estimator on (Xtr, ytr); return predictions for each array in Xtes."""
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    if name == "ols":
        m = LinearRegression().fit(Xtr, ytr)
        return [m.predict(X) for X in Xtes]
    if name == "iso_col":
        j = _most_corr_col(Xtr, ytr)
        m = IsotonicRegression(out_of_bounds="clip").fit(Xtr[:, j], ytr)
        return [m.predict(X[:, j]) for X in Xtes]
    if name == "iso_ols":
        lr = LinearRegression().fit(Xtr, ytr)
        m = IsotonicRegression(out_of_bounds="clip").fit(lr.predict(Xtr), ytr)
        return [m.predict(lr.predict(X)) for X in Xtes]
    if name == "gbm":
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=seed).fit(Xtr, ytr)
        return [m.predict(X) for X in Xtes]
    if name == "mlp":
        from calibration.ffdci_head import train_quantile_mlp
        m = train_quantile_mlp(Xtr.astype(np.float32), ytr.astype(np.float32),
                               alpha_target=0.5, epochs=300, lr=1e-3, seed=seed)
        with torch.no_grad():
            return [m(torch.from_numpy(X.astype(np.float32))).numpy().reshape(-1) for X in Xtes]
    if name == "ridge":
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13),
                                                    cv=TimeSeriesSplit(5))).fit(Xtr, ytr)
        return [m.predict(X) for X in Xtes]
    if name == "rf":
        m = RandomForestRegressor(n_estimators=500, min_samples_leaf=20, random_state=seed,
                                  n_jobs=4).fit(Xtr, ytr)
        return [m.predict(X) for X in Xtes]
    raise ValueError(name)


def cv_select(Xtr, ytr, seed):
    """5-fold TimeSeriesSplit MSE inside the fit split; returns (best name, {name: mse})."""
    from sklearn.model_selection import TimeSeriesSplit
    sc = {}
    for name in EST:
        errs = []
        for a, b in TimeSeriesSplit(5).split(Xtr):
            p = fit_predict(name, Xtr[a], ytr[a], [Xtr[b]], seed)[0]
            errs.append(np.mean((ytr[b] - p) ** 2))
        sc[name] = float(np.mean(errs))
    return min(sc, key=sc.get), sc


# ------------------------------------------------------------------ features
def feature_sets(run, rows, fit_rows):
    """Dict set -> (matrix over `rows`, names). PCA (F4/F5) fit on fit_rows only."""
    from sklearn.decomposition import PCA
    ex = build_exante(run, rows)
    pos = {r: i for i, r in enumerate(rows)}
    fi = np.array([pos[r] for r in fit_rows])
    out = {k: v for k, v in ex.items()}
    out["F0"] = (run.z_cols(run.mvdf)[rows], [f"mvdf_{j}" for j in run.mvdf])
    H = run.hidden[rows].astype(np.float64)
    p4 = PCA(12, random_state=0).fit(H[fi])
    out["F4"] = (p4.transform(H), [f"hid_pc{i}" for i in range(12)])
    Z = run.z_dense(rows).astype(np.float64)
    p5 = PCA(12, random_state=0).fit(Z[fi])
    out["F5"] = (p5.transform(Z), [f"sae_pc{i}" for i in range(12)])
    out["F6"] = (np.column_stack([out[k][0] for k in ("F1", "F1p", "F2", "F3")]),
                 sum([out[k][1] for k in ("F1", "F1p", "F2", "F3")], []))
    out["F7"] = (np.column_stack([out["F6"][0], out["F0"][0]]), out["F6"][1] + out["F0"][1])
    return out


# ------------------------------------------------------------------ gap recovery
def mondrian(shat_cal, s_cal, shat_tests, n_bins=10):
    edges = np.unique(np.quantile(shat_cal, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b_cal = np.digitize(shat_cal, edges)
    q_glob = conf_q(s_cal)
    q = {}
    for k in range(len(edges) + 1):
        m = b_cal == k
        q[k] = conf_q(s_cal[m]) if m.sum() >= 5 else q_glob
    return [np.array([q[k] for k in np.digitize(st, edges)]) for st in shat_tests], len(edges) + 1


def run_cell(ds, seed, out, blocks=(96, 192)):
    t0 = time.time()
    run = Run(ds, seed)
    cd = os.path.join(out, "cells", run.tag)
    os.makedirs(cd, exist_ok=True)
    rng = np.random.default_rng(seed)
    cand = np.unique(np.concatenate([run.idx_id[:2], run.idx_sh[-2:], rng.choice(run.idx_id, 4)]))
    leakage_check(run, cand[cand >= MIN_T], rng)          # refuses to run on a leak
    est_seed = derive_seed(seed, ds)
    fit, ev = phase3_split(run)
    sh = run.idx_sh
    rows = np.concatenate([run.idx_id, sh])
    assert rows.min() >= MIN_T
    pos = {r: i for i, r in enumerate(rows)}
    P = lambda idx: np.array([pos[r] for r in idx])
    S = run.S
    fs = feature_sets(run, rows, fit)
    ybar_fit = S[fit].mean()
    pred_rows, delta_rows, iso_rows, gap_rows = [], [], [], []
    per_win = {"window": rows, "split": np.where(np.isin(rows, sh), "shift",
                                                  np.where(np.isin(rows, ev), "eval_ID", "fit_ID")),
               "S": S[rows]}
    boot = {}
    cvsel = {}
    for name in SETS:
        Xall = fs[name][0]
        Xf, Xe, Xs = Xall[P(fit)], Xall[P(ev)], Xall[P(sh)]
        best, cvsc = cv_select(Xf, S[fit], est_seed)
        cvsel[name] = best
        for est in EST:
            pe, ps = fit_predict(est, Xf, S[fit], [Xe, Xs], est_seed)
            pf = np.full(len(rows), np.nan); pf[P(ev)] = pe; pf[P(sh)] = ps
            per_win[f"{name}__{est}"] = pf
            for split, idx, p in (("ID", ev, pe), ("shift", sh, ps)):
                r = dict(dataset=ds, seed=seed, split=split, feature_set=name, estimator=est,
                         n_features=Xall.shape[1], n_fit=len(fit), n_eval=len(idx),
                         cv_mse=cvsc[est], cv_selected=(est == best),
                         R2_evalmean=r2_vs(S[idx], p, S[idx].mean()))
                for bl in blocks:
                    b = boot_r2_rho(S[idx], p, ybar_fit, block=bl)
                    sfx = "" if bl == 96 else f"_b{bl}"
                    for k in ("R2", "R2_lo", "R2_hi", "rho", "rho_lo", "rho_hi"):
                        if bl == 96 or k.endswith(("lo", "hi")):
                            r[k + sfx] = b[k]
                    if est == best and bl == 96:
                        boot[(split, name)] = dict(r2=b["_r2b"], rho=b["_rhob"])
                pred_rows.append(r)
    # C3.3: paired dR2, F6's CV-selected estimator on both F6 and F7
    e6 = cvsel["F6"]
    for split, idx in (("ID", ev), ("shift", sh)):
        p6 = per_win[f"F6__{e6}"][P(idx)]; p7 = per_win[f"F7__{e6}"][P(idx)]
        for bl in blocks:
            b6 = boot_r2_rho(S[idx], p6, ybar_fit, block=bl)
            b7 = boot_r2_rho(S[idx], p7, ybar_fit, block=bl)
            d = b7["_r2b"] - b6["_r2b"]
            delta_rows.append(dict(dataset=ds, seed=seed, split=split, estimator=e6, block=bl,
                                   R2_F6=b6["R2"], R2_F7=b7["R2"], dR2=b7["R2"] - b6["R2"],
                                   dR2_lo=ci(d)[0], dR2_hi=ci(d)[1]))
            if bl == 96:
                boot[(split, "dR2")] = dict(r2=d)
    # side table: single-feature isotonic
    from sklearn.isotonic import IsotonicRegression
    for name in ("F1", "F1p", "F2", "F3"):
        Xall, names = fs[name]
        for j, fn in enumerate(names):
            x = Xall[:, j]
            if np.std(x[P(fit)]) == 0:
                continue
            m = IsotonicRegression(out_of_bounds="clip").fit(x[P(fit)], S[fit])
            pe = m.predict(x[P(ev)])
            iso_rows.append(dict(dataset=ds, seed=seed, feature_set=name, feature=fn,
                                 R2=r2_vs(S[ev], pe, ybar_fit), rho=spearman(pe, S[ev])))
    # ---------------- gap recovery (50/30/20 of ID, static split conformal)
    tr, cf, te = conformal_split(run)
    fs_g = feature_sets(run, rows, tr)              # PCA refit on the 50% train block only
    q80 = conf_q(S[np.concatenate([tr, cf])])
    for split, idx in (("ID", te), ("shift", sh)):
        s = S[idx]
        hw_o, _ = _oracle_widths(s, ALPHA, 10)
        W_o = float(np.mean(2 * hw_o))
        W_n = 2 * q80
        cov_n = float(np.mean(s <= q80))
        gap_rows.append(dict(dataset=ds, seed=seed, split=split, feature_set="naive_matched",
                             estimator="-", n_cal=len(tr) + len(cf), n_test=len(idx),
                             coverage=cov_n, width=W_n, W_naive=W_n, W_oracle=W_o,
                             GRR=np.nan, coverage_valid=abs(cov_n - 0.9) <= 0.02,
                             oracle_coverage=float(np.mean(s <= hw_o))))
    gap_choice = {}
    for name in SETS:
        Xall = fs_g[name][0]
        best, cvsc = cv_select(Xall[P(tr)], S[tr], est_seed)
        gap_choice[name] = (best, cvsc[best])
        pc, pt, ps = fit_predict(best, Xall[P(tr)], S[tr],
                                 [Xall[P(cf)], Xall[P(te)], Xall[P(sh)]], est_seed)
        (hw_t, hw_s), nb = mondrian(pc, S[cf], [pt, ps])
        for split, idx, hw in (("ID", te, hw_t), ("shift", sh, hw_s)):
            s = S[idx]
            hw_o, _ = _oracle_widths(s, ALPHA, 10)
            W_o = float(np.mean(2 * hw_o)); W_n = 2 * q80
            cov = float(np.mean(s <= hw)); W = float(np.mean(2 * hw))
            red, rlo, rhi, _ = boot_mean((W_n - 2 * hw) / W_n)
            gap_rows.append(dict(dataset=ds, seed=seed, split=split, feature_set=name,
                                 estimator=best, n_cal=len(cf), n_test=len(idx), n_bins=nb,
                                 coverage=cov, width=W, W_naive=W_n, W_oracle=W_o,
                                 GRR=(W_n - W) / (W_n - W_o) if W_n > W_o else np.nan,
                                 grr_denominator_positive=bool(W_n > W_o),
                                 coverage_valid=abs(cov - 0.9) <= 0.02,
                                 width_reduction=red, width_reduction_lo=rlo,
                                 width_reduction_hi=rhi, cv_mse=cvsc[best]))
    # best ex-ante set for C3.4 = lowest CV MSE among EXANTE (chosen before test)
    best_ex = min(EXANTE, key=lambda k: gap_choice[k][1])
    gdf = pd.DataFrame(gap_rows)
    gdf["is_best_exante"] = gdf.feature_set == best_ex
    pd.DataFrame(pred_rows).to_csv(os.path.join(cd, "pred.csv"), index=False)
    gdf.to_csv(os.path.join(cd, "gap.csv"), index=False)
    pd.DataFrame(delta_rows).to_csv(os.path.join(cd, "delta.csv"), index=False)
    pd.DataFrame(iso_rows).to_csv(os.path.join(cd, "iso_single.csv"), index=False)
    pw = pd.DataFrame(per_win); pw.insert(0, "seed", seed); pw.insert(0, "dataset", ds)
    pw.to_parquet(os.path.join(cd, "per_window.parquet"), index=False)
    pickle.dump(boot, open(os.path.join(cd, "boot.pkl"), "wb"))
    pr = pd.DataFrame(pred_rows)
    sel = pr[(pr.split == "ID") & pr.cv_selected]
    print(f"[{run.tag}] {time.time()-t0:.0f}s  CV-selected ID R2/rho: " +
          "  ".join(f"{r.feature_set}:{r.R2:+.2f}/{r.rho:+.2f}" for r in sel.itertuples()), flush=True)
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs/camera_ready/E3")
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()
    for ds in a.datasets:
        for s in a.seeds:
            if a.skip_done and os.path.exists(os.path.join(a.out, "cells", f"{ds}_seed{s}", "boot.pkl")):
                continue
            run_cell(ds, s, a.out)


if __name__ == "__main__":
    main()
