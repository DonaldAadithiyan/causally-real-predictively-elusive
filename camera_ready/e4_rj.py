"""E4: robustness of R_j to the residualisation. See runs/camera_ready/E4/PREREG.md.

Inputs: E5a per-cell features.csv + curves.json (recomputed, reproduction-gated), and the
v2_runs Kaggle single-seed candidate tables (selection/enrichment only).
  python -m camera_ready.e4_rj --datasets ETTh1 ETTh2 electricity traffic --seeds 0 1 2 ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from camera_ready.common import (Run, HERE, CR, enrichment_stats, jaccard, spearman,
                                 available_seeds)
from camera_ready.phase34 import phase3
from validation.difficulty_specificity import feature_type_table, residualize, _curve_monotone

warnings.filterwarnings("ignore")
DEFS = ["isotonic", "binned", "spline", "linear", "rank", "none"]
OUT = os.path.join(CR, "E4")


def _iso_best(x, y):
    best, bsse = None, np.inf
    for inc in (True, False):
        p = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(x, y).predict(x)
        sse = np.sum((y - p) ** 2)
        if sse < bsse:
            bsse, best = sse, p
    return best


def resid(P, U, how):
    P, U = np.asarray(P, float), np.asarray(U, float)
    ok = np.isfinite(P) & np.isfinite(U)
    R = np.full_like(P, np.nan)
    p, u = P[ok], U[ok]
    if how == "isotonic":
        R = residualize(P, U, method="isotonic", provenance_p="interventional_dose_response",
                        provenance_u="interventional_dose_response")
        return R
    if how == "binned":
        rk = stats.rankdata(p, method="ordinal") - 1
        b = (rk * 10 // len(p)).astype(int)
        m = np.array([u[b == k].mean() for k in range(10)])
        R[ok] = u - m[b]
    elif how == "spline":
        from scipy.interpolate import make_smoothing_spline
        # z-score both axes for numerical conditioning (P ~ 1e-5); GCV re-picks lam, so the
        # fitted smoother is unchanged up to the lam reparametrisation.
        px, py = (p - p.mean()) / p.std(), (u - u.mean()) / u.std()
        o = np.argsort(px)
        xs, ys = px[o], py[o]
        ux, inv = np.unique(xs, return_inverse=True)       # average ties (strictly increasing x)
        uy = np.bincount(inv, ys) / np.bincount(inv)
        w = np.bincount(inv).astype(float)
        spl = make_smoothing_spline(ux, uy, w=w, lam=None)  # lam by GCV
        R[ok] = u - (spl(px) * u.std() + u.mean())
    elif how == "linear":
        A = np.vstack([p, np.ones_like(p)]).T
        R[ok] = u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
    elif how == "rank":
        rp, ru = stats.rankdata(p), stats.rankdata(u)
        R[ok] = ru - _iso_best(rp, ru)
    elif how == "none":
        R[ok] = u
    return R


def select(P, U, mono, pool, R, k=12):
    gate1 = (np.abs(U) >= 0.0) & mono & np.isfinite(R)
    cand = np.where(gate1)[0]
    cand = cand[np.argsort(-R[cand])]
    return pool[cand[:k]]


def cell_from_e5a(ds, seed):
    d = os.path.join(CR, "E5", "cells", f"{ds}_seed{seed}")
    f = pd.read_csv(os.path.join(d, "features.csv"))
    cv = json.load(open(os.path.join(d, "curves.json")))
    lam = cv["lambdas"]
    mono = np.array([_curve_monotone({l: cv["curves"][str(int(j))]["D_unc"][i] for i, l in enumerate(lam)})
                     for j in f.feature])
    return f.feature.to_numpy(int), f.P_j.to_numpy(float), f.U_j.to_numpy(float), mono


def cell_from_v2(ds):
    d = os.path.join(HERE, "v2_runs", "v2_phase1", f"{ds}_pre_head")
    c = pd.read_csv(os.path.join(d, "candidates.csv"))
    cv = json.load(open(os.path.join(d, "dose_curves.json")))["unc_curves"]
    mono = np.array([_curve_monotone({float(l): v for l, v in cv[str(int(j))].items()})
                     for j in c.feature])
    stored = json.load(open(os.path.join(d, "mvdf_v2_selection.json")))
    return c.feature.to_numpy(int), c.P_j.to_numpy(float), c.U_j.to_numpy(float), mono, stored


def analyse(tag, ds, seed, pool, P, U, mono, run=None, stored_mvdf=None):
    ftab = feature_type_table(P, U, feature_indices=pool)
    R_iso = resid(P, U, "isotonic")
    mv_iso = select(P, U, mono, pool, R_iso)
    rows = []
    for how in DEFS:
        R = resid(P, U, how)
        mv = select(P, U, mono, pool, R)
        e = enrichment_stats(mv, ftab, pool)
        ok = np.isfinite(R) & np.isfinite(R_iso)
        r = dict(dataset=ds, seed=seed, source=tag, definition=how,
                 rho_vs_iso=spearman(R[ok], R_iso[ok]), jaccard_vs_iso=jaccard(mv, mv_iso),
                 mvdf=" ".join(str(int(x)) for x in sorted(mv)), **e)
        if how == "isotonic" and stored_mvdf is not None:
            r["iso_reproduces_stored_mvdf"] = sorted(int(x) for x in mv) == sorted(int(x) for x in stored_mvdf)
        if run is not None:
            run.mvdf = np.asarray(mv, int)
            p3, _ = phase3(run)
            r.update(n_fit=p3["n_fit"], n_eval=p3["n_eval"],
                     marg_best_r2=p3["marg_best_r2"], marg_best_r2_lo=p3["marg_best_r2_lo"],
                     marg_best_r2_hi=p3["marg_best_r2_hi"], cond_best_r2=p3["cond_best_r2"],
                     cond_best_r2_lo=p3["cond_best_r2_lo"], cond_best_r2_hi=p3["cond_best_r2_hi"])
            for b in ("marg", "cond"):
                v, lo = r[f"{b}_best_r2"], r[f"{b}_best_r2_lo"]
                r[f"{b}_null_holds"] = bool((not np.isfinite(v)) or v <= 0 or lo <= 0)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "electricity", "traffic"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--no_v2", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = []
    for ds in a.datasets:
        seeds = a.seeds if a.seeds is not None else available_seeds(ds)
        for s in seeds:
            if not os.path.exists(os.path.join(CR, "E5", "cells", f"{ds}_seed{s}", "features.csv")):
                print(f"[E4] no E5a output for {ds} s{s}; skipping", flush=True)
                continue
            pool, P, U, mono = cell_from_e5a(ds, s)
            run = Run(ds, s, load_X=False)
            stored = run.mvdf.copy()
            rr = analyse("local", ds, s, pool, P, U, mono, run=run, stored_mvdf=stored)
            rows += rr
            print(f"[E4 {ds} s{s}] " + "  ".join(f"{r['definition']}:J={r['jaccard_vs_iso']:.2f},"
                                                  f"E={r['enrichment']:.2f}" for r in rr), flush=True)
    if not a.no_v2:
        for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "solar"]:
            pool, P, U, mono, st = cell_from_v2(ds)
            rows += analyse("v2_kaggle", ds, 0, pool, P, U, mono, stored_mvdf=st["MVDF_v2"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "AGG_E4_rj_robustness.csv"), index=False)
    print(f"E4 done {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
