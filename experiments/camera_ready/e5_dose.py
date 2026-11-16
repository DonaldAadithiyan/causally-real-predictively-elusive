"""E5: dose-response diagnostics (a), natural-range re-selection (b), interchange (c).
See results/camera_ready/E5/PREREG.md.

  python -m experiments.camera_ready.e5_dose a --datasets ETTh1 ETTh2 --seeds 0 1 2     # per-cell recompute
  python -m experiments.camera_ready.e5_dose c --datasets ETTh1 ETTh2 --seeds 0 1 2     # interchange
E5b and the aggregate tables are produced by camera_ready.e5_report from the E5a outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from experiments.camera_ready.common import (Run, HERE, CR, block_idx, ci, candidates_path, weights_dir,
                                 spearman)
from experiments.round2.selection import fast_dose, slope, load_real_models, LAM
from experiments.v2.phase3_real_predictability import CFG as P3CFG
from experiments._seed import derive_seed
from mechconf.validation.correlation_screen import screen
from mechconf.validation.difficulty_specificity import select_mvdf_v2, feature_type_table

torch.set_num_threads(4)
OUT = os.path.join(CR, "E5")


def quad_p(lams, v):
    v = np.asarray(v, float)
    A = np.vstack([np.ones_like(lams), lams, lams ** 2]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    res = v - A @ coef
    df = len(v) - 3
    s2 = (res ** 2).sum() / df
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(cov[2, 2])
    if not np.isfinite(se) or se == 0:
        return float(coef[2]), np.nan
    t = coef[2] / se
    return float(coef[2]), float(2 * stats.t.sf(abs(t), df))


def regen_pool(run, sae_n_vars=None):
    """Pool + dose windows exactly as frozen_selection draws them."""
    sel = run.idx_sel
    z_sel = run.z_dense(sel)
    scr = screen(z_sel, run.S[sel], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(derive_seed(run.seed, run.ds))
    pool = np.sort(rng.choice(elig, size=min(P3CFG["candidate_pool_size"], len(elig)), replace=False))
    cap = min(P3CFG["window_cap"], len(sel))
    wsel = np.sort(rng.choice(len(sel), size=cap, replace=False))
    return sel, z_sel, pool, wsel


def load_models(run):
    return load_real_models(weights_dir(run.tag), run.tag, run.X.shape[2], P3CFG)


def part_a(ds, seed):
    t0 = time.time()
    run = Run(ds, seed)
    od = os.path.join(OUT, "cells", run.tag); os.makedirs(od, exist_ok=True)
    model, sae = load_models(run)
    sel, z_sel, pool, wsel = regen_pool(run)
    Xw, yw, zw = run.X[sel][wsel], run.y[sel][wsel], z_sel[wsel]
    bidx = block_idx(len(wsel), 96)
    i1 = list(LAM).index(1.0)
    rows, curves = [], {}
    for j in pool:
        j = int(j)
        Dp, Du, mse_w, sc_w, na = fast_dose(model, sae, Xw, yw, zw, run.tv, j, run.prof_sel,
                                            per_window=True)
        Dp, Du = Dp[0], Du[0]
        r = dict(dataset=ds, seed=seed, feature=j, n_active=na)
        a = zw[:, j] != 0
        a_all = z_sel[:, j]
        q99_act = np.percentile(a_all[a_all != 0], 99) if (a_all != 0).any() else np.nan
        q99_all = np.percentile(a_all, 99)
        if na >= 5:
            r["P_j"], r["P_r2"] = slope(LAM, Dp)
            r["U_j"], r["U_r2"] = slope(LAM, Du)
            r["P_spearman"] = spearman(LAM, Dp); r["U_spearman"] = spearman(LAM, Du)
            r["P_quad_c"], r["P_quad_p"] = quad_p(LAM, Dp)
            r["U_quad_c"], r["U_quad_p"] = quad_p(LAM, Du)
            av = zw[a, j]
            r["frac_exceed_q99_active"] = float(np.mean(1.25 * av > q99_act))
            r["frac_exceed_q99_all"] = float(np.mean(1.25 * av > q99_all))
            # block bootstrap of mean deltas vs lambda=1 over wsel time positions
            aidx = np.where(a)[0]
            posmap = -np.ones(len(wsel), int); posmap[aidx] = np.arange(len(aidx))
            dL = mse_w[0] - mse_w[0][i1][None, :]          # (nl, n_a)
            dS = sc_w[0] - sc_w[0][i1][None, :]
            bl, bs = [], []
            for b in bidx:
                m = posmap[b]; m = m[m >= 0]
                if len(m) < 3:
                    continue
                bl.append(dL[:, m].mean(1)); bs.append(dS[:, m].mean(1))
            bl, bs = np.array(bl), np.array(bs)
            for li, lam in enumerate(LAM):
                r[f"dLoss_{lam:g}"] = float(dL[li].mean())
                r[f"dLoss_{lam:g}_lo"], r[f"dLoss_{lam:g}_hi"] = ci(bl[:, li])
                r[f"dS_{lam:g}"] = float(dS[li].mean())
                r[f"dS_{lam:g}_lo"], r[f"dS_{lam:g}_hi"] = ci(bs[:, li])
        curves[j] = dict(D_pred=[float(x) for x in Dp], D_unc=[float(x) for x in Du])
        rows.append(r)
    df = pd.DataFrame(rows)
    # ---- reproduction gate vs stored candidates + stored MVDF
    dose = dict(P_j=df.P_j.to_numpy(float), U_j=df.U_j.to_numpy(float), feature_indices=pool,
                unc_curves={int(j): {lam: curves[int(j)]["D_unc"][i] for i, lam in enumerate(LAM)}
                            for j in pool})
    mv = select_mvdf_v2(dose, u_threshold=0.0, k=12, residualize_method="isotonic")
    ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
    df["R_j"] = mv["R_j"]; df["gate1"] = mv["gate1_pass"]
    df["in_mvdf"] = df.feature.isin(mv["selected"])
    df["type"] = [next(t for t, ids in ftab["cells"].items() if int(j) in ids)
                  if np.isfinite(p) else "?" for j, p in zip(df.feature, df.P_j)]
    st = pd.read_csv(candidates_path(ds, seed)).set_index("feature")
    common = df.feature[df.feature.isin(st.index)]
    rep = dict(tag=run.tag, pool_matches_stored=bool(set(pool) == set(st.index)),
               max_abs_dP=float(np.nanmax(np.abs(df.set_index("feature").loc[common, "P_j"] - st.loc[common, "P_j"]))),
               max_abs_dU=float(np.nanmax(np.abs(df.set_index("feature").loc[common, "U_j"] - st.loc[common, "U_j"]))),
               max_rel_dU=float(np.nanmax(np.abs(df.set_index("feature").loc[common, "U_j"] - st.loc[common, "U_j"]) /
                                          (np.abs(st.loc[common, "U_j"]) + 1e-12))),
               mvdf_recomputed=sorted(int(x) for x in mv["selected"]),
               mvdf_stored=sorted(int(x) for x in run.mvdf),
               mvdf_matches=sorted(int(x) for x in mv["selected"]) == sorted(int(x) for x in run.mvdf),
               n_pool=len(pool), n_wsel=len(wsel), seconds=time.time() - t0)
    df.to_csv(os.path.join(od, "features.csv"), index=False)
    json.dump(dict(lambdas=list(LAM), curves=curves, pool=[int(p) for p in pool]),
              open(os.path.join(od, "curves.json"), "w"))
    json.dump(rep, open(os.path.join(od, "repro.json"), "w"), indent=2)
    print(f"[E5a {run.tag}] {rep['seconds']:.0f}s pool_ok={rep['pool_matches_stored']} "
          f"max|dU|={rep['max_abs_dU']:.1e} rel={rep['max_rel_dU']:.1e} mvdf_ok={rep['mvdf_matches']}",
          flush=True)


@torch.no_grad()
def _S_with(model, sae, X, y_t, tv, prof, z, j, new_val):
    zt = torch.from_numpy(z.astype(np.float32))
    rec = sae.decode(zt)
    d = sae.W_dec[:, j][None, :]
    h = rec + (torch.from_numpy(np.asarray(new_val, np.float32))[:, None] - zt[:, j][:, None]) * d
    h = sae.unstandardize(h)
    xb = torch.from_numpy(X[:, :, tv].astype(np.float32))
    mu, sd = xb.mean(1), xb.std(1) + 1e-6
    yp = (model.head(h) * sd[:, None] + mu[:, None]).numpy()
    e = yp - y_t
    return (np.abs(e) / prof[None, :]).max(1), (e ** 2).mean(1)


def part_c(ds, seed):
    t0 = time.time()
    run = Run(ds, seed)
    od = os.path.join(OUT, "cells", run.tag); os.makedirs(od, exist_ok=True)
    feats = pd.read_csv(os.path.join(od, "features.csv")).set_index("feature")
    model, sae = load_models(run)
    sel, idv = run.idx_sel, run.idx_id
    z_sel = run.z_dense(sel); z_id = run.z_dense(idv)
    rng = np.random.default_rng(0)
    rows = []
    for j in run.mvdf:
        j = int(j)
        a_sel = z_sel[:, j]; act = a_sel[a_sel != 0]
        lo_q, hi_q = np.percentile(act, [20, 80])
        top, bot = act[act >= hi_q], act[act <= lo_q]
        ev = np.where(z_id[:, j] != 0)[0]
        if len(ev) < 5:
            rows.append(dict(dataset=ds, seed=seed, feature=j, n_eval=len(ev))); continue
        X, y, z = run.X[idv][ev], run.y[idv][ev], z_id[ev]
        v_top = rng.choice(top, len(ev)); v_bot = rng.choice(bot, len(ev))
        S_t, L_t = _S_with(model, sae, X, y, run.tv, run.prof_sel, z, j, v_top)
        S_b, L_b = _S_with(model, sae, X, y, run.tv, run.prof_sel, z, j, v_bot)
        S_0, L_0 = _S_with(model, sae, X, y, run.tv, run.prof_sel, z, j, z[:, j])
        dS = S_t - S_b; dL = L_t - L_b
        bi = block_idx(len(ev), 96)
        U = float(feats.loc[j, "U_j"])
        # scaling-protocol prediction of the same contrast: U_j per unit lambda, times the
        # mean lambda-equivalent change (v_top - v_bot)/a_j
        lam_eq = np.mean((v_top - v_bot) / z[:, j])
        rows.append(dict(dataset=ds, seed=seed, feature=j, n_eval=len(ev),
                         a_top_mean=float(top.mean()), a_bot_mean=float(bot.mean()),
                         dS_top_minus_bot=float(dS.mean()),
                         dS_lo=ci(dS[bi].mean(1))[0], dS_hi=ci(dS[bi].mean(1))[1],
                         dLoss_top_minus_bot=float(dL.mean()),
                         dLoss_lo=ci(dL[bi].mean(1))[0], dLoss_hi=ci(dL[bi].mean(1))[1],
                         dS_top_vs_recon=float((S_t - S_0).mean()),
                         dS_bot_vs_recon=float((S_b - S_0).mean()),
                         U_j=U, lambda_equiv=float(lam_eq), U_implied_dS=U * float(lam_eq),
                         sign_agree=bool(np.sign(dS.mean()) == np.sign(U))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "interchange.csv"), index=False)
    print(f"[E5c {run.tag}] {time.time()-t0:.0f}s sign agreement "
          f"{df.sign_agree.mean():.2f} ({int(df.sign_agree.sum())}/{len(df)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "c"])
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()
    for ds in a.datasets:
        for s in a.seeds:
            f = os.path.join(OUT, "cells", f"{ds}_seed{s}",
                             "features.csv" if a.part == "a" else "interchange.csv")
            if a.skip_done and os.path.exists(f):
                continue
            (part_a if a.part == "a" else part_c)(ds, s)


if __name__ == "__main__":
    main()
