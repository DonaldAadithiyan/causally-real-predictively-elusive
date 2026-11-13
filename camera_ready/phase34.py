"""Phase 3 (L1 battery) + Phase 4 (oracle) from saved per-window arrays, ORIGINAL definitions.

Phase 3 = experiments/phase3_real_predictability.py:167-191 with
experiments/phase1_synthetic_control._fit_eval_estimators copied verbatim except that it
also returns predictions (needed for the block bootstrap). Target = E_eval_ID-local S
(collect() semantics), 60/40 split, eval-mean R2.
Phase 4 = experiments/phase4_oracle_ceiling.run_one from the scoring step on.

Reproduction gate: on ETTh seeds 0-2 every stored number must match
runs/phase3_real/AGG_phase3.csv and runs/phase4_oracle/AGG_oracle.csv.

Usage: python -m camera_ready.phase34 --datasets ETTh1 ETTh2 --seeds 0 1 2 --out runs/camera_ready/E2
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from camera_ready.common import (Run, HERE, ALPHA, block_idx, ci, boot_mean, summ, md_table,
                                 provenance)
from experiments._seed import derive_seed
from experiments.phase4_oracle_ceiling import _oracle_widths, _coverage_and_width
from calibration.aci import empirical_quantile

torch.set_num_threads(4)


def fit_eval_estimators(feat_tr, s_tr, feat_te, s_te, seed):
    """Verbatim copy of phase1_synthetic_control._fit_eval_estimators + returns preds."""
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from scipy import stats
    out, preds = {}, {}

    def _metrics(pred):
        ss_res = float(np.sum((s_te - pred) ** 2))
        ss_tot = float(np.sum((s_te - s_te.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        rc = float(stats.spearmanr(pred, s_te).statistic) if np.std(pred) > 1e-12 else 0.0
        return dict(r2=r2, rank_corr=rc)

    lr = LinearRegression().fit(feat_tr, s_tr)
    preds["linear"] = lr.predict(feat_te)
    if feat_tr.shape[1] == 1:
        xcol_tr, xcol_te = feat_tr[:, 0], feat_te[:, 0]
    else:
        corrs = [abs(np.corrcoef(feat_tr[:, j], s_tr)[0, 1]) if np.std(feat_tr[:, j]) > 0 else 0
                 for j in range(feat_tr.shape[1])]
        jbest = int(np.argmax(corrs))
        xcol_tr, xcol_te = feat_tr[:, jbest], feat_te[:, jbest]
    iso = IsotonicRegression(out_of_bounds="clip").fit(xcol_tr, s_tr)
    preds["isotonic"] = iso.predict(xcol_te)
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=seed).fit(feat_tr, s_tr)
    preds["gbm"] = gbm.predict(feat_te)
    from calibration.ffdci_head import train_quantile_mlp
    mlp = train_quantile_mlp(feat_tr.astype(np.float32), s_tr.astype(np.float32),
                             alpha_target=0.5, epochs=300, lr=1e-3, seed=seed)
    with torch.no_grad():
        preds["mlp"] = mlp(torch.from_numpy(feat_te.astype(np.float32))).numpy()
    for k, p in preds.items():
        out[k] = _metrics(p)
    return out, preds


def boot_best_r2(s_te, preds, block):
    """Block-bootstrap CI of eval-mean R2 for each estimator and of the max over estimators."""
    idx = block_idx(len(s_te), block)
    Y = s_te[idx]
    ss = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1) + 1e-12
    per = {k: 1 - ((Y - np.asarray(p)[idx]) ** 2).sum(1) / ss for k, p in preds.items()}
    best = np.max(np.stack(list(per.values())), 0)
    return {k: ci(v) for k, v in per.items()}, ci(best), best


def phase3(run, block=96):
    ds_seed = derive_seed(run.seed, run.ds)
    idx = run.idx_id
    s_id = run.S_local(idx)
    npd = len(idx); cpt = int(0.6 * npd)
    feat = run.z_cols(run.mvdf)[idx]
    m, p = fit_eval_estimators(feat[:cpt], s_id[:cpt], feat[cpt:], s_id[cpt:], ds_seed)
    per_ci, best_ci, best_b = boot_best_r2(s_id[cpt:], p, block)
    active = (feat != 0).any(1)
    tr_a, te_a = active[:cpt], active[cpt:]
    row = dict(dataset=run.ds, seed=run.seed, n_fit=cpt, n_eval=npd - cpt,
               mvdf_activation_rate=float(active.mean()),
               n_active_windows_eval=int(te_a.sum()))
    for k in m:
        row[f"marg_{k}_r2"] = m[k]["r2"]
        row[f"marg_{k}_r2_lo"], row[f"marg_{k}_r2_hi"] = per_ci[k]
    row["marg_best_r2"] = max(v["r2"] for v in m.values())
    row["marg_best_r2_lo"], row["marg_best_r2_hi"] = best_ci
    row["marg_best_rho"] = max(m.items(), key=lambda kv: kv[1]["r2"])[1]["rank_corr"]
    row["cond_best_r2"] = row["cond_best_r2_lo"] = row["cond_best_r2_hi"] = np.nan
    if tr_a.sum() >= 40 and te_a.sum() >= 40:
        mc, pc = fit_eval_estimators(feat[:cpt][tr_a], s_id[:cpt][tr_a], feat[cpt:][te_a],
                                     s_id[cpt:][te_a], ds_seed)
        _, bci, _ = boot_best_r2(s_id[cpt:][te_a], pc, block)
        for k in mc:
            row[f"cond_{k}_r2"] = mc[k]["r2"]
        row["cond_best_r2"] = max(v["r2"] for v in mc.values())
        row["cond_best_r2_lo"], row["cond_best_r2_hi"] = bci
    return row, best_b


def phase4(run, block=96):
    ds_seed = derive_seed(run.seed, run.ds)
    out = dict(dataset=run.ds, seed=run.seed)
    boots = {}
    rng = np.random.default_rng(ds_seed)          # same consumption order as the script
    for tag, idx in (("id", run.idx_id), ("shift", run.idx_sh)):
        s = run.S_local(idx)
        q = empirical_quantile(s, 1 - ALPHA)
        n_cov, n_w = _coverage_and_width(s, np.full_like(s, q))
        hw, _ = _oracle_widths(s, ALPHA, 10)
        o_cov, o_w = _coverage_and_width(s, hw)
        d = 2.0 * (np.full_like(s, q) - hw)
        bi = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        g, glo, ghi, gb = boot_mean(d, block)
        out.update({f"n_{tag}": len(s), f"naive_{tag}_coverage": n_cov, f"naive_{tag}_width": n_w,
                    f"oracle_{tag}_coverage": o_cov, f"oracle_{tag}_width": o_w,
                    f"gap_{tag}": g, f"gap_{tag}_ci_lo_iid": float(np.percentile(bi, 2.5)),
                    f"gap_{tag}_ci_hi_iid": float(np.percentile(bi, 97.5)),
                    f"gap_{tag}_ci_lo": glo, f"gap_{tag}_ci_hi": ghi,
                    f"ratio_{tag}": g / n_w, f"ratio_{tag}_ci_lo": glo / n_w,
                    f"ratio_{tag}_ci_hi": ghi / n_w})
        boots[tag] = gb / n_w
    return out, boots


def check_repro(p3, p4):
    a3 = pd.read_csv(os.path.join(HERE, "runs/phase3_real/AGG_phase3.csv"))
    a4 = pd.read_csv(os.path.join(HERE, "runs/phase4_oracle/AGG_oracle.csv"))
    rows = []
    for _, r in p3.iterrows():
        m = a3[(a3.dataset == r.dataset) & (a3.seed == r.seed)]
        if not len(m):
            continue
        m = m.iloc[0]
        for k_new, k_old in (("marg_linear_r2", "pred_marginal_linear_r2"),
                             ("marg_isotonic_r2", "pred_marginal_isotonic_r2"),
                             ("marg_gbm_r2", "pred_marginal_gbm_r2"),
                             ("marg_mlp_r2", "pred_marginal_mlp_r2"),
                             ("marg_best_r2", "pred_marginal_best_r2"),
                             ("cond_best_r2", "pred_cond_best_r2")):
            rows.append(dict(dataset=r.dataset, seed=r.seed, quantity=k_old, stored=m[k_old],
                             recomputed=r[k_new], abs_diff=abs(m[k_old] - r[k_new])))
    for _, r in p4.iterrows():
        m = a4[(a4.dataset == r.dataset) & (a4.seed == r.seed)]
        if not len(m):
            continue
        m = m.iloc[0]
        for k in ("naive_id_width", "oracle_id_width", "ratio_id", "naive_shift_width",
                  "oracle_shift_width", "ratio_shift", "gap_id_ci_lo", "gap_shift_ci_lo"):
            kn = k + "_iid" if k.endswith("ci_lo") else k
            rows.append(dict(dataset=r.dataset, seed=r.seed, quantity=k, stored=m[k],
                             recomputed=r[kn], abs_diff=abs(m[k] - r[kn])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    p3, p4, b3, b4 = [], [], {}, {}
    for ds in a.datasets:
        for s in a.seeds:
            run = Run(ds, s, load_X=False)
            r3, bb = phase3(run); p3.append(r3); b3[(ds, s)] = bb
            r4, bo = phase4(run); p4.append(r4); b4[(ds, s)] = bo
            r4b, _ = phase4(run, block=192)
            p4[-1].update({k + "_b192": v for k, v in r4b.items() if k.startswith("ratio_") and "ci" in k})
            r3b, _ = phase3(run, block=192)
            p3[-1].update({k + "_b192": r3b[k] for k in ("marg_best_r2_lo", "marg_best_r2_hi",
                                                         "cond_best_r2_lo", "cond_best_r2_hi")})
            print(f"[{run.tag}] P3 best {r3['marg_best_r2']:+.3f} [{r3['marg_best_r2_lo']:+.3f},"
                  f"{r3['marg_best_r2_hi']:+.3f}] cond {r3['cond_best_r2']:+.3f} | P4 ratio ID "
                  f"{r4['ratio_id']:.3f} shift {r4['ratio_shift']:.3f}", flush=True)
    p3, p4 = pd.DataFrame(p3), pd.DataFrame(p4)
    sfx = f"_{a.tag}" if a.tag else ""
    p3.to_csv(os.path.join(a.out, f"AGG_phase3{sfx}.csv"), index=False)
    p4.to_csv(os.path.join(a.out, f"AGG_phase4{sfx}.csv"), index=False)
    rep = check_repro(p3, p4)
    if len(rep):
        rep.to_csv(os.path.join(a.out, f"REPRO_phase34{sfx}.csv"), index=False)
        print(f"reproduction max |diff| = {rep.abs_diff.max():.2e} over {len(rep)} stored numbers")
    pd.to_pickle(dict(b3=b3, b4=b4), os.path.join(a.out, f"_boot_phase34{sfx}.pkl"))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
