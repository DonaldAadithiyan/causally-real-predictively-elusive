"""G2 (E_select-profile oracle + Phase 3), G4 (direct L2 evidence), G5 (round-2 static check).
See runs/camera_ready/G/PREREG.md.     python -m camera_ready.g_analysis g2|g4|g5
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

from camera_ready.common import (Run, CR, HERE, ALPHA, block_idx, ci, boot_mean, spearman,
                                 available_seeds, weights_dir)
from camera_ready.phase34 import fit_eval_estimators, boot_best_r2
from camera_ready.e5_dose import regen_pool, load_models
from experiments._seed import derive_seed
from experiments.phase4_oracle_ceiling import _oracle_widths, _coverage_and_width
from calibration.aci import empirical_quantile
from p126.selection import fast_dose, LAM

torch.set_num_threads(4)
OUT = os.path.join(CR, "G")
DS = ["ETTh1", "ETTh2", "electricity", "traffic"]


def cells():
    return [(ds, s) for ds in DS for s in available_seeds(ds)]


# ------------------------------------------------------------------ G2
def g2():
    rows = []
    for ds, s in cells():
        r = Run(ds, s, load_X=False)
        ds_seed = derive_seed(s, ds)
        row = dict(dataset=ds, seed=s)
        for b, idx in (("id", r.idx_id), ("shift", r.idx_sh)):
            s_loc, s_sel = r.S_local(idx), r.S[idx]
            row[f"rho_Sdefs_{b}"] = spearman(s_loc, s_sel)
            for nm, sv in (("local", s_loc), ("select", s_sel)):
                q = empirical_quantile(sv, 1 - ALPHA)
                _, nw = _coverage_and_width(sv, np.full_like(sv, q))
                hw, _ = _oracle_widths(sv, ALPHA, 10)
                ocov, ow = _coverage_and_width(sv, hw)
                g, lo, hi, _ = boot_mean(2.0 * (q - hw))
                row.update({f"ratio_{b}_{nm}": g / nw, f"ratio_{b}_{nm}_lo": lo / nw,
                            f"ratio_{b}_{nm}_hi": hi / nw, f"naive_cov_{b}_{nm}": float(np.mean(sv <= q)),
                            f"oracle_cov_{b}_{nm}": ocov})
        # Phase 3 with each S definition (original 60/40 split, eval-mean R2, same estimators)
        idx = r.idx_id; cpt = int(0.6 * len(idx))
        feat = r.z_cols(r.mvdf)[idx]
        act = (feat != 0).any(1)
        for nm, sv in (("local", r.S_local(idx)), ("select", r.S[idx])):
            m, p = fit_eval_estimators(feat[:cpt], sv[:cpt], feat[cpt:], sv[cpt:], ds_seed)
            _, bci, _ = boot_best_r2(sv[cpt:], p, 96)
            row[f"P3_marg_best_{nm}"] = max(v["r2"] for v in m.values())
            row[f"P3_marg_best_{nm}_lo"], row[f"P3_marg_best_{nm}_hi"] = bci
            ta, ea = act[:cpt], act[cpt:]
            if ta.sum() >= 40 and ea.sum() >= 40:
                mc, pc = fit_eval_estimators(feat[:cpt][ta], sv[:cpt][ta], feat[cpt:][ea], sv[cpt:][ea], ds_seed)
                _, bc, _ = boot_best_r2(sv[cpt:][ea], pc, 96)
                row[f"P3_cond_best_{nm}"] = max(v["r2"] for v in mc.values())
                row[f"P3_cond_best_{nm}_lo"], row[f"P3_cond_best_{nm}_hi"] = bc
        rows.append(row)
        print(f"[G2 {r.tag}] ratio ID {row['ratio_id_local']:.3f}->{row['ratio_id_select']:.3f} "
              f"shift {row['ratio_shift_local']:.3f}->{row['ratio_shift_select']:.3f} "
              f"P3 {row['P3_marg_best_local']:+.3f}->{row['P3_marg_best_select']:+.3f} "
              f"rho {row['rho_Sdefs_id']:.3f}/{row['rho_Sdefs_shift']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "AGG_G2_select_profile.csv"), index=False)
    return df


# ------------------------------------------------------------------ G4
C = (LAM - LAM.mean()) / ((LAM - LAM.mean()) ** 2).sum()      # slope weights: U = sum_l C_l D(l)


def u_ci_per_feature(model, sae, X, y, z, tv, prof, pool, nb=2000):
    """Per pool feature: U_j (slope of mean S over lambda, active windows) and its moving-block
    bootstrap CI over the time positions of the given windows (inactive dropped)."""
    bidx = block_idx(len(X), 96, nb)
    out = {}
    for j in pool:
        j = int(j)
        a = z[:, j] != 0
        if a.sum() < 5:
            out[j] = (np.nan, np.nan, np.nan, int(a.sum())); continue
        _, _, _, sc_w, na = fast_dose(model, sae, X, y, z, tv, j, prof, per_window=True)
        slope_w = (C[:, None] * sc_w[0]).sum(0)                  # per-window slope, (n_a,)
        aidx = np.where(a)[0]
        posmap = -np.ones(len(X), int); posmap[aidx] = np.arange(len(aidx))
        sums = np.zeros(nb); cnts = np.zeros(nb)
        m = posmap[bidx]                                          # (nb, n)
        valid = m >= 0
        sums = np.where(valid, slope_w[np.where(valid, m, 0)], 0).sum(1)
        cnts = valid.sum(1)
        bs = np.where(cnts >= 3, sums / np.maximum(cnts, 1), np.nan)
        lo, hi = ci(bs)
        out[j] = (float(slope_w.mean()), lo, hi, int(na))
    return out


def g4():
    rows, feat_rows = [], []
    for ds, s in cells():
        r = Run(ds, s)
        model, sae = load_models(r)
        sel, z_sel, pool, wsel = regen_pool(r)
        mv = set(int(x) for x in r.mvdf)
        # (a) in-sample: E_select dose windows (selection windows)
        ins = u_ci_per_feature(model, sae, r.X[sel][wsel], r.y[sel][wsel], z_sel[wsel], r.tv, r.prof_sel, pool)
        # (b) out-of-sample: all E_eval_ID windows (never used for selection)
        idv = r.idx_id
        oos = u_ci_per_feature(model, sae, r.X[idv], r.y[idv], r.z_dense(idv), r.tv, r.prof_sel, pool)
        rng = np.random.default_rng(derive_seed(s, ds) + 4)
        row = dict(dataset=ds, seed=s)
        for nm, d in (("in", ins), ("oos", oos)):
            mov = {j: (np.isfinite(v[1]) and (v[1] > 0 or v[2] < 0)) for j, v in d.items()}
            pos = {j: (np.isfinite(v[1]) and v[1] > 0) for j, v in d.items()}
            fm = np.mean([mov[j] for j in pool if j in mv])
            fr = np.mean([mov[j] for j in pool if j not in mv])
            arr = np.array([mov[int(j)] for j in pool], float)
            rnd = np.array([arr[rng.choice(len(pool), 12, replace=False)].mean() for _ in range(1000)])
            row.update({f"{nm}_frac_mvdf": fm, f"{nm}_frac_rest": fr, f"{nm}_frac_random12_mean": rnd.mean(),
                        f"{nm}_mvdf_pct_vs_random12": float((rnd < fm).mean() + 0.5 * (rnd == fm).mean()),
                        f"{nm}_mvdf_gt_rest": bool(fm > fr),
                        f"{nm}_frac_mvdf_pos": np.mean([pos[j] for j in pool if j in mv]),
                        f"{nm}_frac_rest_pos": np.mean([pos[j] for j in pool if j not in mv]),
                        f"{nm}_n_mvdf_estimable": int(sum(np.isfinite(d[j][1]) for j in pool if j in mv))})
            for j, v in d.items():
                feat_rows.append(dict(dataset=ds, seed=s, version=nm, feature=j, in_mvdf=j in mv,
                                      U=v[0], U_lo=v[1], U_hi=v[2], n_active=v[3]))
        rows.append(row)
        print(f"[G4 {r.tag}] in: mvdf {row['in_frac_mvdf']:.2f} rest {row['in_frac_rest']:.2f} | "
              f"oos: mvdf {row['oos_frac_mvdf']:.2f} rest {row['oos_frac_rest']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "AGG_G4_L2_evidence.csv"), index=False)
    pd.DataFrame(feat_rows).to_csv(os.path.join(OUT, "G4_per_feature_U_ci.csv"), index=False)
    return df


# ------------------------------------------------------------------ G5
def g5():
    w = pd.read_csv(os.path.join(HERE, "runs/p126_round2/summary/wp1_continuous_red.csv"))
    x = w[(w.protocol == "P1_chrono_static") & (w.segment == "shift")].copy()
    x["n_positive"] = (x[["0", "1", "2"]] > 0).sum(1)
    x["consistently_narrower"] = x.n_positive == 3
    x.to_csv(os.path.join(OUT, "AGG_G5_static_shift.csv"), index=False)
    print(x[["dataset", "method", "0", "1", "2", "n_positive"]].to_string())
    return x


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"g2": g2, "g4": g4, "g5": g5}[sys.argv[1]]()
