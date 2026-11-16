"""Follow-up F1 (regime gating, 10 seeds), F2 (pooled enrichment tests), F5 (shift GRR +
scorer check). See runs/camera_ready/F/PREREG.md.   python -m camera_ready.f_followup"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from camera_ready.common import (Run, CR, HERE, block_idx, ci, two_level, enrichment_stats,
                                 available_seeds)
from experiments._seed import derive_seed
from p126.selection import type_B_enrichment
from validation.difficulty_specificity import feature_type_table

OUT = os.path.join(CR, "F")
DS_ALL = ["ETTh1", "ETTh2", "electricity", "traffic"]


def rq(s, d):
    if d.sum() < 10 or (~d).sum() < 10:
        return np.nan
    return np.quantile(s[d], .9) / np.quantile(s[~d], .9)


def f1():
    rows = []
    for ds in ("ETTh1", "ETTh2"):
        for seed in available_seeds(ds):
            r = Run(ds, seed, load_X=False)
            f = pd.read_csv(os.path.join(CR, "E5", "cells", r.tag, "features.csv"))
            cellB = f.feature[f.type == "B"].to_numpy(int)
            A = (r.z_cols(cellB) != 0).mean(1)
            tau = float(np.percentile(A[r.idx_sel], 90))
            for env, ix in (("E_eval_ID", r.idx_id), ("E_eval_shift", r.idx_sh)):
                s, d = r.S[ix], A[ix] >= tau
                row = dict(dataset=ds, seed=seed, env=env, n_cellB=len(cellB), tau_A=tau,
                           n1=int(d.sum()), n0=int((~d).sum()), R_Q=rq(s, d))
                for bl in (96, 192):
                    bi = block_idx(len(ix), bl)
                    v = np.array([rq(s[b], d[b]) for b in bi])
                    lo, hi = ci(v)
                    sfx = "" if bl == 96 else "_b192"
                    row.update({f"ci_lo{sfx}": lo, f"ci_hi{sfx}": hi,
                                f"excludes_1_above{sfx}": bool(lo > 1)})
                    if bl == 96:
                        row["_boot"] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    boots = {(a, b, c): v for a, b, c, v in zip(df.dataset, df.seed, df.env, df["_boot"])}
    df = df.drop(columns="_boot")
    df.to_csv(os.path.join(OUT, "AGG_F1_regime_gating.csv"), index=False)
    # reproduction vs round 2 (seeds 0-2, block 192 = 2H, same construction)
    r2 = pd.read_csv(os.path.join(HERE, "runs/p126_round2/wp1/regime_gating.csv"))
    m = df.merge(r2, on=["dataset", "seed", "env"], suffixes=("", "_r2"))
    m["dRQ"] = (m.R_Q - m.R_Q_r2).abs()
    m[["dataset", "seed", "env", "R_Q", "R_Q_r2", "dRQ", "n1", "n1_r2", "n_cellB", "n_cellB_r2"]].to_csv(
        os.path.join(OUT, "REPRO_F1_vs_round2.csv"), index=False)
    pooled = {}
    for ds in ("ETTh1", "ETTh2"):
        for env in ("E_eval_ID", "E_eval_shift"):
            pooled[(ds, env)] = two_level([v for k, v in boots.items() if k[0] == ds and k[2] == env])
    return df, m, pooled


def f2():
    rows, boots = [], []
    for ds in DS_ALL:
        for seed in available_seeds(ds):
            f = pd.read_csv(os.path.join(CR, "E5", "cells", f"{ds}_seed{seed}", "features.csv"))
            pool = f.feature.to_numpy(int)
            P, U = f.P_j.to_numpy(float), f.U_j.to_numpy(float)
            import json
            cv = json.load(open(os.path.join(CR, "E5", "cells", f"{ds}_seed{seed}", "curves.json")))
            from validation.difficulty_specificity import _curve_monotone
            mono = np.array([_curve_monotone({l: cv["curves"][str(int(j))]["D_unc"][i]
                                              for i, l in enumerate(cv["lambdas"])}) for j in pool])
            ftab = feature_type_table(P, U, feature_indices=pool)
            mv = f.feature[f.in_mvdf].to_numpy(int)
            e = enrichment_stats(mv, ftab, pool)
            obs, sel = type_B_enrichment(P, U, None, mono, pool, 12)
            assert abs(obs - e["enrichment"]) < 1e-9 and sorted(sel) == sorted(mv), "helper mismatch"
            rng = np.random.default_rng(derive_seed(seed, ds) + 2026)
            perm = []
            for _ in range(1000):
                p = rng.permutation(len(pool))
                perm.append(type_B_enrichment(P, U[p], None, mono[p], pool, 12)[0])
            perm = np.array(perm, float)
            p_perm = float((1 + np.sum(perm >= obs - 1e-12)) / (1 + np.isfinite(perm).sum()))
            B = set(ftab["cells"]["B"])
            sb = np.array([j in B for j in sorted(mv)], float); pb = np.array([j in B for j in sorted(pool)], float)
            r2 = np.random.default_rng(0)
            num = sb[r2.integers(0, len(sb), (2000, len(sb)))].mean(1)
            den = pb[r2.integers(0, len(pb), (2000, len(pb)))].mean(1)
            boots.append(num / np.where(den > 0, den, np.nan))
            rows.append(dict(dataset=ds, seed=seed, **e, permU_mean=float(np.nanmean(perm)),
                             permU_q95=float(np.nanquantile(perm, .95)), permU_p=p_perm,
                             exceeds_permU_q95=bool(obs > np.nanquantile(perm, .95))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "AGG_F2_enrichment_tests.csv"), index=False)

    def comb(p):
        p = np.clip(np.asarray(p, float), 1e-300, 1)
        fisher = stats.combine_pvalues(p, method="fisher").pvalue
        stouffer = stats.combine_pvalues(p, method="stouffer").pvalue
        return float(fisher), float(stouffer)
    summ = []
    for name, sub, bb in [("all 30", df, boots)] + [
            (ds, df[df.dataset == ds], [b for b, (_, r) in zip(boots, df.iterrows()) if r.dataset == ds])
            for ds in DS_ALL]:
        lo, hi = two_level(bb)
        fh, sh = comb(sub.enr_p_hypergeom)
        fp, sp = comb(sub.permU_p)
        summ.append(dict(group=name, n=len(sub), mean_enrichment=sub.enrichment.mean(),
                         two_level_CI_lo=lo, two_level_CI_hi=hi,
                         fisher_hypergeom=fh, stouffer_hypergeom=sh,
                         n_permU_p_lt_05=int((sub.permU_p < .05).sum()),
                         n_exceeds_permU_q95=int(sub.exceeds_permU_q95.sum()),
                         mean_permU_null=sub.permU_mean.mean(),
                         fisher_permU=fp, stouffer_permU=sp))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, "AGG_F2_pooled.csv"), index=False)
    return df, S


def f5():
    g = pd.concat([pd.read_csv(os.path.join(CR, "E3", "AGG_E3_gap_recovery.csv")),
                   pd.read_csv(os.path.join(CR, "E1", "AGG_E1_E3_gap_recovery.csv"))], ignore_index=True)
    sh = g[g.split == "shift"].copy()
    sh["denominator_positive"] = sh.W_naive > sh.W_oracle
    sh.to_csv(os.path.join(OUT, "AGG_F5_shift_gap.csv"), index=False)
    sc = pd.read_csv(os.path.join(CR, "E2", "AGG_E2_scorer_vs_naive.csv"))
    sc = sc[sc["mode"] != "naive_matched"].copy()
    sc["narrower_point"] = sc.width_minus_naive < 0
    sc["narrower_sig"] = sc.wmn_hi < 0
    sc["cov_valid"] = (sc.coverage - 0.9).abs() <= 0.02
    sc["naive_cov_valid"] = (sc.naive_coverage - 0.9).abs() <= 0.02
    sc.to_csv(os.path.join(OUT, "AGG_F5_scorer_check.csv"), index=False)
    return sh, sc


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d1, rep, pooled = f1()
    print(d1[["dataset", "seed", "env", "n1", "R_Q", "ci_lo", "ci_hi", "excludes_1_above",
              "excludes_1_above_b192"]].round(3).to_string())
    print("repro max |dRQ| (s0-2):", rep.dRQ.max(), "cellB equal:", bool((rep.n_cellB == rep.n_cellB_r2).all()))
    print({k: tuple(round(x, 3) for x in v) for k, v in pooled.items()})
    pd.Series({f"{k[0]}|{k[1]}": v for k, v in pooled.items()}).to_json(os.path.join(OUT, "F1_pooled.json"))
    d2, S = f2()
    print(S.round(4).to_string())
    sh, sc = f5()
    print(sh.groupby(["dataset", "feature_set"]).apply(lambda x: pd.Series(dict(
        valid=int(x.coverage_valid.sum()), n=len(x), cov_med=x.coverage.median(),
        GRR_valid=", ".join(f"s{int(s)}:{v:+.2f}" for s, v, c in zip(x.seed, x.GRR, x.coverage_valid) if c)))).to_string())
    print(sc.groupby(["dataset", "mode", "block"]).agg(n=("seed", "size"), narrower_point=("narrower_point", "sum"),
                                                       narrower_sig=("narrower_sig", "sum"),
                                                       narrower_point_and_valid=("narrower_point", lambda v: int((v & sc.loc[v.index, "cov_valid"]).sum())),
                                                       cov_valid=("cov_valid", "sum")).to_string())
