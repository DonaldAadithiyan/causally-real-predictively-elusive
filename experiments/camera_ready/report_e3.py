"""Aggregate E3 cells and evaluate C3.1-C3.5 exactly as pre-registered.

  python -m experiments.camera_ready.report_e3 --cells results/camera_ready/E3/cells --out results/camera_ready/E3 \
      --datasets ETTh1 ETTh2
Writes AGG_E3_predictability.csv, AGG_E3_gap_recovery.csv, AGG_E3_delta.csv,
CRITERIA_E3.json, and returns markdown blocks used by the RESULT writers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from experiments.camera_ready.common import pass_threshold, two_level, summ, md_table

EXANTE = ["F1", "F1p", "F2", "F3", "F6"]
LABEL = {"F0": "F0 MVDF", "F1": "F1 lagged S (full)", "F1p": "F1p lagged S (partial)",
         "F2": "F2 input stats", "F3": "F3 forecast stats", "F4": "F4 hidden PCA-12",
         "F5": "F5 SAE PCA-12", "F6": "F6 all ex-ante", "F7": "F7 all + MVDF"}


def load(cells, datasets):
    P, G, D, B = [], [], [], {}
    for d in sorted(glob.glob(os.path.join(cells, "*"))):
        tag = os.path.basename(d)
        ds, s = tag.rsplit("_seed", 1)
        if ds not in datasets or not os.path.exists(os.path.join(d, "boot.pkl")):
            continue
        P.append(pd.read_csv(os.path.join(d, "pred.csv")))
        G.append(pd.read_csv(os.path.join(d, "gap.csv")))
        if os.path.getsize(os.path.join(d, "delta.csv")) > 5:
            D.append(pd.read_csv(os.path.join(d, "delta.csv")))
        B[(ds, int(s))] = pickle.load(open(os.path.join(d, "boot.pkl"), "rb"))
    P = pd.concat(P, ignore_index=True)
    G = pd.concat(G, ignore_index=True)
    G["grr_denominator_positive"] = G.W_naive > G.W_oracle
    D = pd.concat(D, ignore_index=True) if D else pd.DataFrame()
    return P, G, D, B


def predictive(r, sfx=""):
    return bool(r[f"R2_lo{sfx}"] > 0 and r[f"rho_lo{sfx}"] > 0)


def criteria(P, G, D, ds, sfx="", gap_split="ID"):
    sel = P[(P.dataset == ds) & (P.split == "ID") & P.cv_selected]
    seeds = sorted(sel.seed.unique())
    N = len(seeds); thr = pass_threshold(N)
    pred = {fs: {s: predictive(sel[(sel.feature_set == fs) & (sel.seed == s)].iloc[0], sfx)
                 for s in seeds if len(sel[(sel.feature_set == fs) & (sel.seed == s)])}
            for fs in sel.feature_set.unique()}
    cnt = {fs: int(sum(v.values())) for fs, v in pred.items()}
    ex_sets = [fs for fs in EXANTE if fs in cnt]
    c31_sets = [fs for fs in ex_sets if cnt[fs] >= thr]
    lenient = int(sum(any(pred[fs].get(s, False) for fs in ex_sets) for s in seeds))
    c31 = bool(c31_sets)
    f0_not = int(sum(not v for v in pred.get("F0", {}).values()))
    c32 = bool(c31 and f0_not >= thr)
    out = dict(dataset=ds, N=N, threshold=thr, predictive_counts=cnt,
               C3_1=dict(pass_=c31, sets_passing=c31_sets, lenient_any_set_seeds=lenient),
               C3_2=dict(pass_=c32, F0_not_predictive_seeds=f0_not))
    if len(D):
        dd = D[(D.dataset == ds) & (D.split == "ID") & (D.block == (192 if sfx else 96))]
        ok = int((dd.dR2_lo <= 0).sum())
        out["C3_3"] = dict(pass_=bool(ok >= thr), seeds_ci_contains_or_below_0=ok, n=len(dd),
                           dR2=summ(dd.dR2), dR2_per_seed=dict(zip(dd.seed.astype(int), dd.dR2.round(4))))
    g = G[(G.dataset == ds) & (G.split == gap_split) & G.is_best_exante]
    ok = g[(g.GRR >= 0.20) & g.coverage_valid & g.grr_denominator_positive.fillna(False).astype(bool)]
    out["C3_4"] = dict(pass_=bool(len(ok) >= thr), block=gap_split, seeds_ok=len(ok),
                       best_set_per_seed=dict(zip(g.seed.astype(int), g.feature_set)),
                       GRR_per_seed=dict(zip(g.seed.astype(int), g.GRR.round(3))),
                       coverage_per_seed=dict(zip(g.seed.astype(int), g.coverage.round(3))),
                       denominator_positive=dict(zip(g.seed.astype(int),
                                                     g.grr_denominator_positive.astype(bool))))
    rep = {fs: cnt.get(fs, 0) for fs in ("F4", "F5")}
    out["C3_5"] = dict(pass_=bool(max(rep.values(), default=0) >= thr), counts=rep)
    return out


def pooled(B, ds, split, fs, key="r2"):
    arr = [v[(split, fs)][key] for (d, s), v in B.items() if d == ds and (split, fs) in v]
    return two_level(arr) if arr else (np.nan, np.nan)


def tables(P, G, D, B, datasets):
    md = []
    sel = P[(P.split == "ID") & P.cv_selected]
    # 1. CV-selected per seed
    md.append("### CV-selected estimator, E_eval_ID: R² (vs fit mean) [block CI] / Spearman ρ [CI]; ✓ = predictive (C3.1 rule)\n")
    for ds in datasets:
        s = sel[sel.dataset == ds]
        if not len(s):
            continue
        rows = []
        for fs in [f for f in LABEL if f in s.feature_set.unique()]:
            r = {"set": LABEL[fs]}
            for _, x in s[s.feature_set == fs].sort_values("seed").iterrows():
                r[f"s{int(x.seed)}"] = (f"{x.R2:+.3f} [{x.R2_lo:+.2f},{x.R2_hi:+.2f}] / "
                                        f"{x.rho:+.2f} [{x.rho_lo:+.2f},{x.rho_hi:+.2f}] "
                                        f"{'✓' if predictive(x) else '·'} ({x.estimator})")
            v = s[s.feature_set == fs]
            sm = summ(v.R2)
            lo, hi = pooled(B, ds, "ID", fs)
            r["median R² [min,max]"] = f"{sm['median']:+.3f} [{sm['min']:+.3f},{sm['max']:+.3f}]"
            r["pooled 2-level CI"] = f"[{lo:+.3f},{hi:+.3f}]"
            r["k/N predictive"] = f"{sum(predictive(x) for _, x in v.iterrows())}/{len(v)}"
            rows.append(r)
        n = len(s[s.feature_set == "F0"])
        md.append(f"\n**{ds}** (n_fit = {int(s.n_fit.iloc[0])}, n_eval = {int(s.n_eval.iloc[0])}, N = {n} seeds)\n\n")
        md.append(md_table(pd.DataFrame(rows)))
    # 2. best-on-eval (descriptive) and full grid medians
    md.append("\n### Descriptive only: best-on-eval R² per set (max over the 7 estimators; selection on eval is optimistic), median over seeds [min, max]\n\n")
    rows = []
    for ds in datasets:
        for split in ("ID", "shift"):
            q = P[(P.dataset == ds) & (P.split == split)]
            if not len(q):
                continue
            b = q.groupby(["feature_set", "seed"]).R2.max().reset_index()
            r = {"dataset": ds, "split": split}
            for fs in [f for f in LABEL if f in b.feature_set.unique()]:
                sm = summ(b[b.feature_set == fs].R2)
                r[fs] = f"{sm['median']:+.2f} [{sm['min']:+.2f},{sm['max']:+.2f}]"
            rows.append(r)
    md.append(md_table(pd.DataFrame(rows)))
    md.append("\n### Full estimator × feature-set grid, E_eval_ID, median R² over seeds\n\n")
    for ds in datasets:
        q = P[(P.dataset == ds) & (P.split == "ID")]
        if not len(q):
            continue
        g = q.groupby(["feature_set", "estimator"]).R2.median().unstack().round(3).reset_index()
        md.append(f"\n**{ds}**\n\n" + md_table(g))
    md.append("\n### Same grid, median Spearman ρ\n\n")
    for ds in datasets:
        q = P[(P.dataset == ds) & (P.split == "ID")]
        if not len(q):
            continue
        g = q.groupby(["feature_set", "estimator"]).rho.median().unstack().round(3).reset_index()
        md.append(f"\n**{ds}**\n\n" + md_table(g))
    # 3. shift
    md.append("\n### CV-selected estimator, E_eval_shift (models fit on the ID fit split)\n\n")
    s = P[(P.split == "shift") & P.cv_selected]
    rows = []
    for ds in datasets:
        for fs in [f for f in LABEL if f in s[s.dataset == ds].feature_set.unique()]:
            v = s[(s.dataset == ds) & (s.feature_set == fs)]
            sm, rs = summ(v.R2), summ(v.rho)
            rows.append({"dataset": ds, "set": LABEL[fs],
                         "R² median [min,max]": f"{sm['median']:+.3f} [{sm['min']:+.3f},{sm['max']:+.3f}]",
                         "ρ median [min,max]": f"{rs['median']:+.2f} [{rs['min']:+.2f},{rs['max']:+.2f}]",
                         "k/N predictive": f"{sum(predictive(x) for _, x in v.iterrows())}/{len(v)}"})
    md.append(md_table(pd.DataFrame(rows)))
    # 4. delta
    if len(D):
        md.append("\n### C3.3 ΔR² = R²(F7) − R²(F6), F6's CV-selected estimator, E_eval_ID, block 96\n\n")
        dd = D[(D.split == "ID") & (D.block == 96)].copy()
        dd["CI"] = [f"[{a:+.3f},{b:+.3f}]" for a, b in zip(dd.dR2_lo, dd.dR2_hi)]
        md.append(md_table(dd[["dataset", "seed", "estimator", "R2_F6", "R2_F7", "dR2", "CI"]]))
    # 5. gap
    md.append("\n### Gap recovery (static split conformal; 50/30/20 of E_eval_ID; same test windows for all)\n\n")
    gg = G.copy()
    gg["GRR_str"] = [("—" if fs == "naive_matched" else
                      "undefined (W_naive ≤ W_oracle)" if not bool(dp) else
                      f"{grr:+.2f}" if cv else f"coverage-invalid ({grr:+.2f})")
                     for fs, grr, cv, dp in zip(gg.feature_set, gg.GRR, gg.coverage_valid,
                                                gg.get("grr_denominator_positive", True).fillna(True))]
    for ds in datasets:
        for split in ("ID", "shift"):
            q = gg[(gg.dataset == ds) & (gg.split == split)]
            if not len(q):
                continue
            q = q.copy()
            q["best_exante"] = q.is_best_exante.map({True: "★", False: ""})
            md.append(f"\n**{ds} — {split}** (n_test = {int(q.n_test.iloc[0])})\n\n")
            md.append(md_table(q[["seed", "feature_set", "estimator", "coverage", "width", "W_naive",
                                  "W_oracle", "GRR_str", "best_exante"]].sort_values(["feature_set", "seed"])))
    return "".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/camera_ready/E3/cells")
    ap.add_argument("--out", default="results/camera_ready/E3")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--prefix", default="E3")
    a = ap.parse_args()
    P, G, D, B = load(a.cells, a.datasets)
    cols = ["dataset", "seed", "split", "feature_set", "estimator", "n_fit", "n_eval", "R2", "R2_lo",
            "R2_hi", "rho", "rho_lo", "rho_hi"]
    P[cols + [c for c in P.columns if c not in cols]].to_csv(
        os.path.join(a.out, f"AGG_{a.prefix}_predictability.csv"), index=False)
    gcols = ["dataset", "seed", "split", "feature_set", "coverage", "width", "W_naive", "W_oracle",
             "GRR", "coverage_valid"]
    G[gcols + [c for c in G.columns if c not in gcols]].to_csv(
        os.path.join(a.out, f"AGG_{a.prefix}_gap_recovery.csv"), index=False)
    if len(D):
        D.to_csv(os.path.join(a.out, f"AGG_{a.prefix}_delta.csv"), index=False)
    crit = {}
    for ds in a.datasets:
        if ds not in set(P.dataset):
            continue
        gs = "ID"
        if ds == "traffic":
            nv = G[(G.dataset == ds) & (G.split == "ID") & (G.feature_set == "naive_matched")].coverage
            gs = "shift" if (nv > 0.97).any() else "ID"
        crit[ds] = dict(block96=criteria(P, G, D, ds, gap_split=gs),
                        block192=criteria(P, G, D, ds, sfx="_b192", gap_split=gs),
                        shift_block=criteria(P, G, D, ds, gap_split="shift"))
    json.dump(crit, open(os.path.join(a.out, f"CRITERIA_{a.prefix}.json"), "w"), indent=2, default=str)
    open(os.path.join(a.out, f"_tables_{a.prefix}.md"), "w").write(tables(P, G, D, B, a.datasets))
    for ds, c in crit.items():
        b = c["block96"]
        print(ds, {k: (v["pass_"] if isinstance(v, dict) and "pass_" in v else v) for k, v in b.items()
                   if k.startswith("C3")}, b["predictive_counts"])


if __name__ == "__main__":
    main()
