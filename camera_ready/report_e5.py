"""E5 aggregation: E5a diagnostics (C5a), E5b natural-range re-selection (C5b), E5c (C5c).
  python -m camera_ready.report_e5
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from camera_ready.common import (CR, enrichment_stats, jaccard, pass_threshold, md_table, summ,
                                 provenance)
from validation.difficulty_specificity import feature_type_table, select_mvdf_v2
from p126.selection import slope

OUT = os.path.join(CR, "E5")
PRIMARY = ("ETTh1", "ETTh2")


def cells():
    out = []
    for d in sorted(glob.glob(os.path.join(OUT, "cells", "*"))):
        if os.path.exists(os.path.join(d, "features.csv")):
            ds, s = os.path.basename(d).rsplit("_seed", 1)
            out.append((ds, int(s), d))
    return out


def e5b_cell(ds, s, d):
    f = pd.read_csv(os.path.join(d, "features.csv"))
    cv = json.load(open(os.path.join(d, "curves.json")))
    lam = np.array(cv["lambdas"]); keep = lam <= 1.0
    pool = f.feature.to_numpy(int)
    P_full, U_full = f.P_j.to_numpy(float), f.U_j.to_numpy(float)
    ftab_full = feature_type_table(P_full, U_full, feature_indices=pool)
    mv_full = f.feature[f.in_mvdf].to_numpy(int)
    P5, U5, curves5 = [], [], {}
    for j in pool:
        c = cv["curves"][str(int(j))]
        dp, du = np.array(c["D_pred"])[keep], np.array(c["D_unc"])[keep]
        P5.append(slope(lam[keep], dp)[0]); U5.append(slope(lam[keep], du)[0])
        curves5[int(j)] = {float(l): float(v) for l, v in zip(lam[keep], du)}
    P5, U5 = np.array(P5), np.array(U5)
    dose = dict(P_j=P5, U_j=U5, feature_indices=pool, unc_curves=curves5)
    mv5 = np.asarray(select_mvdf_v2(dose, u_threshold=0.0, k=12, residualize_method="isotonic")["selected"], int)
    ftab5 = feature_type_table(P5, U5, feature_indices=pool)
    ef = enrichment_stats(mv_full, ftab_full, pool)
    e5 = enrichment_stats(mv5, ftab5, pool)
    return dict(dataset=ds, seed=s, jaccard_vs_full=jaccard(mv5, mv_full),
                rho_U5_vs_U=float(pd.Series(U5).corr(pd.Series(U_full), method="spearman")),
                **{f"full_{k}": v for k, v in ef.items()}, **{f"nat_{k}": v for k, v in e5.items()},
                mvdf_nat=" ".join(map(str, sorted(mv5))), mvdf_full=" ".join(map(str, sorted(mv_full))))


def main():
    t0 = time.time()
    C = cells()
    feats = pd.concat([pd.read_csv(os.path.join(d, "features.csv")) for _, _, d in C], ignore_index=True)
    feats.to_csv(os.path.join(OUT, "AGG_E5a_diagnostics.csv"), index=False)
    repro = pd.DataFrame([json.load(open(os.path.join(d, "repro.json"))) for _, _, d in C])
    b = pd.DataFrame([e5b_cell(ds, s, d) for ds, s, d in C])
    b.to_csv(os.path.join(OUT, "AGG_E5b_natural_range.csv"), index=False)
    ic = [pd.read_csv(os.path.join(d, "interchange.csv")) for _, _, d in C
          if os.path.exists(os.path.join(d, "interchange.csv"))]
    ic = pd.concat(ic, ignore_index=True) if ic else pd.DataFrame()
    if len(ic):
        ic.to_csv(os.path.join(OUT, "AGG_E5c_interchange.csv"), index=False)

    L = ["# E5 — Dose-response diagnostics and natural-range interventions\n\n",
         provenance("python -m camera_ready.e5_dose a|c --datasets ETTh1 ETTh2 electricity traffic "
                    "--seeds <all>; python -m camera_ready.report_e5",
                    {ds: sorted(feats[feats.dataset == ds].seed.unique().tolist()) for ds in feats.dataset.unique()},
                    sorted(feats.dataset.unique()), t0), "\n"]
    L.append(open(os.path.join(OUT, "PREREG.md")).read().split("## Common setup")[0].replace("# E5", "## Pre-registration (summary; full text in PREREG.md)") + "\n")
    # ---- reproduction first
    L.append("## Reproduction check (reported first, as required)\n\n")
    L.append(md_table(repro[["tag", "pool_matches_stored", "mvdf_matches", "max_abs_dP", "max_abs_dU",
                             "max_rel_dU", "n_pool", "n_wsel"]], fmt="{:.2e}"))
    L.append(f"\nAll {len(repro)} cells: pool identical = {bool(repro.pool_matches_stored.all())}, "
             f"MVDF identical = {bool(repro.mvdf_matches.all())}. Stored P_j/U_j are reproduced to "
             f"≤ {repro.max_abs_dU.max():.1e} absolute (the relative maximum comes from |U_j| ≈ 0 features). "
             "Seeds ≥ 3 are compared against the candidate tables written by their own E2 run.\n\n")
    # ---- E5a
    L.append("## E5a — diagnostics\n\n")
    pm = feats[feats.dataset.isin(PRIMARY)].copy()
    pm["absUs"] = pm.U_spearman.abs(); pm["absPs"] = pm.P_spearman.abs()
    q = lambda v: f"{np.nanpercentile(v,10):.3f} / {np.nanpercentile(v,25):.3f} / {np.nanmedian(v):.3f} / {np.nanpercentile(v,75):.3f}"
    rows = []
    for grp, sub in (("MVDF", pm[pm.in_mvdf]), ("rest of pool", pm[~pm.in_mvdf])):
        rows.append({"group (ETTh, pooled)": grp, "n": len(sub),
                     "U fit R² q10/q25/med/q75": q(sub.U_r2), "|ρ(λ,U)| q10/q25/med/q75": q(sub.absUs),
                     "P fit R² median": f"{np.nanmedian(sub.P_r2):.3f}",
                     "|ρ(λ,P)| median": f"{np.nanmedian(sub.absPs):.3f}",
                     "U quad p<0.05": f"{np.nanmean(sub.U_quad_p < 0.05):.2f}",
                     "exceed q99 @1.25 (active) median": f"{np.nanmedian(sub.frac_exceed_q99_active):.3f}",
                     "exceed q99 (incl. zeros) median": f"{np.nanmedian(sub.frac_exceed_q99_all):.3f}"})
    L.append(md_table(pd.DataFrame(rows)))
    mv = pm[pm.in_mvdf]
    c5a = bool(np.nanmedian(mv.U_r2) >= 0.8 and np.nanmedian(mv.absUs) >= 0.8)
    L.append(f"\n**C5a: {'PASS' if c5a else 'FAIL'}** — MVDF median linear-fit R² of U_j = "
             f"{np.nanmedian(mv.U_r2):.3f} (≥ 0.8 required), median |Spearman(λ, D_unc)| = "
             f"{np.nanmedian(mv.absUs):.3f} (≥ 0.8 required), n = {len(mv)} MVDF features over "
             f"{mv.groupby(['dataset','seed']).ngroups} ETTh cells.\n\n")
    per = pm.groupby(["dataset", "seed"]).apply(lambda g: pd.Series({
        "MVDF med U R²": np.nanmedian(g[g.in_mvdf].U_r2), "MVDF med |ρU|": np.nanmedian(g[g.in_mvdf].absUs),
        "MVDF min U R²": np.nanmin(g[g.in_mvdf].U_r2),
        "MVDF frac dS(0) CI excl. 0": np.nanmean((g[g.in_mvdf]["dS_0_lo"] > 0) | (g[g.in_mvdf]["dS_0_hi"] < 0)),
        "MVDF frac dS(1.25) CI excl. 0": np.nanmean((g[g.in_mvdf]["dS_1.25_lo"] > 0) | (g[g.in_mvdf]["dS_1.25_hi"] < 0)),
        "MVDF med exceed@1.25": np.nanmedian(g[g.in_mvdf].frac_exceed_q99_active)})).reset_index()
    L.append("Per cell (ETTh):\n\n" + md_table(per))
    sup = feats[~feats.dataset.isin(PRIMARY)]
    if len(sup):
        s2 = sup[sup.in_mvdf]
        L.append(f"\nSupplementary (electricity/traffic, {sup.groupby(['dataset','seed']).ngroups} cells): "
                 f"MVDF median U fit R² {np.nanmedian(s2.U_r2):.3f}, median |ρ(λ,U)| "
                 f"{np.nanmedian(s2.U_spearman.abs()):.3f}.\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for a_, col, nm in ((ax[0], "U_r2", "linear-fit R² of D_unc(λ)"), (ax[1], "absUs", "|Spearman(λ, D_unc)|")):
            bins = np.linspace(0, 1, 21)
            a_.hist(pm[~pm.in_mvdf][col].dropna(), bins=bins, alpha=.6, density=True, label="rest of pool")
            a_.hist(pm[pm.in_mvdf][col].dropna(), bins=bins, alpha=.6, density=True, label="MVDF")
            a_.set_xlabel(nm); a_.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "E5a_hist.png"), dpi=130)
        L.append("\nHistogram: `E5a_hist.png`.\n\n")
    except Exception as ex:  # pragma: no cover
        L.append(f"\n(histogram not produced: {ex})\n\n")
    # ---- E5b
    L.append("## E5b — natural range (λ ≤ 1 only)\n\n")
    bb = b.copy()
    bb["full enr [CI]"] = [f"{e:.2f} [{l:.2f},{h:.2f}]" for e, l, h in zip(bb.full_enrichment, bb.full_enr_lo, bb.full_enr_hi)]
    bb["natural enr [CI]"] = [f"{e:.2f} [{l:.2f},{h:.2f}]" for e, l, h in zip(bb.nat_enrichment, bb.nat_enr_lo, bb.nat_enr_hi)]
    L.append(md_table(bb[["dataset", "seed", "jaccard_vs_full", "rho_U5_vs_U", "full enr [CI]",
                          "full_enr_ci_excludes_1", "natural enr [CI]", "nat_enr_ci_excludes_1",
                          "nat_enr_p_hypergeom"]]))
    pb = b[b.dataset.isin(PRIMARY)]
    full_cells = set(map(tuple, pb[pb.full_enr_ci_excludes_1][["dataset", "seed"]].values))
    nat_cells = set(map(tuple, pb[pb.nat_enr_ci_excludes_1][["dataset", "seed"]].values))
    c5b_a = full_cells <= nat_cells
    c5b_b = float(np.median(pb.jaccard_vs_full)) >= 0.5
    L.append(f"\n**C5b: {'PASS' if (c5b_a and c5b_b) else 'FAIL'}** — full-λ CI excludes 1 on "
             f"{len(full_cells)}/{len(pb)} ETTh cells, natural-range on {len(nat_cells)}/{len(pb)}; "
             f"every full-λ passing cell also passes in natural range: {c5b_a}. Median Jaccard "
             f"(natural vs full MVDF) = {np.median(pb.jaccard_vs_full):.2f} "
             f"[{pb.jaccard_vs_full.min():.2f}, {pb.jaccard_vs_full.max():.2f}] (≥ 0.5 required).\n\n")
    # ---- E5c
    c5c = None
    if len(ic):
        L.append("## E5c — interchange (resample) intervention\n\n")
        g = ic.groupby(["dataset", "seed"]).agg(n_feat=("feature", "size"), sign_agree=("sign_agree", "mean"),
                                                 n_eval_med=("n_eval", "median"),
                                                 dS_med=("dS_top_minus_bot", "median"),
                                                 dS_ci_excl0=("dS_lo", lambda v: np.nan)).reset_index()
        g["dS CI excl 0 (frac)"] = ic.groupby(["dataset", "seed"]).apply(
            lambda x: np.mean((x.dS_lo > 0) | (x.dS_hi < 0))).values
        g["implied/observed dS (median)"] = ic.groupby(["dataset", "seed"]).apply(
            lambda x: np.nanmedian(x.U_implied_dS / x.dS_top_minus_bot)).values
        g = g.drop(columns=["dS_ci_excl0"])
        L.append(md_table(g))
        res = {}
        for ds in PRIMARY:
            gd = g[g.dataset == ds]
            if not len(gd):
                continue
            k = int((gd.sign_agree >= 0.70).sum()); N = len(gd)
            res[ds] = (k, N, k >= pass_threshold(N))
        c5c = all(v[2] for v in res.values())
        L.append(f"\n**C5c: {'PASS' if c5c else 'FAIL'}** — cells with ≥ 70% sign agreement: " +
                 "; ".join(f"{ds} {k}/{N} (need {pass_threshold(N)})" for ds, (k, N, _) in res.items()) + ".\n\n")
    json.dump(dict(C5a=c5a, C5b=bool(c5b_a and c5b_b), C5b_cells=bool(c5b_a), C5b_jaccard=bool(c5b_b), C5c=c5c),
              open(os.path.join(OUT, "CRITERIA_E5.json"), "w"), indent=2)
    open(os.path.join(OUT, "_auto_RESULT.md"), "w").write("".join(L))
    print("".join(L)[-2500:])


if __name__ == "__main__":
    main()
