"""E4 report: C4.1-C4.3 from AGG_E4_rj_robustness.csv.  python -m camera_ready.report_e4"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from camera_ready.common import CR, pass_threshold, md_table, provenance

OUT = os.path.join(CR, "E4")
PRIMARY = ("ETTh1", "ETTh2")
DEFS = ["isotonic", "binned", "spline", "linear", "rank", "none"]
NAMES = {"isotonic": "1 isotonic (reference)", "binned": "2 binned mean", "spline": "3 smoothing spline (GCV)",
         "linear": "4 linear (contrast)", "rank": "5 rank residual", "none": "6 none, R=U (contrast)"}


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(OUT, "AGG_E4_rj_robustness.csv"))
    loc = df[df.source == "local"]
    pm = loc[loc.dataset.isin(PRIMARY)]
    n_cells = pm.groupby(["dataset", "seed"]).ngroups
    iso_pass = int(pm[(pm.definition == "isotonic")].enr_ci_excludes_1.sum())
    rows, crit = [], {}
    for d in DEFS:
        q = pm[pm.definition == d]
        r = {"definition": NAMES[d],
             "ρ(R, R_iso) median": f"{q.rho_vs_iso.median():.2f}",
             "Jaccard median [min,max]": f"{q.jaccard_vs_iso.median():.2f} [{q.jaccard_vs_iso.min():.2f},{q.jaccard_vs_iso.max():.2f}]",
             "enrichment median [min,max]": f"{q.enrichment.median():.2f} [{q.enrichment.min():.2f},{q.enrichment.max():.2f}]",
             "enr. CI > 1 (cells)": f"{int(q.enr_ci_excludes_1.sum())}/{len(q)}"}
        nulls = []
        for ds in PRIMARY:
            qd = q[q.dataset == ds]
            N = len(qd); thr = pass_threshold(N)
            km, kc = int(qd.marg_null_holds.sum()), int(qd.cond_null_holds.sum())
            r[f"{ds} R² null marg/active"] = f"{km}/{N}, {kc}/{N}"
            nulls.append(km >= thr and kc >= thr)
        rows.append(r)
        crit[d] = dict(jaccard_median=float(q.jaccard_vs_iso.median()),
                       enr_pass=int(q.enr_ci_excludes_1.sum()), null_ok=bool(all(nulls)))
    T = pd.DataFrame(rows)
    c41 = all(crit[d]["jaccard_median"] >= 0.5 for d in ("binned", "spline", "rank"))
    c42 = all(crit[d]["enr_pass"] >= iso_pass for d in ("binned", "spline", "rank"))
    c43 = all(crit[d]["null_ok"] for d in DEFS)
    json.dump(dict(C4_1=c41, C4_2=c42, C4_3=c43, iso_enr_pass=iso_pass, per_def=crit),
              open(os.path.join(OUT, "CRITERIA_E4.json"), "w"), indent=2)
    L = ["# E4 — Robustness of causal specificity R_j\n\n",
         provenance("python -m camera_ready.e4_rj; python -m camera_ready.report_e4",
                    {ds: sorted(loc[loc.dataset == ds].seed.unique().tolist()) for ds in loc.dataset.unique()},
                    sorted(loc.dataset.unique()), t0), "\n"]
    L.append("## Pre-registered criteria\n\n- **C4.1 Selection robust:** median Jaccard vs isotonic ≥ 0.5 "
             "across seeds for definitions 2, 3, 5.\n- **C4.2 Enrichment robust:** type-B enrichment > 1 with "
             "CI excluding 1 on ≥ the same number of (dataset, seed) cells as the isotonic definition, for "
             "definitions 2, 3, 5.\n- **C4.3 Null robust:** best held-out R² ≤ 0 (or CI includes 0) on ≥ 4/5 "
             "seeds for every definition.\n- Definitions 4 and 6 are contrasts, not pass/fail.\n\n")
    L.append(f"## Result (ETTh1 + ETTh2, {n_cells} cells)\n\n" + md_table(T))
    L.append(f"\n- **C4.1 {'PASS' if c41 else 'FAIL'}** — median Jaccard: binned {crit['binned']['jaccard_median']:.2f}, "
             f"spline {crit['spline']['jaccard_median']:.2f}, rank {crit['rank']['jaccard_median']:.2f}.\n")
    L.append(f"- **C4.2 {'PASS' if c42 else 'FAIL'}** — cells with enrichment CI > 1: isotonic {iso_pass}, binned "
             f"{crit['binned']['enr_pass']}, spline {crit['spline']['enr_pass']}, rank {crit['rank']['enr_pass']} "
             f"(of {n_cells}).\n")
    L.append(f"- **C4.3 {'PASS' if c43 else 'FAIL'}** — " + ", ".join(
        f"{d}: {'ok' if crit[d]['null_ok'] else 'fails'}" for d in DEFS) + ".\n\n")
    if (~loc.dataset.isin(PRIMARY)).any():
        s = loc[~loc.dataset.isin(PRIMARY)]
        g = s.groupby(["dataset", "definition"]).agg(J_med=("jaccard_vs_iso", "median"),
                                                     enr_med=("enrichment", "median"),
                                                     enr_ci_gt1=("enr_ci_excludes_1", "sum"),
                                                     marg_null=("marg_null_holds", "sum"),
                                                     n=("seed", "size")).reset_index()
        L.append("## Supplementary: electricity, traffic (local models)\n\n" + md_table(g))
    v2 = df[df.source == "v2_kaggle"]
    if len(v2):
        g = v2.pivot_table(index="dataset", columns="definition", values="enrichment").round(2)
        j = v2.pivot_table(index="dataset", columns="definition", values="jaccard_vs_iso").round(2)
        L.append("\n## Supplementary: v2_runs Kaggle single-seed candidate tables (8 datasets)\n\n"
                 f"Isotonic definition reproduces the stored MVDF set on "
                 f"{int(v2[v2.definition == 'isotonic'].iso_reproduces_stored_mvdf.sum())}/"
                 f"{int((v2.definition == 'isotonic').sum())} datasets.\n\nEnrichment:\n\n"
                 + md_table(g.reset_index()) + "\nJaccard vs isotonic:\n\n" + md_table(j.reset_index()))
    open(os.path.join(OUT, "_auto_RESULT.md"), "w").write("".join(L))
    print("".join(L)[:3000])


if __name__ == "__main__":
    main()
