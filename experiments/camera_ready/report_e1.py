"""E1 report: Phase 3/4 + E3 on electricity and traffic, and the 4-dataset paper table.
Needs: results/camera_ready/E1/AGG_phase{3,4}.csv (phase34 on elec/traffic), results/camera_ready/E2/AGG_phase{3,4}.csv
(ETTh), E3 aggregates for ETTh (E3/) and elec/traffic (E1/, prefix E1_E3).
  python -m experiments.camera_ready.report_e1
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from experiments.camera_ready.common import CR, pass_threshold, summ, md_table, provenance, available_seeds

OUT = os.path.join(CR, "E1")
NEW = ("electricity", "traffic")
EXANTE = ["F1", "F1p", "F2", "F3", "F6"]


def fmt(sm, f="{:+.2f}"):
    return f"{f.format(sm['median'])} [{f.format(sm['min'])}, {f.format(sm['max'])}]"


def main():
    t0 = time.time()
    p3 = pd.concat([pd.read_csv(os.path.join(CR, d, "AGG_phase3.csv")) for d in ("E2", "E1")], ignore_index=True)
    p4 = pd.concat([pd.read_csv(os.path.join(CR, d, "AGG_phase4.csv")) for d in ("E2", "E1")], ignore_index=True)
    pr = pd.concat([pd.read_csv(os.path.join(CR, "E3", "AGG_E3_predictability.csv")),
                    pd.read_csv(os.path.join(OUT, "AGG_E1_E3_predictability.csv"))], ignore_index=True)
    gp = pd.concat([pd.read_csv(os.path.join(CR, "E3", "AGG_E3_gap_recovery.csv")),
                    pd.read_csv(os.path.join(OUT, "AGG_E1_E3_gap_recovery.csv"))], ignore_index=True)
    crit3 = json.load(open(os.path.join(OUT, "CRITERIA_E1_E3.json")))
    for ds in NEW:
        dd = os.path.join(OUT, ds); os.makedirs(dd, exist_ok=True)
        p3[p3.dataset == ds].to_csv(os.path.join(dd, "AGG_E1_phase3.csv"), index=False)
        p4[p4.dataset == ds].to_csv(os.path.join(dd, "AGG_E1_phase4.csv"), index=False)
        pr[pr.dataset == ds].to_csv(os.path.join(dd, "AGG_E3_predictability.csv"), index=False)
        gp[gp.dataset == ds].to_csv(os.path.join(dd, "AGG_E3_gap_recovery.csv"), index=False)
    L = ["# E1 — Phases 3 + 4 (+ E3) on electricity and traffic\n\n",
         provenance("python -m experiments.camera_ready.phase34 --datasets electricity traffic --out results/camera_ready/E1; "
                    "python -m experiments.camera_ready.e3_exante --datasets electricity traffic; "
                    "python -m experiments.camera_ready.report_e3 --datasets electricity traffic --out results/camera_ready/E1 "
                    "--prefix E1_E3; python -m experiments.camera_ready.report_e1",
                    {d: available_seeds(d) for d in NEW}, list(NEW), t0), "\n"]
    L.append(open(os.path.join(OUT, "PREREG.md")).read().replace("# E1 — Phases 3 + 4 (+ E3) on electricity and traffic: pre-registration", "## Pre-registration (verbatim)") + "\n")
    # Phase 3
    L.append("## Phase 3 (original definitions), per seed\n\n")
    q = p3[p3.dataset.isin(NEW)].copy()
    q["marginal best R² [block CI]"] = [f"{a:+.3f} [{b:+.3f},{c:+.3f}]" for a, b, c in zip(q.marg_best_r2, q.marg_best_r2_lo, q.marg_best_r2_hi)]
    q["active best R² [block CI]"] = [f"{a:+.3f} [{b:+.3f},{c:+.3f}]" for a, b, c in zip(q.cond_best_r2, q.cond_best_r2_lo, q.cond_best_r2_hi)]
    L.append(md_table(q[["dataset", "seed", "n_fit", "n_eval", "mvdf_activation_rate", "n_active_windows_eval",
                         "marginal best R² [block CI]", "active best R² [block CI]"]]))
    c11 = {}
    for ds in NEW:
        d = q[q.dataset == ds]; N = len(d); thr = 2 if N == 3 else pass_threshold(N)
        km = int(((d.marg_best_r2 <= 0) | (d.marg_best_r2_lo <= 0)).sum())
        kc = int(((d.cond_best_r2 <= 0) | (d.cond_best_r2_lo <= 0) | d.cond_best_r2.isna()).sum())
        c11[ds] = dict(marg=f"{km}/{N}", active=f"{kc}/{N}", need=thr, pass_=bool(km >= thr and kc >= thr))
    L.append("\n**C1.1:** " + "; ".join(f"{ds}: marginal {v['marg']}, active {v['active']} seeds with CI ∋ 0 or "
                                        f"≤ 0 (need {v['need']}) → **{'PASS' if v['pass_'] else 'FAIL'}**"
                                        for ds, v in c11.items()) + "\n\n")
    # Phase 4
    L.append("## Phase 4 oracle (original construction), per seed\n\n")
    q4 = p4[p4.dataset.isin(NEW)].copy()
    for b in ("id", "shift"):
        q4[f"gap % {b} [block CI]"] = [f"{100*a:.1f} [{100*l:.1f},{100*h:.1f}]" for a, l, h in
                                       zip(q4[f"ratio_{b}"], q4[f"ratio_{b}_ci_lo"], q4[f"ratio_{b}_ci_hi"])]
    L.append(md_table(q4[["dataset", "seed", "n_id", "n_shift", "naive_id_coverage", "oracle_id_coverage",
                          "gap % id [block CI]", "naive_shift_coverage", "gap % shift [block CI]"]]))
    c12 = {}
    for ds in NEW:
        d = q4[q4.dataset == ds]
        ok = bool(((d.ratio_id_ci_lo > 0) & (d.ratio_shift_ci_lo > 0)).all())
        c12[ds] = dict(pass_=ok, id=fmt(summ(100 * d.ratio_id), "{:.1f}"), shift=fmt(summ(100 * d.ratio_shift), "{:.1f}"))
    L.append("\n**C1.2:** " + "; ".join(f"{ds}: gap ID {v['id']}%, shift {v['shift']}% → **{'PASS' if v['pass_'] else 'FAIL'}** "
                                        "(CI excludes 0 on every seed, both blocks)" for ds, v in c12.items()) + "\n\n")
    # E3 criteria
    L.append("## C1.3 — E3 criteria on the new datasets\n\n")
    for ds in NEW:
        if ds not in crit3:
            continue
        c = crit3[ds]["block96"]
        L.append(f"- **{ds}** (N = {c['N']}, need {c['threshold']}): C3.1 {'PASS' if c['C3_1']['pass_'] else 'FAIL'} "
                 f"(sets passing {c['C3_1']['sets_passing']}; lenient any-set seeds {c['C3_1']['lenient_any_set_seeds']}); "
                 f"C3.2 {'PASS' if c['C3_2']['pass_'] else 'FAIL'}; C3.3 {'PASS' if c.get('C3_3', {}).get('pass_') else 'FAIL'} "
                 f"({c.get('C3_3', {}).get('seeds_ci_contains_or_below_0')}/{c['N']}); C3.4 "
                 f"{'PASS' if c['C3_4']['pass_'] else 'FAIL'} (block {c['C3_4']['block']}, GRR {c['C3_4']['GRR_per_seed']}, "
                 f"coverage {c['C3_4']['coverage_per_seed']}); C3.5 {'PASS' if c['C3_5']['pass_'] else 'FAIL'} "
                 f"({c['C3_5']['counts']}). Predictive counts: {c['predictive_counts']}.\n")
    L.append("\nFull E3 tables for these datasets: `_tables_E1_E3.md`.\n\n")
    # 4-dataset table
    rows = []
    for ds in ("ETTh1", "ETTh2", "electricity", "traffic"):
        a = p3[p3.dataset == ds]; b = p4[p4.dataset == ds]
        sel = pr[(pr.dataset == ds) & (pr.split == "ID") & pr.cv_selected & pr.feature_set.isin(EXANTE)]
        best = sel.loc[sel.groupby("seed").cv_mse.idxmin()] if len(sel) else sel
        blk = "ID"
        if ds == "traffic":
            nv = gp[(gp.dataset == ds) & (gp.split == "ID") & (gp.feature_set == "naive_matched")].coverage
            blk = "shift" if (nv > 0.97).any() else "ID"
        g = gp[(gp.dataset == ds) & (gp.split == blk) & gp.is_best_exante]
        gv = g[g.coverage_valid & (g.W_naive > g.W_oracle)]
        rows.append({"dataset": ds, "N": len(a),
                     "MVDF best R² (marg.)": fmt(summ(a.marg_best_r2)),
                     "oracle gap % (ID)": fmt(summ(100 * b.ratio_id), "{:.1f}"),
                     "best ex-ante R² (CV-sel.)": fmt(summ(best.R2)) if len(best) else "–",
                     "ex-ante predictive seeds": f"{int(((best.R2_lo > 0) & (best.rho_lo > 0)).sum())}/{len(best)}" if len(best) else "–",
                     f"GRR best ex-ante": (f"{blk}: valid on {len(gv)}/{len(g)}; " +
                                           (fmt(summ(gv.GRR)) if len(gv) else "none valid"))})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "PAPER_TABLE_4datasets.csv"), index=False)
    L.append("## Four-dataset summary\n\n" + md_table(T))
    json.dump(dict(C1_1=c11, C1_2=c12, C1_3={ds: crit3.get(ds) for ds in NEW}),
              open(os.path.join(OUT, "CRITERIA_E1.json"), "w"), indent=2, default=str)
    open(os.path.join(OUT, "_auto_RESULT.md"), "w").write("".join(L))
    print(T.to_string())


if __name__ == "__main__":
    main()
