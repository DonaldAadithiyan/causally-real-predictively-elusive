"""E2 report: health, the four core metrics over all seeds, C2.1, C2.2, SEED_TABLE.md.
Requires: phase34 output in results/camera_ready/E2 (ETTh) and results/camera_ready/E1 (elec/traffic),
e2_scorer output, E5a cells (for enrichment + its CI).
  python -m experiments.camera_ready.report_e2
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from experiments.camera_ready.common import (CR, meta, enrichment_stats, two_level, sign_test, pass_threshold,
                                 summ, md_table, provenance, available_seeds, N_BOOT, BOOT_SEED)
from mechconf.validation.difficulty_specificity import feature_type_table

OUT = os.path.join(CR, "E2")
DS_ALL = ["ETTh1", "ETTh2", "electricity", "traffic"]


def enr_boot(sel_types, pool_types, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    num = sel_types[rng.integers(0, len(sel_types), (n_boot, len(sel_types)))].mean(1)
    den = pool_types[rng.integers(0, len(pool_types), (n_boot, len(pool_types)))].mean(1)
    return num / np.where(den > 0, den, np.nan)


def enrichment_table():
    rows, boots = [], {}
    for ds in DS_ALL:
        for s in available_seeds(ds):
            d = os.path.join(CR, "E5", "cells", f"{ds}_seed{s}", "features.csv")
            if not os.path.exists(d):
                continue
            f = pd.read_csv(d)
            pool = f.feature.to_numpy(int)
            ftab = feature_type_table(f.P_j.to_numpy(float), f.U_j.to_numpy(float), feature_indices=pool)
            mv = f.feature[f.in_mvdf].to_numpy(int)
            e = enrichment_stats(mv, ftab, pool)
            m = meta(ds, s)
            B = set(ftab["cells"]["B"])
            boots[(ds, s)] = enr_boot(np.array([j in B for j in sorted(mv)], float),
                                      np.array([j in B for j in sorted(pool)], float))
            rows.append(dict(dataset=ds, seed=s, stored_enrichment=m.get("enrichment_typeB"), **e))
    return pd.DataFrame(rows), boots


def health():
    rows = []
    for ds in DS_ALL:
        seeds = available_seeds(ds)
        if not seeds:
            continue
        m0 = meta(ds, 0)
        for s in seeds:
            m = meta(ds, s)
            unhealthy = (m["sae_fvu"] > 0.05) or (m["val_mse"] > 1.5 * m0["val_mse"])
            rows.append(dict(dataset=ds, seed=s, val_mse=m["val_mse"], sae_fvu=m["sae_fvu"],
                             val_mse_ratio_to_seed0=m["val_mse"] / m0["val_mse"],
                             device=m.get("device", "cpu"), wall_clock_s=m.get("wall_clock_s", np.nan),
                             unhealthy=unhealthy))
    return pd.DataFrame(rows)


def per_metric(df, ds, val, lo, hi, boots=None, passfn=None, fmt="{:+.3f}", seeds_old=(0, 1, 2)):
    d = df[df.dataset == ds].sort_values("seed")
    out = {"dataset": ds}
    for _, r in d.iterrows():
        out[f"s{int(r.seed)}"] = (fmt.format(r[val]) + (f" [{fmt.format(r[lo])},{fmt.format(r[hi])}]"
                                                         if lo else ""))
    sm = summ(d[val]); old = summ(d[d.seed.isin(seeds_old)][val])
    out["median [min,max]"] = f"{fmt.format(sm['median'])} [{fmt.format(sm['min'])},{fmt.format(sm['max'])}]"
    pc = (np.nan, np.nan)
    if boots is not None:
        arr = [boots[k] for k in boots if k[0] == ds]
        pc = two_level(arr)
        out["pooled 2-level CI"] = f"[{fmt.format(pc[0])},{fmt.format(pc[1])}]"
    k = None
    if passfn is not None:
        k = int(sum(passfn(r) for _, r in d.iterrows()))
        out["pass k/N"] = f"{k}/{len(d)}"
        out["sign test p"] = f"{sign_test(k, len(d)):.3g}"
    stats_ = dict(N=len(d), k=k, median_all=sm["median"], median_3=old["median"],
                  pooled_lo=pc[0], pooled_hi=pc[1])
    return out, stats_


def main():
    t0 = time.time()
    H = health()
    H.to_csv(os.path.join(OUT, "AGG_E2_health.csv"), index=False)
    E, Eb = enrichment_table()
    E.to_csv(os.path.join(OUT, "AGG_E2_enrichment.csv"), index=False)
    p3 = pd.concat([pd.read_csv(os.path.join(CR, d, "AGG_phase3.csv")) for d in ("E2", "E1")
                    if os.path.exists(os.path.join(CR, d, "AGG_phase3.csv"))], ignore_index=True)
    p4 = pd.concat([pd.read_csv(os.path.join(CR, d, "AGG_phase4.csv")) for d in ("E2", "E1")
                    if os.path.exists(os.path.join(CR, d, "AGG_phase4.csv"))], ignore_index=True)
    b34 = {}
    for d in ("E2", "E1"):
        p = os.path.join(CR, d, "_boot_phase34.pkl")
        if os.path.exists(p):
            x = pd.read_pickle(p)
            b34.setdefault("b3", {}).update(x["b3"]); b34.setdefault("b4", {}).update(x["b4"])
    p3.to_csv(os.path.join(OUT, "AGG_E2_phase3.csv"), index=False)
    p4.to_csv(os.path.join(OUT, "AGG_E2_phase4.csv"), index=False)
    sc = pd.read_csv(os.path.join(OUT, "AGG_E2_scorer_vs_naive.csv"))
    scb = pd.read_pickle(os.path.join(OUT, "AGG_E2_scorer_vs_naive_boot.pkl"))
    datasets = [d for d in DS_ALL if d in set(p3.dataset)]

    L = ["# E2 — More seeds for the core results\n\n",
         provenance("python -m experiments.camera_ready.train_seeds ...; python -m experiments.camera_ready.phase34 ...; "
                    "python -m experiments.camera_ready.e2_scorer ...; python -m experiments.camera_ready.report_e2",
                    {d: available_seeds(d) for d in datasets}, datasets, t0)]
    L.append("\n## Pre-registered criteria\n\n- **C2.1:** every claim currently stated as \"3/3 seeds\" "
             "holds on ≥ 4/5 seeds (≥ 8/10 where N = 10). Any claim that drops below this is reported as "
             "weakened, and the paper text changes to match.\n- **C2.2:** no headline median moves by more "
             "than its own pooled CI half-width when going from 3 to 5 (or 10) seeds.\n- Health rule: "
             "FVU > 0.05 or val MSE > 1.5 × seed-0 val MSE ⇒ unhealthy (reported, never dropped).\n\n")
    L.append("## Seed health\n\n" + md_table(H, fmt="{:.4f}"))
    L.append(f"\nUnhealthy seeds: {int(H.unhealthy.sum())} "
             f"({', '.join(f'{r.dataset} s{r.seed}' for r in H[H.unhealthy].itertuples()) or 'none'}). "
             f"FVU range over new seeds: {H[H.seed >= 3].sae_fvu.min():.4f}–{H[H.seed >= 3].sae_fvu.max():.4f} "
             f"(existing range 0.003–0.033).\n\n")
    stats_all = []
    blocks = []

    def block(title, df, val, lo, hi, boots, passfn, claim, fmt="{:+.3f}", dsets=datasets):
        rows = []
        for ds in dsets:
            if ds not in set(df.dataset):
                continue
            r, st = per_metric(df, ds, val, lo, hi, boots, passfn, fmt)
            rows.append(r)
            st.update(metric=title, dataset=ds, claim=claim)
            stats_all.append(st)
        blocks.append(f"\n### {title}\n\n*Claim tested:* {claim}\n\n" + md_table(pd.DataFrame(rows)))

    # 1. enrichment
    block("1. Type-B enrichment of MVDF (H1c)", E, "enrichment", "enr_lo", "enr_hi", Eb,
          lambda r: r.enrichment >= 1.0, "MVDF enrichment ≥ 1.0× on every run (Table 2: 12/12)", fmt="{:.2f}")
    block("1b. Type-B enrichment, CI excludes 1 (added uncertainty; not a paper claim)", E, "enrichment",
          "enr_lo", "enr_hi", None, lambda r: r.enr_lo > 1.0, "enrichment CI > 1", fmt="{:.2f}")
    # 2. phase 3
    b3 = {k: v for k, v in b34.get("b3", {}).items()}
    block("2a. Phase 3 best held-out R², marginal", p3, "marg_best_r2", "marg_best_r2_lo",
          "marg_best_r2_hi", b3, lambda r: r.marg_best_r2 < 0,
          "best held-out R² negative on every seed (Table 4, 3/3)")
    block("2b. Phase 3 best held-out R², active windows", p3, "cond_best_r2", "cond_best_r2_lo",
          "cond_best_r2_hi", None, lambda r: r.cond_best_r2 < 0, "active-window best R² negative on every seed")
    # 3. phase 4
    b4i = {k: v["id"] for k, v in b34.get("b4", {}).items()}
    b4s = {k: v["shift"] for k, v in b34.get("b4", {}).items()}
    block("3a. Oracle gap ratio (W_naive − W_oracle)/W_naive, ID", p4, "ratio_id", "ratio_id_ci_lo",
          "ratio_id_ci_hi", b4i, lambda r: r.ratio_id_ci_lo > 0, "oracle gap CI excludes 0 on every seed (Table 6)")
    block("3b. Oracle gap ratio, shift", p4, "ratio_shift", "ratio_shift_ci_lo", "ratio_shift_ci_hi", b4s,
          lambda r: r.ratio_shift_ci_lo > 0, "oracle gap CI excludes 0 on every seed (Table 6)")
    # 4. scorer
    for mode in ("dense", "mvdf"):
        for blk in ("shift", "ID"):
            q = sc[(sc["mode"] == mode) & (sc.block == blk)].copy()
            bb = {(k[0], k[1]): v for k, v in scb.items() if k[2] == mode and k[3] == blk}
            block(f"4. Scorer width − naive_matched width, {mode}, {blk}", q, "width_minus_naive", "wmn_lo",
                  "wmn_hi", bb, lambda r: r.wmn_lo > 0,
                  f"naive_matched significantly narrower than {mode} (§5.6: 12/12)")
            cv = q[["dataset", "seed", "coverage", "naive_coverage"]]
            blocks.append(f"\nCoverage ({mode}, {blk}): " + "; ".join(
                f"{d} " + ", ".join(f"s{int(r.seed)} {r.coverage:.3f}/{r.naive_coverage:.3f}"
                                    for r in cv[cv.dataset == d].itertuples()) for d in cv.dataset.unique())
                          + " (scorer / naive_matched)\n")
    L += blocks
    S = pd.DataFrame(stats_all)
    S["thr"] = S.N.map(pass_threshold)
    S["C2_1_holds"] = S.k >= S.thr
    S["half_width"] = (S.pooled_hi - S.pooled_lo) / 2
    S["median_shift"] = (S.median_all - S.median_3).abs()
    S["C2_2_ok"] = np.where(S.half_width.notna(), S.median_shift <= S.half_width, np.nan)
    S.to_csv(os.path.join(OUT, "AGG_E2_criteria.csv"), index=False)
    L.append("\n## Criteria\n\n")
    show = S[["metric", "dataset", "N", "k", "thr", "C2_1_holds", "median_3", "median_all", "median_shift",
              "half_width", "C2_2_ok"]]
    L.append(md_table(show))
    open(os.path.join(OUT, "_auto_RESULT.md"), "w").write("".join(L))
    # SEED_TABLE
    T = H[["dataset", "seed", "val_mse", "sae_fvu"]].merge(
        E[["dataset", "seed", "enrichment", "enr_lo", "enr_hi"]], on=["dataset", "seed"], how="left").merge(
        p3[["dataset", "seed", "marg_best_r2", "marg_best_r2_lo", "marg_best_r2_hi", "cond_best_r2"]],
        on=["dataset", "seed"], how="left").merge(
        p4[["dataset", "seed", "ratio_id", "ratio_id_ci_lo", "ratio_id_ci_hi", "ratio_shift"]],
        on=["dataset", "seed"], how="left")
    for mode in ("dense", "mvdf"):
        q = sc[(sc["mode"] == mode) & (sc.block == "shift")][["dataset", "seed", "width_minus_naive", "coverage"]]
        T = T.merge(q.rename(columns={"width_minus_naive": f"dW_shift_{mode}", "coverage": f"cov_shift_{mode}"}),
                    on=["dataset", "seed"], how="left")
    T.to_csv(os.path.join(OUT, "SEED_TABLE.csv"), index=False)
    open(os.path.join(OUT, "SEED_TABLE.md"), "w").write(
        "# SEED_TABLE — every per-seed number the paper quotes (camera-ready)\n\n"
        "Enrichment CI: bootstrap over features (E2 PREREG). R² / oracle CIs: moving-block bootstrap, block 96. "
        "dW = scorer width − naive_matched width on E_eval_shift (v2_h6_topup protocol). Seeds 0–2 ETTh = "
        "the paper's stored runs (reproduced exactly); electricity/traffic 0–2 = round-2 local models.\n\n"
        + md_table(T, fmt="{:.3f}"))
    print(show.to_string())


if __name__ == "__main__":
    main()
