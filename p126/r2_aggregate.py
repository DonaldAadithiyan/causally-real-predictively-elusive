"""Round 2: aggregate per-WP outputs into compact summary tables (runs/p126_round2/summary/).
Reads only saved outputs; computes nothing new except group summaries."""
import glob, json, os, sys
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "p126_round2")
S = os.path.join(R, "summary"); os.makedirs(S, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)

def cat(pattern, base):
    fs = [f for f in sorted(glob.glob(os.path.join(R, pattern))) if not f.endswith("_ETTh10.csv") and "ETTh20" not in f]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def wp1():
    cs = pd.concat([cat("wp1/continuous_scoring.csv", R), cat("wp6/wp1/*_continuous_scoring.csv", R)], ignore_index=True)
    cs = cs.drop_duplicates(["dataset", "seed", "protocol", "segment", "method"], keep="last")
    piv = cs.pivot_table(index=["dataset", "protocol", "segment", "method"], columns="seed",
                         values="red_vs_naive_matched").round(3)
    cov = cs.groupby(["dataset", "protocol", "segment", "method"])["cov"].agg(["min", "max"]).round(3)
    out = piv.join(cov)
    out.to_csv(os.path.join(S, "wp1_continuous_red.csv"))
    sb = pd.concat([cat("wp1/standard_baselines.csv", R), cat("wp6/wp1/*_standard_baselines.csv", R)], ignore_index=True)
    sbs = sb.groupby(["dataset", "segment", "method", "update"]).agg(
        cov_min=("cov", "min"), cov_max=("cov", "max"), red_min=("red_vs_ref_matched", "min"),
        red_max=("red_vs_ref_matched", "max"), ref=("like_for_like_ref", "first")).round(3)
    sbs.to_csv(os.path.join(S, "wp1_standard_baselines.csv"))
    rg = pd.concat([cat("wp1/regime_gating.csv", R), cat("wp6/wp1/*_regime_gating.csv", R)], ignore_index=True)
    rg.round(3).to_csv(os.path.join(S, "wp1_regime_gating.csv"), index=False)
    lo = pd.concat([cat("wp1/localization.csv", R), cat("wp6/wp1/*_localization.csv", R)], ignore_index=True)
    lo.round(3).to_csv(os.path.join(S, "wp1_localization.csv"), index=False)
    return out, sbs, rg, lo

def wp3():
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "wp3", "*", "summary.json"))):
        d = json.load(open(f)); sn = d["selection_null"]; sh = d["split_half"]; oo = d["out_of_sample"]; ne = d["necessity"]
        rows.append(dict(tag=d["tag"], enrich=d["observed_enrichment"], permU_q95=sn["permU_q95"], permU_p=sn["permU_p"],
                         randdir_q95=sn["randdir_q95"], randdir_p=sn["randdir_p"],
                         exceeds=sn["exceeds_selection_alone_permU"] and sn["exceeds_selection_alone_randdir"],
                         zU_mvdf=d["z_U_randdir"]["mvdf_median"], zU_pool=d["z_U_randdir"]["pool_median"],
                         frac_mvdf_absz2=d["z_U_randdir"]["frac_mvdf_absz_gt2"],
                         U_over_sdS=d["effect_size"]["mvdf_median_U_over_sdS"],
                         rho_P=sh["rho_P"], rho_U=sh["rho_U"], rho_R=sh["rho_R"], topk=sh["topk_overlap"],
                         topk_p=sh["overlap_p_hypergeom"], oos_pct_B=oo["pct_B"], oos_pct_eval=oo["pct_eval"],
                         nec_dS_over_sd=ne["dS_over_sdS"], nec_z_set=ne["z_vs_set"], nec_z_dir=ne["z_vs_dir"]))
    t = pd.DataFrame(rows).round(3); t.to_csv(os.path.join(S, "wp3_summary.csv"), index=False); return t

def stepzero():
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "wp6", "stepzero", "*_rule.json"))):
        d = json.load(open(f))
        for cell in ("ID_static", "ID_aci", "shift_static", "shift_aci"):
            c = d[cell]
            rows.append(dict(dataset=d["dataset"], seed=d["seed"], best=d["best_predictor"], cell=cell,
                             red=c["red"], red_lo=c["red_ci"][0], red_hi=c["red_ci"][1], rho=c["spearman"],
                             rho_lo=c["spearman_ci"][0], rho_hi=c["spearman_ci"][1], n_blocks=c["n_blocks"],
                             base_cov=c["base_cov"], passes=c["passes"],
                             ci_degenerate=bool(c["n_blocks"] < 2)))
    t = pd.DataFrame(rows).round(3); t.to_csv(os.path.join(S, "stepzero_cells.csv"), index=False)
    v = t.groupby(["dataset", "cell"]).agg(n_seeds=("seed", "nunique"), n_pass=("passes", "sum"),
                                           min_blocks=("n_blocks", "min")).reset_index()
    v["rule_pass"] = np.where(v.n_seeds >= 3, v.n_pass >= 2, v.n_pass >= 1)
    v["valid_cell"] = v.min_blocks >= 2
    v.to_csv(os.path.join(S, "stepzero_verdict.csv"), index=False); return t, v

def wp4():
    loc = cat("wp4/locate.csv", R); iv = cat("wp4/interventions.csv", R)
    loc.round(3).to_csv(os.path.join(S, "wp4_locate.csv"), index=False)
    g = iv.copy(); g["ctrl_both"] = (g.z_slope_S_normnull.abs() > 2) & (g.z_slope_S_varnull.abs() > 2)
    g["projout_both"] = (g.z_dS_projout_normnull.abs() > 2) & (g.z_dS_projout_varnull.abs() > 2)
    keep = ["dataset", "seed", "row", "group", "slope_S", "z_slope_S_normnull", "z_slope_S_varnull", "dS_projout",
            "z_dS_projout_normnull", "z_dS_projout_varnull", "slope_loss", "z_slope_loss_varnull", "ctrl_both", "projout_both"]
    g[keep].round(4).to_csv(os.path.join(S, "wp4_grid.csv"), index=False); return loc, g[keep]

def wp2():
    import re
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "wp2", "B", "*_B.csv"))):
        b = pd.read_csv(f); tag = os.path.basename(f)[:-6]
        m = re.match(r"synthR([\d.]+)P([\d.]+)_seed(\d+)", tag)
        ratio, p, seed = (float(m[1]), float(m[2]), int(m[3])) if m else (5.0, 0.995, int(tag.split("seed")[1]))
        for seg, g in b.groupby("segment"):
            def get(pred, col):
                x = g[g.predictor == pred]; return x[col].iloc[0] if len(x) else np.nan
            best = g[g.selected_best == True].predictor.iloc[0] if (g.selected_best == True).any() else None
            rows.append(dict(tag=tag, ratio=ratio, p_stay=p, seed=seed, segment=seg,
                             oracle_ratio=get("ORACLE_outcome_binned", "oracle_ratio"),
                             true_rho=get("TRUE_mean_sigma", "spearman"), true_red=get("TRUE_mean_sigma", "bin10_red"),
                             true_red_ci=get("TRUE_mean_sigma", "bin10_red_ci"),
                             best=best, best_rho=get(best, "spearman"), best_red=get(best, "bin10_red"),
                             best_red_ci=get(best, "bin10_red_ci"),
                             mvdf_rho=get("rep_mvdf", "spearman"), mvdf_red=get("rep_mvdf", "bin10_red"),
                             mvdf_red_ci=get("rep_mvdf", "bin10_red_ci"),
                             base_cov=get(best, "base_cov"), n_blocks=get(best, "n_blocks_test")))
    t = pd.DataFrame(rows).sort_values(["segment", "ratio", "p_stay", "seed"])
    for c in ("true_red_ci", "best_red_ci", "mvdf_red_ci"):
        t[c] = t[c].apply(lambda v: "[%.3f, %.3f]" % tuple(json.loads(v)) if isinstance(v, str) else v)
    t = t.round(3); t.to_csv(os.path.join(S, "wp2_dial.csv"), index=False); return t

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in [("wp2", wp2), ("wp1", wp1), ("wp3", wp3), ("stepzero", stepzero), ("wp4", wp4)]:
        if only and name not in only: continue
        res = fn(); res = res if isinstance(res, tuple) else (res,)
        for r in res:
            print(f"==== {name}"); print(r.to_string())
