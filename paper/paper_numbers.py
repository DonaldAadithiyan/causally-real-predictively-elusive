"""Recompute every number printed in the camera-ready paper from the committed result files.

Each function returns the number as a short string. paper/paper_numbers.yaml maps each printed
number (copied from the LaTeX source) to a function; tests/test_paper_numbers.py normalises both
strings (drops $ \\ { } and spaces; treats --, en dash and minus alike) and asserts equality.
Rounding is half-up, as typeset.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from mechconf.paths import REPO_ROOT

R = os.path.join(REPO_ROOT, "results")
CR = os.path.join(R, "camera_ready")
V2 = os.path.join(R, "v2")
ETT = ["ETTh1", "ETTh2"]


# ------------------------------------------------------------------ formatting helpers
def rd(x, nd):
    """Round half-up to nd decimals."""
    q = Decimal(1).scaleb(-nd) if nd > 0 else Decimal(1)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def sg(x, nd):
    return ("+" if float(x) > 0 else "") + rd(x, nd)


def pct(x, nd=0):
    return rd(100 * float(x), nd)


def frac(k, n):
    return f"{int(k)}/{int(n)}"


def _csv(*p):
    return pd.read_csv(os.path.join(*p))


def _e2(name):
    return _csv(CR, "E2", f"AGG_E2_{name}.csv")


def _g4():
    d = _csv(CR, "G", "AGG_G4_L2_evidence.csv")
    d["all"] = (12 * d.oos_frac_mvdf + 108 * d.oos_frac_rest) / 120      # pool = 120 incl. 12 MVDF
    return d


def _pred():
    return pd.concat([_csv(CR, "E3", "AGG_E3_predictability.csv"),
                      _csv(CR, "E1", "AGG_E1_E3_predictability.csv")], ignore_index=True)


def _cvsel(ds, fs):
    p = _pred()
    return p[(p.dataset == ds) & (p.split == "ID") & p.cv_selected & (p.feature_set == fs)]


def _predictive(x):
    return int(((x.R2_lo > 0) & (x.rho_lo > 0)).sum())


def _slcp():
    return json.load(open(os.path.join(V2, "slcp_gate", "SLCP_gate_decision.json")))


# ------------------------------------------------------------------ abstract / intro
def l2_share_range():
    med = _g4().groupby("dataset")["all"].median()
    return f"{pct(med.min())}-{pct(med.max())}%"


def n_runs_r2_negative():
    a = _e2("phase3"); b = _csv(CR, "E6", "AGG_phase3.csv")
    return str(int((a.marg_best_r2 < 0).sum() + (b.marg_best_r2 < 0).sum()))


def n_itransformer_runs():
    return str(len(_e2("phase3")))


def oracle_bound_range():
    med = _e2("phase4").groupby("dataset").ratio_id.median()
    return f"{pct(med.min())}-{pct(med.max())}%"


# ------------------------------------------------------------------ method / setup
def layer_sweep_range():
    s = _csv(V2, "v2_runs", "v2_phase0", "layer_sweep.csv")      # PCA-based max |corr| per tap
    return f"{rd(s.max_abs_corr_pc.min(), 2)}-{rd(s.max_abs_corr_pc.max(), 2)}"


def layer_sweep_n_taps():
    return str(_csv(V2, "v2_runs", "v2_phase0", "layer_sweep.csv").tap.nunique())


def profile_rank_agreement():
    g = _csv(CR, "G", "AGG_G2_select_profile.csv")
    v = pd.concat([g.rho_Sdefs_id, g.rho_Sdefs_shift])
    return f"{rd(v.min(), 2)}-{rd(v.max(), 3)}"


def ks_all_below_005():
    ps = []
    for s in range(3):
        j = json.load(open(os.path.join(V2, "v2_confirm", f"v2_phase1_seed{s}", "run_log.json")))
        for ds in ["ETTh1", "ETTh2", "electricity", "traffic"]:
            ps.append(j["per_run"][f"{ds}_pre_head"]["env_split"]["ks_evalShift_vs_select"]["p"])
    return "p<0.05" if max(ps) < 0.05 else f"maxp{max(ps):.3g}"


def etth_id_max_coverage():
    s = _e2("scorer_vs_naive"); e = s[s.dataset.isin(ETT) & (s.block == "ID")]
    return rd(e.coverage.max(), 2)


def traffic_id_coverage():
    s = _e2("scorer_vs_naive"); t = s[(s.dataset == "traffic") & (s.block == "ID")]
    return rd(t.coverage.min(), 3) if t.coverage.min() == t.coverage.max() else "varies"


# ------------------------------------------------------------------ 5.1 L2 evidence (Table 2)
def l2_share(ds, who):
    d = _g4(); d = d[d.dataset == ds]
    v = d["all"] if who == "all" else d.oos_frac_mvdf
    return f"{pct(v.median())}[{pct(v.min())},{pct(v.max())}]"


def e5a_fit_r2():
    f = _csv(CR, "E5", "AGG_E5a_diagnostics.csv"); f = f[f.dataset.isin(ETT) & f.in_mvdf]
    return rd(f.U_r2.median(), 3)


def mvdf_heldout_wins():
    d = _g4(); return frac(d.oos_mvdf_gt_rest.sum(), len(d))


def mvdf_insample_wins():
    d = _g4(); return frac(d.in_mvdf_gt_rest.sum(), len(d))


def enrichment_median(ds):
    e = _e2("enrichment"); return rd(e[e.dataset == ds].enrichment.median(), 2) + "times"


def enrichment_median_range():
    med = _e2("enrichment").groupby("dataset").enrichment.median()
    return f"{rd(med.min(), 1)}-{rd(med.max(), 1)}times"


def enrichment_ge1_runs():
    e = _e2("enrichment"); return f"all{len(e)}" if (e.enrichment >= 1).all() else frac((e.enrichment >= 1).sum(), len(e))


def selection_null_mean():
    d = _csv(CR, "F", "AGG_F2_enrichment_tests.csv"); return rd(d.permU_mean.mean(), 1) + "times"


def above_selection_null(ds):
    d = _csv(CR, "F", "AGG_F2_enrichment_tests.csv"); x = d[d.dataset == ds]
    return frac(x.exceeds_permU_q95.sum(), len(x))


def fisher_selection_null(group):
    p = float(_csv(CR, "F", "AGG_F2_pooled.csv").set_index("group").loc[group, "fisher_permU"])
    if p < 1e-3:
        mant, e = f"{p:.0e}".split("e")
        return f"p={int(mant)}times10^{int(e)}"
    return f"p={rd(p, 2)}"


# ------------------------------------------------------------------ 5.2 natural information
def p3_summary(ds):
    v = _e2("phase3"); v = v[v.dataset == ds].marg_best_r2
    return f"{rd(v.median(), 2)}[{rd(v.min(), 2)},{rd(v.max(), 2)}]"


def p3_all_ci_below_zero():
    v = _e2("phase3"); return f"{int((v.marg_best_r2_hi < 0).sum())}of{len(v)}"


def p3_active_range(ds):
    v = _e2("phase3"); v = v[v.dataset == ds].cond_best_r2
    return f"{rd(v.min(), 2)}to{rd(v.max(), 2)}"


def p3_active_all_negative(ds):
    v = _e2("phase3"); v = v[v.dataset == ds].cond_best_r2
    return frac((v < 0).sum(), len(v))


def rj_spearman_range():
    p = _csv(V2, "phase3_real", "AGG_phase3.csv")
    return f"{rd(p.rj_vs_predictive_rho.min(), 2)}to{sg(p.rj_vs_predictive_rho.max(), 2)}"


def exante_cell(ds, fs):
    x = _cvsel(ds, fs); return f"{sg(x.rho.median(), 2)}({_predictive(x)}/{len(x)})"


def exante_predictive(ds, fs):
    x = _cvsel(ds, fs); return frac(_predictive(x), len(x))


def exante_rho(ds, fs):
    return rd(_cvsel(ds, fs).rho.median(), 2)


def exante_max_predictive(ds):
    p = _pred()
    s = p[(p.dataset == ds) & (p.split == "ID") & p.cv_selected & p.feature_set.isin(["F1", "F1p", "F2", "F3", "F6"])]
    return frac(max(_predictive(g) for _, g in s.groupby("feature_set")), s.seed.nunique())


def mvdf_adds_nothing():
    d = pd.concat([_csv(CR, "E3", "AGG_E3_delta.csv"), _csv(CR, "E1", "AGG_E1_E3_delta.csv")])
    d = d[(d.split == "ID") & (d.block == 96)]
    return f"{int((d.dR2_lo <= 0).sum())}of{len(d)}"


# ------------------------------------------------------------------ 5.3 synthetic control
def synthetic_link(i):
    a = _csv(V2, "phase1_synthetic", "AGG_phase1.csv"); return frac(a[f"link{i}"].sum(), len(a))


def synthetic_r2_range():
    a = _csv(V2, "phase1_synthetic", "AGG_phase1.csv")
    return f"{rd(a.pred_marginal_best_r2.min(), 2)}-{rd(a.pred_cond_best_r2.max(), 2)}"


# ------------------------------------------------------------------ 5.4 oracle and downstream
def oracle_summary(ds):
    v = _e2("phase4"); v = 100 * v[v.dataset == ds].ratio_id
    return f"{rd(v.median(), 1)}%[{rd(v.min(), 1)},{rd(v.max(), 1)}]"


def oracle_all_ci_positive():
    v = _e2("phase4"); return frac(((v.ratio_id_ci_lo > 0) & (v.ratio_shift_ci_lo > 0)).sum(), len(v))


def oracle_shift_median_range():
    med = 100 * _e2("phase4").groupby("dataset").ratio_shift.median()
    return f"{rd(med.min(), 1)}-{rd(med.max(), 1)}%"


def shift_grr_etth_valid():
    g = _csv(CR, "F", "AGG_F5_shift_gap.csv")
    v = g[g.dataset.isin(ETT) & g.coverage_valid & (g.feature_set != "naive_matched")]
    return f"{rd(v.GRR.max(), 2)}to{rd(v.GRR.min(), 2)}"


def shift_electricity_static_coverage():
    g = _csv(CR, "F", "AGG_F5_shift_gap.csv"); e = g[g.dataset == "electricity"]
    return f"{rd(e.coverage.min(), 2)}-{rd(e.coverage.max(), 2)}"


def shift_traffic_f3_grr():
    g = _csv(CR, "F", "AGG_F5_shift_gap.csv")
    t = g[(g.dataset == "traffic") & (g.feature_set == "F3")]
    v = t[t.coverage_valid]
    return f"{rd(v.GRR.min(), 2)}-{rd(v.GRR.max(), 2)}ofthegapon{len(v)}/{len(t)}"


def scorer_combinations():
    s = _e2("scorer_vs_naive"); return str(len(s[s["mode"] != "naive_matched"]))


def scorer_narrower_valid():
    s = _e2("scorer_vs_naive"); s = s[s["mode"] != "naive_matched"]
    return str(int(((s.width_minus_naive < 0) & ((s.coverage - 0.9).abs() <= 0.02)).sum()))


def scorer_cell(ds, mode):
    s = _e2("scorer_vs_naive"); x = s[(s.dataset == ds) & (s["mode"] == mode) & (s.block == "shift")]
    return f"{sg(x.width_minus_naive.median(), 2)}({int((x.wmn_lo > 0).sum())}/{len(x)})"


def naive_shift_coverage_range():
    s = _e2("scorer_vs_naive"); n = s[(s["mode"] == "naive_matched") & (s.block == "shift")]
    return f"{rd(n.coverage.min(), 2)}-{rd(n.coverage.max(), 2)}"


def undercover_range():
    s = _e2("scorer_vs_naive"); x = s[(s.dataset == "electricity") & (s["mode"] == "mvdf") & (s.block == "shift")]
    return f"{rd(x.coverage.min(), 2)}-{rd(x.coverage.max(), 2)}"


def earlier_shift_widths():
    h = _csv(V2, "v2_h6_topup", "AGG_h6.csv").groupby("mode").shift_width.mean()
    return f"{rd(h['correlated'], 2)},{rd(h['random_k'], 2)},{rd(h['pca_k'], 2)}vs.{rd(h['naive_matched'], 2)}"


def naive_no_scorer_gap():
    h = _csv(V2, "v2_h6_topup", "AGG_h6.csv").groupby("mode").shift_width.mean()
    return "within0.02" if abs(h["naive_no_scorer"] - h["naive_matched"]) < 0.02 else "no"


def regime_passes():
    g = _csv(CR, "F", "AGG_F1_regime_gating.csv"); g = g[g.env == "E_eval_ID"]
    return f"{int(g.excludes_1_above.sum())}of{len(g)}"


def regime_rq_range(ds):
    g = _csv(CR, "F", "AGG_F1_regime_gating.csv"); g = g[(g.env == "E_eval_ID") & (g.dataset == ds)]
    return f"{rd(g.R_Q.min(), 2)}-{rd(g.R_Q.max(), 2)}"


def regime_sizes():
    g = _csv(CR, "F", "AGG_F1_regime_gating.csv"); g = g[g.env == "E_eval_ID"]
    return f"{int(g.n1.min())}-{int(g.n1.max())}"


def localization_time_only():
    v = _slcp()["control_A_temporal_adjacency"]["ratio_vs_dense_by_cell"]
    return f"{rd(min(v), 2)}-{rd(max(v), 2)}"


def localization(space, how="passing"):
    j = _slcp()["temporal_exclusion_check"]["per_sae_space"][space]
    v = j["ratio_vs_dense_by_cell"]
    k = j["cells_passing_both"] if how == "passing" else sum(x < 0.8 for x in v)
    return f"{rd(sum(v) / len(v), 2)}({k}/{len(v)})"


def pid_mean_shift_diff():
    b = _csv(V2, "baseline_comparison", "AGG_baselines.csv")
    p = b.pivot_table(index=["dataset", "seed"], columns="mode", values="shift_width")
    return sg((p.pid_conformal - p.naive_matched).mean(), 3)


def pid_direction():
    b = _csv(V2, "baseline_comparison", "AGG_baselines.csv")
    p = b.pivot_table(index=["dataset", "seed"], columns="mode", values="shift_width")
    d = p.pid_conformal - p.naive_matched
    return f"narroweronETTh1{int((d.loc['ETTh1'] < 0).sum())}/3,widerETTh2{int((d.loc['ETTh2'] > 0).sum())}/3"


def cqr_coverage(ds, block):
    b = _csv(V2, "baseline_comparison", "AGG_baselines.csv")
    x = b[(b.dataset == ds) & (b["mode"] == "cqr")][f"{block}_coverage"]
    return f"{rd(x.min(), 2)}-{rd(x.max(), 2)}"


# ------------------------------------------------------------------ 5.5 robustness
def e4_jaccard_range():
    d = _csv(CR, "E4", "AGG_E4_rj_robustness.csv"); d = d[(d.source == "local") & d.dataset.isin(ETT)]
    med = d[d.definition.isin(["binned", "spline", "rank"])].groupby("definition").jaccard_vs_iso.median()
    return f"{rd(med.min(), 2)}-{rd(med.max(), 2)}"


def e4_enrichment_ci(defn):
    d = _csv(CR, "E4", "AGG_E4_rj_robustness.csv")
    d = d[(d.source == "local") & d.dataset.isin(ETT) & (d.definition == defn)]
    return frac(d.enr_ci_excludes_1.sum(), len(d))


def e4_null_all_definitions():
    d = _csv(CR, "E4", "AGG_E4_rj_robustness.csv"); d = d[(d.source == "local") & d.dataset.isin(ETT)]
    n = d.groupby("dataset").seed.nunique().sum()
    return f"all{n}ETThruns" if (d.marg_best_r2 < 0).all() else "notall"


def e5_exceed_q99():
    f = _csv(CR, "E5", "AGG_E5a_diagnostics.csv"); f = f[f.dataset.isin(ETT) & f.in_mvdf]
    return pct(f.frac_exceed_q99_active.median()) + "%"


def e5b_jaccard():
    b = _csv(CR, "E5", "AGG_E5b_natural_range.csv"); return rd(b[b.dataset.isin(ETT)].jaccard_vs_full.median(), 2)


def e5c_sign():
    i = _csv(CR, "E5", "AGG_E5c_interchange.csv"); return pct(i[i.dataset.isin(ETT)].sign_agree.mean()) + "%"


def tm_fvu_range():
    t = _csv(V2, "arch_gen_timemixer", "AGG_arch_gen.csv")
    return f"{rd(t.sae_fvu.min(), 4)}-{rd(t.sae_fvu.max(), 4)}"


def tm_enrichment_range():
    t = _csv(V2, "arch_gen_timemixer", "AGG_arch_gen.csv")
    return f"{rd(t.enrichment_v2.min(), 1)}-{rd(t.enrichment_v2.max(), 1)}times,{int((t.enrichment_v2 >= 1).sum())}/{len(t)}"


def tm_scorer_wider():
    t = _csv(V2, "arch_gen_timemixer", "AGG_arch_gen.csv")
    return frac(((t.nm_minus_dense_shift < 0) & t.nm_minus_dense_sig).sum(), len(t))


def tm_r2_range():
    p = _csv(CR, "E6", "AGG_phase3.csv")
    return f"{rd(p.marg_best_r2.max(), 2)}to{rd(p.marg_best_r2.min(), 2)}"


def tm_oracle_range():
    p = _csv(CR, "E6", "AGG_phase4.csv"); v = pd.concat([p.ratio_id, p.ratio_shift])
    return f"{pct(v.min())}-{pct(v.max())}%"


def tm_exante_predictive():
    p = _csv(CR, "E6", "AGG_E6_predictability.csv")
    s = p[(p.split == "ID") & p.cv_selected & (p.feature_set != "F0")]
    return str(_predictive(s))


def sae_match_range():
    d = _csv(V2, "phase5_sae_repro", "AGG_phase5.csv")
    return f"{pct(d.mvdf_match_rate.min())}-{pct(d.mvdf_match_rate.max())}%ofMVDFfeatures(mean{pct(d.mvdf_match_rate.mean())}%)"


def sae_base_range():
    d = _csv(V2, "phase5_sae_repro", "AGG_phase5.csv")
    return f"{pct(d.pool_base_rate.min())}-{pct(d.pool_base_rate.max())}%"


def necessity_range():
    n = _csv(V2, "phase2_necessity", "AGG_phase2.csv"); neg = n[n.delta_noise < 0].delta_noise
    return f"{rd(neg.max(), 4)}to{rd(neg.min(), 4)}on{len(neg)}of{len(n)}"


def static_scorer_range():
    x = _csv(CR, "G", "AGG_G5_static_shift.csv")
    x = x[x.dataset.isin(ETT) & ~x.method.isin(["naive_matched", "naive_no_scorer"])]
    v = x[["0", "1", "2"]].to_numpy().ravel()
    return f"{rd(v.min(), 2)}to{sg(v.max(), 3)}"


_HELPERS = {"rd", "sg", "pct", "frac"}
REGISTRY = {k: v for k, v in dict(globals()).items()
            if callable(v) and getattr(v, "__module__", None) == __name__
            and not k.startswith("_") and k not in _HELPERS}
