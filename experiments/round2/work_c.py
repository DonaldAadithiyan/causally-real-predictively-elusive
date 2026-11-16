"""Work item C — explain the coverage gap (brief §8). Real data only. Outputs results/round2/p126/C/."""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore")

from experiments.round2.common import (Run, REAL, OUT, ALPHA, HERE, boot_ci, n_eff, conf_q, profile_of,  # noqa
                         score_with, dump)
from experiments._seed import derive_seed  # noqa: E402
from mechconf.calibration.aci import empirical_quantile, GlobalACI  # noqa: E402
from mechconf.calibration.difficulty_scorer import (naive_no_scorer_run, DifficultyScorer,  # noqa: E402
                                           build_features)
from mechconf.baselines.cqr import cqr_calibrate, cqr_run, cqr_conformity  # noqa: E402
from mechconf.baselines.pid_conformal import pid_conformal_run  # noqa: E402

COUT = os.path.join(OUT, "C")
os.makedirs(COUT, exist_ok=True)
GG = 0.03


def original_blocks(r):
    """baseline_comparison.py / phase3_scorer_h6.py: E_eval_ID -> 50/30/20, no embargo, and
    collect() called per block, so every block gets its OWN step profile."""
    id_ix = r.idx_id
    n = len(id_ix); c1, c2 = int(0.5 * n), int(0.8 * n)
    B = {}
    for name, ix in (("tr", id_ix[:c1]), ("conf", id_ix[c1:c2]), ("idw", id_ix[c2:]),
                     ("sh", r.idx_sh)):
        s, p = r.S_local(ix)
        B[name] = dict(ix=ix, S=s, prof=p, yhat=r.yhat[ix].astype(float), y=r.y[ix].astype(float),
                       lo=(r.q_lo[ix].astype(float) if hasattr(r, "q_lo") else None),
                       hi=(r.q_hi[ix].astype(float) if hasattr(r, "q_hi") else None))
    return B


def aci_run_delayed(score_cal, S_te, alpha, gamma, delay):
    """Same GlobalACI update as naive_no_scorer_run, but window t's miss indicator is only
    fed to the update once it is observable (delay = H windows later)."""
    q = empirical_quantile(score_cal, 1 - alpha)
    aci = GlobalACI(base_q=max(q, 1e-6), alpha_target=alpha, gamma=gamma, clip=20 * max(q, 1e-6))
    cov = np.empty(len(S_te), bool); hw = np.empty(len(S_te))
    for t in range(len(S_te)):
        if t - delay >= 0:
            aci.update(bool(cov[t - delay]))
        hw[t] = aci.half_width()
        cov[t] = S_te[t] <= hw[t]
    return cov, hw


def aci_trace(score_cal, S_te, alpha, gamma):
    q = empirical_quantile(score_cal, 1 - alpha)
    aci = GlobalACI(base_q=max(q, 1e-6), alpha_target=alpha, gamma=gamma, clip=20 * max(q, 1e-6))
    hw, cov = [], []
    for s in S_te:
        h = aci.half_width(); hw.append(h); c = s <= h; cov.append(c); aci.update(bool(c))
    al = np.array([h_["alpha"] for h_ in aci.history]); a = np.array([h_["a"] for h_ in aci.history])
    return np.array(hw), np.array(cov), al, a, q


# ------------------------------------------------------------------ per run
def run(ds, seed):
    r = Run(ds, seed)
    H, L = r.H, r.L
    sseed = derive_seed(seed, ds)
    B = original_blocks(r)
    cal80 = np.concatenate([B["tr"]["S"], B["conf"]["S"]])
    out = dict(dataset=ds, seed=seed, n_tr=len(B["tr"]["ix"]), n_conf=len(B["conf"]["ix"]),
               n_idw=len(B["idw"]["ix"]), n_sh=len(B["sh"]["ix"]))
    rows = []

    # ---- (task 1/4) reproduce the original numbers exactly, then decompose -------------
    for seg in ("idw", "sh"):
        b = B[seg]
        r0 = naive_no_scorer_run(cal80, b["yhat"], b["y"], b["prof"], ALPHA, gamma=GG)
        q = empirical_quantile(cal80, 1 - ALPHA)
        cov_static_local = float(np.mean(b["S"] <= q))
        # interval-based coverage of the same static band must equal mean(S<=q)
        hwv = q * b["prof"]
        cov_static_intervals = float(np.mean(np.all(np.abs(b["y"] - b["yhat"]) <= hwv[None], 1)))
        # consistent profile: E_select profile everywhere (cal and test)
        S_sel_cal = np.concatenate([r.S[B["tr"]["ix"]], r.S[B["conf"]["ix"]]])
        q_sel = empirical_quantile(S_sel_cal, 1 - ALPHA)
        cov_static_sel = float(np.mean(r.S[b["ix"]] <= q_sel))
        # consistent profile: calibration (tr u conf) profile applied to cal and test
        p_cal = profile_of(r.abs_err[np.concatenate([B["tr"]["ix"], B["conf"]["ix"]])])
        S_cc = score_with(r.abs_err[np.concatenate([B["tr"]["ix"], B["conf"]["ix"]])], p_cal)
        q_cc = empirical_quantile(S_cc, 1 - ALPHA)
        S_te_cc = score_with(r.abs_err[b["ix"]], p_cal)
        cov_static_calprof = float(np.mean(S_te_cc <= q_cc))
        cov_delay_local, _ = aci_run_delayed(cal80, b["S"], ALPHA, GG, H)
        cov_delay_cal, _ = aci_run_delayed(S_cc, S_te_cc, ALPHA, GG, H)
        cov_imm_cal, _ = aci_run_delayed(S_cc, S_te_cc, ALPHA, GG, 1)
        c0 = r0["covered"].astype(float)
        lo, hi, nbk = boot_ci(lambda ix: c0[ix].mean(), len(c0), 2 * H, 2000, sseed)
        bci = {str(bl): boot_ci(lambda ix: c0[ix].mean(), len(c0), bl, 2000, sseed)[:2]
               for bl in (H, 4 * H)}
        rows.append(dict(dataset=ds, seed=seed, segment=seg, n=len(c0),
                         orig_ACI_cov=float(c0.mean()), orig_ACI_width=float(r0["width"].mean()),
                         orig_ACI_cov_ci=[lo, hi], ci_other_blocks=bci, n_blocks=nbk,
                         n_eff=n_eff(c0),
                         static_localprof_cov=cov_static_local,
                         static_localprof_cov_from_intervals=cov_static_intervals,
                         static_selprof_cov=cov_static_sel, static_calprof_cov=cov_static_calprof,
                         ACI_delayedH_localprof_cov=float(cov_delay_local.mean()),
                         ACI_delayedH_calprof_cov=float(cov_delay_cal.mean()),
                         ACI_immediate_calprof_cov=float(cov_imm_cal.mean()),
                         q_cal80=q, q_calprof=q_cc))
        # ACI trajectory (task 7)
        hw, cov, al, a, qb = aci_trace(cal80, b["S"], ALPHA, GG)
        np.savez_compressed(os.path.join(COUT, f"{r.tag}_{seg}_aci_trace.npz"), hw=hw, covered=cov,
                            alpha_t=al, a_t=a, base_q=qb, S=b["S"], window_index=b["ix"])
        rows[-1].update(aci_hw_min=float(hw.min()), aci_hw_max=float(hw.max()),
                        aci_clip=20 * qb, aci_hits_clip=bool(np.any(hw >= 20 * qb - 1e-12)),
                        aci_hits_zero=bool(np.any(hw <= 0)),
                        aci_alpha_t_final=float(al[-1]), aci_a_final=float(a[-1]))
    # PID + CQR exact reproduction
    q0 = empirical_quantile(cal80, 1 - ALPHA)
    for seg in ("idw", "sh"):
        b = B[seg]
        rp = pid_conformal_run(q0, b["yhat"], b["y"], b["prof"], ALPHA, eta=GG)
        rows_seg = [x for x in rows if x["segment"] == seg][0]
        rows_seg["pid_cov"] = float(rp["covered"].mean())
    if B["conf"]["lo"] is not None:
        qh = cqr_calibrate(B["conf"]["lo"], B["conf"]["hi"], B["conf"]["y"], ALPHA,
                           step_profile=B["conf"]["prof"])
        out["cqr_q_hat"] = qh
        for seg in ("conf", "idw", "sh"):
            b = B[seg]
            rc = cqr_run(b["lo"], b["hi"], b["y"], qh, step_profile=b["prof"])
            E = cqr_conformity(b["lo"], b["hi"], b["y"], b["prof"])
            # consistent: E_select profile everywhere
            E_sel = cqr_conformity(b["lo"], b["hi"], b["y"], r.prof_sel)
            raw_step_cov = ((b["y"] >= b["lo"]) & (b["y"] <= b["hi"])).mean(0)
            out[f"cqr_{seg}"] = dict(
                cov_original=float(rc["covered"].mean()),
                cov_check_E_le_qhat=float(np.mean(E <= qh)),
                crossing_rate_window_any=float(np.mean((b["lo"] > b["hi"]).any(1))),
                crossing_rate_cells=float(np.mean(b["lo"] > b["hi"])),
                raw_joint_cov=float(np.mean(np.all((b["y"] >= b["lo"]) & (b["y"] <= b["hi"]), 1))),
                raw_step_cov_h1=float(raw_step_cov[0]), raw_step_cov_mid=float(raw_step_cov[H // 2]),
                raw_step_cov_last=float(raw_step_cov[-1]),
                E_median=float(np.median(E)), E_q90=float(np.quantile(E, .9)),
                E_sel_q90=float(np.quantile(E_sel, .9)),
                mean_raw_width=float((b["hi"] - b["lo"]).mean()))
        qh_sel = cqr_calibrate(B["conf"]["lo"], B["conf"]["hi"], B["conf"]["y"], ALPHA,
                               step_profile=r.prof_sel)
        for seg in ("idw", "sh"):
            b = B[seg]
            out[f"cqr_{seg}"]["cov_selprof"] = float(np.mean(
                cqr_conformity(b["lo"], b["hi"], b["y"], r.prof_sel) <= qh_sel))
        # CQR calibrated with BOTH conf and the tr block (80%), consistent profile
        tr_conf = np.concatenate([B["tr"]["ix"], B["conf"]["ix"]])
        E80 = cqr_conformity(r.q_lo[tr_conf], r.q_hi[tr_conf], r.y[tr_conf], r.prof_sel)
        q80 = empirical_quantile(E80, 1 - ALPHA)
        for seg in ("idw", "sh"):
            b = B[seg]
            out[f"cqr_{seg}"]["cov_selprof_cal80"] = float(np.mean(
                cqr_conformity(b["lo"], b["hi"], b["y"], r.prof_sel) <= q80))

    # ---- (task 6) drift: conformalize vs width-eval, random contiguous blocks ---------
    S = r.S                                      # consistent E_select profile
    sc, si = S[B["conf"]["ix"]], S[B["idw"]["ix"]]
    from scipy import stats
    ks = stats.ks_2samp(sc, si).statistic
    blk = 2 * H
    allx = np.concatenate([sc, si]); lab = np.r_[np.zeros(len(sc)), np.ones(len(si))]
    nbk = int(np.ceil(len(allx) / blk)); bid = np.arange(len(allx)) // blk
    nb_te = int(np.ceil(len(si) / blk))
    rng = np.random.default_rng(sseed)
    null = []
    for _ in range(2000):
        te_b = rng.choice(nbk, nb_te, replace=False)
        m = np.isin(bid, te_b)
        if m.all() or (~m).all():
            continue
        null.append(stats.ks_2samp(allx[~m], allx[m]).statistic)
    null = np.array(null)
    out["drift_conf_vs_idw"] = dict(
        conf_q50=float(np.median(sc)), idw_q50=float(np.median(si)),
        conf_q90=float(np.quantile(sc, .9)), idw_q90=float(np.quantile(si, .9)),
        ks=float(ks), ks_block_null_p=float((1 + np.sum(null >= ks)) / (1 + len(null))),
        n_blocks_total=nbk, n_distinct_partitions=int(len(set(map(tuple, [[0]])))) if False else None)
    # random contiguous-block assignment of the ID block (static constant band, E_select prof)
    id_ix = r.idx_id; Sid = S[id_ix]; n = len(Sid)
    bid = np.arange(n) // blk; nbk = bid.max() + 1
    covs = []
    for _ in range(1000):
        te_b = rng.choice(nbk, max(1, int(round(0.2 * nbk))), replace=False)
        te = np.isin(bid, te_b)
        # embargo: drop calibration windows within L+H of any test window
        te_pos = np.where(te)[0]
        dist = np.min(np.abs(np.arange(n)[:, None] - te_pos[None, :]), axis=1)
        cal = (~te) & (dist > L + H)
        if cal.sum() < 50:
            continue
        q = empirical_quantile(Sid[cal], 1 - ALPHA)
        covs.append(np.mean(Sid[te] <= q))
    covs = np.array(covs)
    q_chrono = empirical_quantile(Sid[:int(0.8 * n)], 1 - ALPHA)
    out["random_block_assignment"] = dict(mean_cov=float(covs.mean()), sd_cov=float(covs.std()),
                                          q05=float(np.quantile(covs, .05)),
                                          q95=float(np.quantile(covs, .95)), reps=len(covs),
                                          chrono_cov=float(np.mean(Sid[int(0.8 * n):] <= q_chrono)))
    # rolling coverage over ID + shift for the static band from cal80 (E_select profile)
    qs = empirical_quantile(S[np.concatenate([B["tr"]["ix"], B["conf"]["ix"]])], 1 - ALPHA)
    ix_all = np.concatenate([id_ix, r.idx_sh])
    cv = (S[ix_all] <= qs).astype(float)
    np.savez_compressed(os.path.join(COUT, f"{r.tag}_rolling.npz"), window_index=ix_all,
                        S=S[ix_all], covered_static=cv, q=qs, origin_ts=r.origin_ts[ix_all],
                        seg=np.r_[np.zeros(len(id_ix)), np.ones(len(r.idx_sh))])

    # ---- (task 7) learned-scorer path: clip ceiling diagnostics (local proxy) ----------
    out["scorer"] = scorer_diag(r, B, sseed)
    return out, rows


def scorer_diag(r, B, sseed):
    """phase3_scorer_h6 path with the frozen phase-3 SAE / MVDF (local proxy: the scorer run
    itself used the Kaggle data and a 20k-step SAE). Logs q_resid, the ACI clip ceiling
    20*max(q_resid,1e-6), and how often the half-width is pinned at that ceiling."""
    import torch
    from sklearn.decomposition import PCA
    from experiments.round2.work_a import feature_sets
    fs = feature_sets(r, B["tr"]["ix"], sseed)
    res = {}
    for mode in ("mvdf", "correlated", "random_k", "dense"):
        F = fs[mode]
        sc = DifficultyScorer(mode, hidden_dim=F.shape[1]).fit(
            F[B["tr"]["ix"]].astype(np.float32), B["tr"]["S"], alpha=ALPHA, seed=sseed)
        sc.conformalize(F[B["conf"]["ix"]].astype(np.float32), B["conf"]["S"])
        qr = float(sc._q_resid)
        base_q = max(qr, 1e-6)
        d = dict(q_resid=qr, clip_ceiling=20 * base_q)
        for seg in ("idw", "sh"):
            b = B[seg]
            rr = sc.run(F[b["ix"]].astype(np.float32), b["yhat"], b["y"], b["prof"], ALPHA, gamma=GG)
            with torch.no_grad():
                pq = sc.model(torch.from_numpy(F[b["ix"]].astype(np.float32))).numpy()
            hw = rr["width"] / 2 / np.mean(b["prof"])
            pinned = np.isclose(hw, 20 * base_q, rtol=1e-4)
            d[f"{seg}_cov"] = float(rr["covered"].mean())
            d[f"{seg}_frac_pinned_at_clip"] = float(pinned.mean())
            d[f"{seg}_pred_q_median"] = float(np.median(pq))
            d[f"{seg}_pred_q_max"] = float(np.max(pq))
            # same scorer with the clip removed (ceiling far above any half-width)
            from mechconf.calibration.ffdci_head import FFDCIHeadACI
            runner = FFDCIHeadACI(sc.model, base_q=qr, alpha_target=ALPHA, gamma=GG,
                                  clip_mult=1e9, step_profile=b["prof"], center_pred=False)
            runner.aci.clip = None
            rr2 = runner.run(F[b["ix"]].astype(np.float32), b["yhat"], b["y"])
            d[f"{seg}_cov_noclip_signed_qresid"] = float(rr2["covered"].mean())
        res[mode] = d
    return res


# ------------------------------------------------------------------ exchangeable test (task 2)
def exchangeable_test(reps=300, scorer_reps=100, seed=0):
    """Feed the ORIGINAL code paths with i.i.d. synthetic errors (per-step scale from a real
    profile). Coverage must average 0.90 within MC error for a correct implementation."""
    rng = np.random.default_rng(seed)
    r = Run("ETTh1", 0)
    prof_true = r.prof_sel
    N, H = len(r.idx_id), r.H
    n_sh = len(r.idx_sh)
    res = {k: [] for k in ("naive_ACI", "naive_static", "naive_ACI_delayed", "pid", "cqr",
                           "scorer_mvdf_like")}
    for rep in range(reps):
        tot = N + n_sh
        e = prof_true[None, :] * rng.standard_normal((tot, H))
        yhat = np.zeros_like(e); y = e
        lo = yhat - 1.6 * prof_true[None, :] + 0.1 * rng.standard_normal((tot, H))
        hi = yhat + 1.6 * prof_true[None, :] + 0.1 * rng.standard_normal((tot, H))
        c1, c2 = int(0.5 * N), int(0.8 * N)
        blocks = dict(tr=slice(0, c1), conf=slice(c1, c2), idw=slice(c2, N))
        def sc(sl):
            ae = np.abs(e[sl]); p = profile_of(ae); return score_with(ae, p), p
        S_tr, _ = sc(blocks["tr"]); S_cf, p_cf = sc(blocks["conf"]); S_iw, p_iw = sc(blocks["idw"])
        cal80 = np.r_[S_tr, S_cf]
        r0 = naive_no_scorer_run(cal80, yhat[blocks["idw"]], y[blocks["idw"]], p_iw, ALPHA, GG)
        res["naive_ACI"].append(r0["covered"].mean())
        res["naive_static"].append(np.mean(S_iw <= empirical_quantile(cal80, 1 - ALPHA)))
        cd, _ = aci_run_delayed(cal80, S_iw, ALPHA, GG, H)
        res["naive_ACI_delayed"].append(cd.mean())
        rp = pid_conformal_run(empirical_quantile(cal80, 1 - ALPHA), yhat[blocks["idw"]],
                               y[blocks["idw"]], p_iw, ALPHA, eta=GG)
        res["pid"].append(rp["covered"].mean())
        qh = cqr_calibrate(lo[blocks["conf"]], hi[blocks["conf"]], y[blocks["conf"]], ALPHA, p_cf)
        rc = cqr_run(lo[blocks["idw"]], hi[blocks["idw"]], y[blocks["idw"]], qh, p_iw)
        res["cqr"].append(rc["covered"].mean())
        if rep < scorer_reps:
            F = np.where(rng.random((N, 12)) < 0.03, rng.exponential(1, (N, 12)), 0.0).astype(np.float32)
            s_ = DifficultyScorer("mvdf", 12).fit(F[blocks["tr"]], S_tr, ALPHA, seed=rep, epochs=200)
            s_.conformalize(F[blocks["conf"]], S_cf)
            rr = s_.run(F[blocks["idw"]], yhat[blocks["idw"]], y[blocks["idw"]], p_iw, ALPHA, GG)
            res["scorer_mvdf_like"].append(rr["covered"].mean())
    out = {}
    for k, v in res.items():
        v = np.array(v)
        out[k] = dict(mean=float(v.mean()), mc_se=float(v.std(ddof=1) / np.sqrt(len(v))),
                      sd=float(v.std()), reps=len(v),
                      within_3se_of_target=bool(abs(v.mean() - 0.9) <= 3 * v.std(ddof=1) / np.sqrt(len(v))))
    return out


def quantile_arith():
    rng = np.random.default_rng(1)
    ok = True; worst = 0
    for n in (5, 9, 10, 50, 393, 1572):
        for _ in range(200):
            s = rng.random(n)
            a = empirical_quantile(s, 0.9)
            k = int(np.ceil((n + 1) * 0.9))
            b = np.inf if k > n else np.sort(s)[k - 1]
            c = np.quantile(s, (min(k, n) - 1) / (n - 1), method="higher")  # k-th order stat
            if not (a == b or (np.isinf(b) and a == s.max())):
                ok = False
            worst = max(worst, abs(a - c))
    return dict(rank_rule_matches_ceil_np1_level=ok, max_abs_diff_vs_kth_order_stat_numpy=worst,
                note="empirical_quantile clips rank to n: when ceil((n+1)(1-a)) > n (n < 9 at "
                     "a=0.1) it returns max(score) instead of +inf. Not reachable at the block "
                     "sizes used (>= 350).")


def reconcile():
    """naive_matched ID coverage in the scorer run vs the baseline-comparison run."""
    a = pd.read_csv(os.path.join(HERE, "results/v2/v2_h6_topup/AGG_h6.csv"))
    b = pd.read_csv(os.path.join(HERE, "results/v2/baseline_comparison/AGG_baselines.csv"))
    a = a[a["mode"] == "naive_matched"][["dataset", "seed", "id_coverage", "shift_coverage"]]
    b = b[b["mode"] == "naive_matched"][["dataset", "seed", "id_coverage", "shift_coverage"]]
    m = a.merge(b, on=["dataset", "seed"], how="outer", suffixes=("_scorer_run", "_baseline_run"))
    m["id_diff"] = m.id_coverage_scorer_run - m.id_coverage_baseline_run
    return m


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "results", "v2"):
        outs, rows = [], []
        for ds, s in REAL:
            print("C:", ds, s, flush=True)
            o, rr = run(ds, s)
            outs.append(o); rows += rr
        dump(outs, os.path.join(COUT, "per_run.json"))
        pd.DataFrame(rows).to_csv(os.path.join(COUT, "coverage_decomposition.csv"), index=False)
    if which in ("all", "static"):
        m = reconcile(); m.to_csv(os.path.join(COUT, "reconcile.csv"), index=False)
        print(m.to_string())
        dump(quantile_arith(), os.path.join(COUT, "quantile_arith.json"))
    if which in ("all", "exch"):
        ex = exchangeable_test()
        dump(ex, os.path.join(COUT, "exchangeable_test.json")); print(ex)
    print("C done")
