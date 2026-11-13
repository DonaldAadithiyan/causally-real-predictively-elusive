"""Round 2 WP3: nulls and diagnostics for the causal (dose-response) selection.
Outputs runs/p126_round2/wp3/<tag>/ . Generic over real and synthetic runs."""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ.setdefault("P126_ARR", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs", "p126_round2", "arrays"))
torch.set_num_threads(4)

from p126.common import Run, HERE, spearman, block_bootstrap, dump  # noqa: E402
from p126.selection import (frozen_selection, fast_dose, slope, monotone, LAM,  # noqa: E402
                            type_B_enrichment, load_real_models)
from sae.train_sae import encode_all  # noqa: E402
from validation.difficulty_specificity import residualize, select_mvdf_v2  # noqa: E402

R2 = os.path.join(HERE, "runs", "p126_round2")
N_DIR = 50
N_PERM = 1000


def iso_fit(P, U):
    """The isotonic E[U|P] used by residualize (best direction) as a callable."""
    from sklearn.isotonic import IsotonicRegression
    ok = np.isfinite(P) & np.isfinite(U)
    best, bsse = None, np.inf
    for inc in (True, False):
        ir = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(P[ok], U[ok])
        sse = np.sum((U[ok] - ir.predict(P[ok])) ** 2)
        if sse < bsse:
            bsse, best = sse, ir
    return best.predict


def rand_dirs(d, n, rng):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@torch.no_grad()
def S_from_std(model, sae, X, y_t, tvar, prof, h_std):
    h = sae.unstandardize(torch.from_numpy(h_std.astype(np.float32)))
    xb = torch.from_numpy(X[:, :, tvar].astype(np.float32))
    mu, sd = xb.mean(1), xb.std(1) + 1e-6
    yp = (model.head(h) * sd[:, None] + mu[:, None]).numpy()
    e = yp - y_t
    return (np.abs(e) / prof[None, :]).max(1), (e ** 2).mean(1)


def dose_dict(P, U, pool, unc_curves):
    return dict(P_j=P, U_j=U, feature_indices=pool, unc_curves=unc_curves)


def run_wp3(r, model, sae, rng_seed, pool_size, wcap, k, saved_mvdf=None, outdir=None):
    os.makedirs(outdir, exist_ok=True)
    H, L = r.H, r.L
    tv = int(r.target_var)
    sel = r.idx_sel
    Yf = np.zeros(r.X.shape[:1] + (H, r.X.shape[2]), np.float32); Yf[:, :, tv] = r.y
    fs = frozen_selection(model, sae, r.X[sel], Yf[sel], r.hidden[sel], r.S[sel], r.prof_sel,
                          tv, rng_seed, pool_size, wcap, k)
    pool, w = fs["pool"], fs["wsel"]
    out = dict(tag=r.tag, n_pool=len(pool), n_wsel=len(w), mvdf=fs["mvdf"].tolist(),
               mvdf_reproduces_saved=(None if saved_mvdf is None else
                                      sorted(fs["mvdf"].tolist()) == sorted(saved_mvdf)),
               observed_enrichment=fs["enr"].get("enrichment_ratio"),
               n_gate1=int(fs["gate1"].sum()))
    Xw, yw, zw = r.X[sel][w], r.y[sel][w], fs["z_sel"][w]
    tpos = sel[w]                                  # time index of each wsel window
    P, U = fs["dose"]["P_j"], fs["dose"]["U_j"]
    R = fs["R"]
    rng = np.random.default_rng(rng_seed + 2026)

    # ---------------- task 1 + 2 + 6: curves, bootstrap SE, random-direction null, effect size
    f_iso = iso_fit(P, U)
    rows, curves = [], {}
    nullU = np.full((len(pool), N_DIR), np.nan); nullP = nullU.copy()
    null_mono = np.zeros((len(pool), N_DIR), bool)
    bidx = block_bootstrap(len(w), 2 * H, 500, rng_seed)       # blocks over wsel time order
    for i, j in enumerate(pool):
        j = int(j)
        Dp, Du, mse_w, sc_w, na = fast_dose(model, sae, Xw, yw, zw, tv, j, r.prof_sel,
                                            per_window=True)
        a = zw[:, j] != 0
        pj, r2p = slope(LAM, Dp[0]); uj, r2u = slope(LAM, Du[0])
        # block bootstrap SE: resample wsel positions in blocks, keep active ones
        aidx = np.where(a)[0]; posmap = -np.ones(len(w), int); posmap[aidx] = np.arange(len(aidx))
        bsP, bsU = [], []
        if na >= 5:
            for b in bidx[:500]:
                m = posmap[b]; m = m[m >= 0]
                if len(m) < 3:
                    continue
                bsP.append(slope(LAM, mse_w[0][:, m].mean(1))[0])
                bsU.append(slope(LAM, sc_w[0][:, m].mean(1))[0])
        # random-direction null (same activation pattern + magnitudes)
        dirs = rand_dirs(sae.W_dec.shape[0], N_DIR, np.random.default_rng(rng_seed * 7919 + j))
        nDp, nDu, _ = fast_dose(model, sae, Xw, yw, zw, tv, j, r.prof_sel, dirs=dirs)
        for d in range(N_DIR):
            nullP[i, d] = slope(LAM, nDp[d])[0]
            nullU[i, d] = slope(LAM, nDu[d])[0]
            null_mono[i, d] = monotone(nDu[d])
        Rn = nullU[i] - f_iso(nullP[i])
        sdS = float(np.std(sc_w[0][list(LAM).index(1.0)])) if na >= 5 else np.nan
        rows.append(dict(feature=j, n_active=na, P_j=pj, U_j=uj, R_j=float(R[i]),
                         P_r2=r2p, U_r2=r2u, unc_monotone=monotone(Du[0]),
                         unc_spearman_lambda=spearman(LAM, Du[0]) if na >= 5 else np.nan,
                         P_se=float(np.std(bsP)) if bsP else np.nan,
                         U_se=float(np.std(bsU)) if bsU else np.nan,
                         z_U_randdir=float((uj - np.nanmean(nullU[i])) / (np.nanstd(nullU[i]) + 1e-15)),
                         z_R_randdir=float((R[i] - np.nanmean(Rn)) / (np.nanstd(Rn) + 1e-15)),
                         null_U_mean=float(np.nanmean(nullU[i])), null_U_sd=float(np.nanstd(nullU[i])),
                         sd_S_active=sdS, U_effect_size=uj / sdS if sdS else np.nan,
                         in_mvdf=j in set(fs["mvdf"].tolist()),
                         type=next((t for t, ids in fs["ftab"]["cells"].items() if j in ids), "?")))
        curves[j] = dict(D_pred=Dp[0].tolist(), D_unc=Du[0].tolist())
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(outdir, "curves_and_nulls.csv"), index=False)
    dump(dict(lambdas=LAM.tolist(), curves=curves), os.path.join(outdir, "dose_curves.json"))
    np.savez_compressed(os.path.join(outdir, "randdir_null.npz"), pool=pool, nullU=nullU,
                        nullP=nullP, null_mono=null_mono)
    mv = cdf.in_mvdf
    out["z_U_randdir"] = dict(mvdf_median=float(cdf[mv].z_U_randdir.median()),
                              pool_median=float(cdf.z_U_randdir.median()),
                              frac_pool_absz_gt2=float((cdf.z_U_randdir.abs() > 2).mean()),
                              frac_mvdf_absz_gt2=float((cdf[mv].z_U_randdir.abs() > 2).mean()))
    out["z_R_randdir"] = dict(mvdf_median=float(cdf[mv].z_R_randdir.median()),
                              pool_median=float(cdf.z_R_randdir.median()),
                              frac_mvdf_z_gt2=float((cdf[mv].z_R_randdir > 2).mean()))
    out["effect_size"] = dict(mvdf_median_U_over_sdS=float(cdf[mv].U_effect_size.median()),
                              mvdf_max_abs=float(cdf[mv].U_effect_size.abs().max()),
                              pool_median_abs=float(cdf.U_effect_size.abs().median()),
                              mvdf_median_U=float(cdf[mv].U_j.median()),
                              mvdf_median_U_se=float(cdf[mv].U_se.median()),
                              frac_mvdf_U_gt_2se=float((cdf[mv].U_j > 2 * cdf[mv].U_se).mean()))
    out["curve_quality"] = dict(mvdf_median_U_r2=float(cdf[mv].U_r2.median()),
                                pool_frac_unc_monotone=float(cdf.unc_monotone.mean()))

    # ---------------- task 3: selection nulls --------------------------------------------
    mono_real = cdf.unc_monotone.values
    perm_enr = []
    for _ in range(N_PERM):
        p = rng.permutation(len(pool))
        e, _ = type_B_enrichment(P, U[p], None, mono_real[p], pool, k)
        perm_enr.append(e)
    dir_enr = []
    for _ in range(N_PERM):
        d = rng.integers(0, N_DIR, len(pool))
        Ud = nullU[np.arange(len(pool)), d]; md = null_mono[np.arange(len(pool)), d]
        e, _ = type_B_enrichment(P, Ud, None, md, pool, k)
        dir_enr.append(e)
    # sanity: the helper reproduces the observed enrichment from the real values
    e_chk, sel_chk = type_B_enrichment(P, U, None, mono_real, pool, k)
    perm_enr, dir_enr = np.array(perm_enr, float), np.array(dir_enr, float)
    obs = out["observed_enrichment"]
    out["selection_null"] = dict(
        helper_reproduces_observed=bool(abs(e_chk - obs) < 1e-9 and
                                        sorted(sel_chk.tolist()) == sorted(fs["mvdf"].tolist())),
        permU_mean=float(np.nanmean(perm_enr)), permU_q95=float(np.nanquantile(perm_enr, .95)),
        permU_p=float((1 + np.sum(perm_enr >= obs - 1e-12)) / (1 + np.isfinite(perm_enr).sum())),
        randdir_mean=float(np.nanmean(dir_enr)), randdir_q95=float(np.nanquantile(dir_enr, .95)),
        randdir_p=float((1 + np.sum(dir_enr >= obs - 1e-12)) / (1 + np.isfinite(dir_enr).sum())),
        exceeds_selection_alone_permU=bool(obs > np.nanquantile(perm_enr, .95)),
        exceeds_selection_alone_randdir=bool(obs > np.nanquantile(dir_enr, .95)))
    np.savez_compressed(os.path.join(outdir, "selection_null.npz"), perm=perm_enr, randdir=dir_enr)

    # ---------------- task 4 + 5: split-half reliability, out-of-sample effect ------------
    order = np.argsort(tpos)
    nW = len(w)
    half = nW // 2
    gap = L + H
    A_ix = order[:max(0, half - gap // 2)]
    B_ix = order[half + gap // 2:]
    halves = {}
    for nm, ix in (("A", A_ix), ("B", B_ix)):
        Ph = np.full(len(pool), np.nan); Uh = Ph.copy(); unc = {}
        for i, j in enumerate(pool):
            Dp, Du, na = fast_dose(model, sae, Xw[ix], yw[ix], zw[ix], tv, int(j), r.prof_sel)
            Ph[i] = slope(LAM, Dp[0])[0]; Uh[i] = slope(LAM, Du[0])[0]
            unc[int(j)] = dict(zip(LAM.tolist(), Du[0].tolist()))
        Rh = residualize(Ph, Uh, method="isotonic", provenance_p="interventional_dose_response",
                         provenance_u="interventional_dose_response")
        mvh = select_mvdf_v2(dose_dict(Ph, Uh, pool, unc), u_threshold=0.0, k=k)
        halves[nm] = dict(P=Ph, U=Uh, R=Rh, sel=np.asarray(mvh["selected"], int),
                          rate=(zw[ix][:, pool] != 0).mean(0))
    ov = len(set(halves["A"]["sel"]) & set(halves["B"]["sel"]))
    from scipy.stats import hypergeom
    nfin = int(np.isfinite(halves["A"]["R"]).sum())
    out["split_half"] = dict(
        n_A=len(A_ix), n_B=len(B_ix),
        rho_P=spearman(*[np.nan_to_num(halves[h]["P"]) for h in "AB"]),
        rho_U=spearman(*[np.nan_to_num(halves[h]["U"]) for h in "AB"]),
        rho_R=spearman(*[np.nan_to_num(halves[h]["R"]) for h in "AB"]),
        topk_overlap=ov, chance_expected=k * k / max(nfin, 1),
        overlap_p_hypergeom=float(hypergeom.sf(ov - 1, max(nfin, 1), k, k)))
    # out-of-sample: select on A, measure on B and on E_eval_ID, vs firing-rate-matched random
    id_ix = r.idx_id
    z_id = encode_all(sae, r.hidden[id_ix])
    selA = halves["A"]["sel"]
    posA = {int(j): i for i, j in enumerate(pool)}

    def U_on(feats, Xs, ys, zs):
        vals = []
        for j in feats:
            Dp, Du, na = fast_dose(model, sae, Xs, ys, zs, tv, int(j), r.prof_sel)
            vals.append(slope(LAM, Du[0])[0])
        return np.array(vals)
    Ucache = {"B": {}, "eval": {}}

    def U_feat(j, where):
        if j not in Ucache[where]:
            if where == "B":
                Ucache[where][j] = U_on([j], Xw[B_ix], yw[B_ix], zw[B_ix])[0]
            else:
                Ucache[where][j] = U_on([j], r.X[id_ix], r.y[id_ix], z_id)[0]
        return Ucache[where][j]
    obsB = np.nanmean([U_feat(int(j), "B") for j in selA])
    obsE = np.nanmean([U_feat(int(j), "eval") for j in selA])
    rateA = halves["A"]["rate"]
    qbin = np.digitize(rateA, np.quantile(rateA, [.2, .4, .6, .8]))
    nulB, nulE = [], []
    for _ in range(300):
        draw = []
        for j in selA:
            cands = pool[qbin == qbin[posA[int(j)]]]
            draw.append(int(rng.choice(cands)))
        nulB.append(np.nanmean([U_feat(j, "B") for j in draw]))
        nulE.append(np.nanmean([U_feat(j, "eval") for j in draw]))
    nulB, nulE = np.array(nulB), np.array(nulE)
    out["out_of_sample"] = dict(
        selA_meanU_on_A=float(np.nanmean([halves["A"]["U"][posA[int(j)]] for j in selA])),
        selA_meanU_on_B=float(obsB), null_B_mean=float(np.nanmean(nulB)),
        null_B_q95=float(np.nanquantile(nulB, .95)),
        pct_B=float(np.mean(nulB < obsB)),
        selA_meanU_on_evalID=float(obsE), null_eval_mean=float(np.nanmean(nulE)),
        null_eval_q95=float(np.nanquantile(nulE, .95)), pct_eval=float(np.mean(nulE < obsE)))

    # ---------------- task 7: necessity rerun on E_eval_ID --------------------------------
    mvdf = fs["mvdf"] if saved_mvdf is None else np.asarray(saved_mvdf, int)
    Xi, yi = r.X[id_ix], r.y[id_ix]
    zt = torch.from_numpy(z_id.astype(np.float32))
    with torch.no_grad():
        rec = sae.decode(zt).numpy()
        Wd = sae.W_dec.detach().numpy()                       # (d, d_sae)
    S0, _ = S_from_std(model, sae, Xi, yi, tv, r.prof_sel, rec)

    def abl(feats):
        delta = -(z_id[:, feats] @ Wd[:, feats].T)            # (n, d)
        return delta
    dlt = abl(mvdf)
    S1, _ = S_from_std(model, sae, Xi, yi, tv, r.prof_sel, rec + dlt)
    dS = S1 - S0
    act_any = (z_id[:, mvdf] != 0).any(1)
    from p126.common import boot_ci
    lo, hi, nbk = boot_ci(lambda ix: dS[ix].mean(), len(dS), 2 * H, 2000, rng_seed)
    rate_eval = (z_id[:, pool] != 0).mean(0)
    qb = np.digitize(rate_eval, np.quantile(rate_eval, [.2, .4, .6, .8]))
    pos = {int(j): i for i, j in enumerate(pool)}
    null_set, null_dir = [], []
    norms = np.linalg.norm(dlt, axis=1)
    for _ in range(200):
        draw = [int(rng.choice(pool[qb == qb[pos[int(j)]]])) if int(j) in pos else int(rng.choice(pool))
                for j in mvdf]
        Sr, _ = S_from_std(model, sae, Xi, yi, tv, r.prof_sel, rec + abl(np.array(draw)))
        null_set.append(np.mean(Sr - S0))
        u = rng.standard_normal(Wd.shape[0]); u /= np.linalg.norm(u)
        Sd, _ = S_from_std(model, sae, Xi, yi, tv, r.prof_sel, rec + norms[:, None] * u[None, :])
        null_dir.append(np.mean(Sd - S0))
    null_set, null_dir = np.array(null_set), np.array(null_dir)
    out["necessity"] = dict(
        mean_dS=float(dS.mean()), ci=[lo, hi], n_blocks=nbk, mean_dS_active=float(dS[act_any].mean()),
        sd_S=float(S0.std()), dS_over_sdS=float(dS.mean() / S0.std()),
        null_set_mean=float(null_set.mean()), null_set_sd=float(null_set.std()),
        z_vs_set=float((dS.mean() - null_set.mean()) / (null_set.std() + 1e-15)),
        null_dir_mean=float(null_dir.mean()), null_dir_sd=float(null_dir.std()),
        z_vs_dir=float((dS.mean() - null_dir.mean()) / (null_dir.std() + 1e-15)))
    np.savez_compressed(os.path.join(outdir, "necessity.npz"), dS=dS, S0=S0, window_index=id_ix,
                        null_set=null_set, null_dir=null_dir)
    dump(out, os.path.join(outdir, "summary.json"))
    return out


def main_real(runs):
    from p126.common import REAL
    from experiments.phase3_real_predictability import CFG
    from experiments._seed import derive_seed
    res = []
    for ds, s in runs:
        r = Run(ds, s)
        m, sae = load_real_models(os.path.join(R2, "weights"), r.tag, r.X.shape[2], CFG)
        sp = os.path.join(HERE, f"runs/phase3_real/{r.tag}/mvdf_selection.json")
        saved_mvdf = json.load(open(sp))["MVDF_v2"] if os.path.exists(sp) else list(map(int, r.mvdf))
        print("WP3", r.tag, flush=True)
        res.append(run_wp3(r, m, sae, derive_seed(s, ds), 120, 2048, 12, saved_mvdf,
                           os.path.join(R2, "wp3", r.tag)))
    return res


if __name__ == "__main__":
    from p126.common import REAL
    runs = REAL if len(sys.argv) < 2 else [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    main_real(runs)
    print("WP3 done")
