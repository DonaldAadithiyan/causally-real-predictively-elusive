"""Round 2 WP4: where the hidden-state signal lives, and whether it controls S_t.
Outputs runs/p126_round2/wp4/. ETTh2 first, then ETTh1."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
HERE0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("P126_ARR", os.path.join(HERE0, "runs", "p126_round2", "arrays"))
torch.set_num_threads(4)

from p126.common import Run, HERE, spearman, auroc, circ_null, dump  # noqa: E402
from p126.selection import load_real_models  # noqa: E402
from p126.work_b import causal_score_feats, input_feats  # noqa: E402
from sae.train_sae import encode_all  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402

R2 = os.path.join(HERE, "runs", "p126_round2")
OUT = os.path.join(R2, "wp4")
os.makedirs(OUT, exist_ok=True)
LAMS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_NULL = 50
K = 12


# ------------------------------------------------------------------ blocked CV utilities
def folds(n, K=5, emb=192):
    edges = np.linspace(0, n, K + 1).astype(int)
    out = []
    for k in range(K):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.where((np.arange(n) < edges[k] - emb) | (np.arange(n) >= edges[k + 1] + emb))[0]
        out.append((tr, te))
    return out


def ridge_fit_predict(Ftr, ytr, Fte):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(sc.transform(Ftr), ytr)
    return m.predict(sc.transform(Fte)), m, sc


def lasso_select(Ftr, ytr, k):
    """k features with the largest |coef| along a Lasso path fitted on training data only."""
    from sklearn.linear_model import lasso_path
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(Ftr)
    alphas, coefs, _ = lasso_path(X, ytr - ytr.mean(), n_alphas=60)
    for a in range(coefs.shape[1]):
        nz = np.where(coefs[:, a] != 0)[0]
        if len(nz) >= k:
            c = np.abs(coefs[:, a])
            return np.argsort(-c)[:k]
    return np.argsort(-np.abs(coefs[:, -1]))[:k]


def partial_spearman(x, y, C):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    RC = np.column_stack([rankdata(c) for c in C.T] + [np.ones(len(x))])
    ex = rx - RC @ np.linalg.lstsq(RC, rx, rcond=None)[0]
    ey = ry - RC @ np.linalg.lstsq(RC, ry, rcond=None)[0]
    return spearman(ex, ey)


def cv_eval(F, y, controls, emb, seed, predict="ridge", select_k=None):
    """Blocked 5-fold CV inside the ID block. Returns within-fold Spearman (mean, per fold),
    within-fold partial Spearman given controls, AUROC, circular-shift p of the within-fold
    mean Spearman."""
    n = len(y)
    fl = folds(n, 5, emb)
    oof = np.full(n, np.nan)
    for tr, te in fl:
        Ftr, Fte = F[tr], F[te]
        if select_k is not None:
            cols = lasso_select(Ftr, y[tr], select_k)
            Ftr, Fte = Ftr[:, cols], Fte[:, cols]
        oof[te] = ridge_fit_predict(Ftr, y[tr], Fte)[0]

    def within(p):
        return float(np.mean([spearman(p[te], y[te]) for _, te in fl]))
    per = [spearman(oof[te], y[te]) for _, te in fl]
    part = [partial_spearman(oof[te], y[te], controls[te]) for _, te in fl]
    au = [auroc(oof[te], y[te] >= np.quantile(y[te], .9)) for _, te in fl]
    _, p = circ_null(lambda xx, _y: within(xx), oof, y, emb, 500, seed)
    return dict(spearman_within=within(oof), spearman_folds=per,
                spearman_fold_min=float(np.min(per)), spearman_fold_max=float(np.max(per)),
                partial_spearman_within=float(np.mean(part)), partial_folds=part,
                auroc_within=float(np.nanmean(au)), circ_p=p)


# ------------------------------------------------------------------ interventions
@torch.no_grad()
def S_of_hidden(model, X, y_t, tvar, prof, h_raw):
    """S and mse from a (raw, pre_head) target-variate token."""
    h = torch.from_numpy(h_raw.astype(np.float32))
    xb = torch.from_numpy(X[:, :, tvar].astype(np.float32))
    mu, sd = xb.mean(1), xb.std(1) + 1e-6
    yp = (model.head(h) * sd[:, None] + mu[:, None]).numpy()
    e = yp - y_t
    return (np.abs(e) / prof[None, :]).max(1), (e ** 2).mean(1)


def steer_effect(model, X, y_t, tvar, prof, h, u, step):
    """Mean S and loss at h + lam*step*u for lam in LAMS; slope over lam; projection-out."""
    mS, mL = [], []
    for lam in LAMS:
        s, l = S_of_hidden(model, X, y_t, tvar, prof, h + lam * step * u[None, :])
        mS.append(s.mean()); mL.append(l.mean())
    mS, mL = np.array(mS), np.array(mL)
    slS = np.polyfit(LAMS, mS, 1)[0]; slL = np.polyfit(LAMS, mL, 1)[0]
    s0, l0 = S_of_hidden(model, X, y_t, tvar, prof, h)
    hp = h - np.outer(h @ u, u)
    s1, l1 = S_of_hidden(model, X, y_t, tvar, prof, hp)
    return dict(slope_S=float(slS), slope_loss=float(slL), dS_lam2=float(mS[-1] - mS[2]),
                dS_lam_m2=float(mS[0] - mS[2]), dS_projout=float((s1 - s0).mean()),
                dloss_projout=float((l1 - l0).mean()), mean_S=mS.tolist())


def z_of(x, null):
    null = np.asarray(null)
    return float((x - null.mean()) / (null.std() + 1e-15))


def run(ds, seed):
    from experiments.phase3_real_predictability import CFG
    r = Run(ds, seed)
    H, L = r.H, r.L
    emb = L + H
    tv = int(r.target_var)
    sseed = derive_seed(seed, ds)
    model, sae = load_real_models(os.path.join(R2, "weights"), r.tag, r.X.shape[2], CFG)
    id_ix = r.idx_id
    S = r.S[id_ix]
    hid = r.hidden[id_ix].astype(np.float64)
    with torch.no_grad():
        zt = torch.from_numpy(encode_all(sae, r.hidden[id_ix]).astype(np.float32))
        Z = zt.numpy().astype(np.float64)
        rec = sae.unstandardize(sae.decode(zt)).numpy().astype(np.float64)
    resid = hid - rec
    # simple forecast-time controls
    G = causal_score_feats(r.S, H)
    inp = input_feats(r)
    ctrl_all = np.column_stack([G["last_obs"], G["ema"], G["roll_q90"], inp["in_std"],
                                inp["in_diff_std"], inp["in_allch_std"]])
    ctrl = ctrl_all[id_ix]
    ctrl = np.where(np.isnan(ctrl), np.nanmean(ctrl, 0), ctrl)

    # ---------------- 1 + 2: confirm + locate ------------------------------------------
    reps = {"raw_hidden": hid, "sae_reconstruction": rec, "sae_residual": resid,
            "sae_all_2048": Z, "mvdf": Z[:, r.mvdf], "simple_controls": ctrl}
    loc = []
    for nm, F in reps.items():
        d = cv_eval(F, S, ctrl, emb, sseed)
        loc.append(dict(dataset=ds, seed=seed, representation=nm, **d))
    d = cv_eval(Z, S, ctrl, emb, sseed, select_k=K)
    loc.append(dict(dataset=ds, seed=seed, representation="sae_lasso_k12", **d))
    # PCA-12 for reference (fit within training folds is approximated by global fit: report only)

    # ---------------- 3: probe on fitting data only (ID first 50%) -----------------------
    n = len(id_ix); c1 = int(0.5 * n)
    fit, ev = np.arange(c1), np.arange(c1 + emb, n)
    p_ev, ridge, scl = ridge_fit_predict(hid[fit], S[fit], hid[ev])
    w_std = ridge.coef_ / scl.scale_                    # direction in raw hidden space
    w = w_std / np.linalg.norm(w_std)
    probe_pred = dict(spearman=spearman(p_ev, S[ev]),
                      auroc=auroc(p_ev, S[ev] >= np.quantile(S[ev], .9)))
    _, probe_pred["circ_p"] = circ_null(lambda xx, _y: spearman(xx, S[ev]), p_ev, S[ev], emb, 500, sseed)
    # sparse probe-selected SAE features (on fitting data)
    psel = lasso_select(Z[fit], S[fit], K)

    # ---------------- 4 + 5: interventions on ID evaluation windows ----------------------
    Xe, ye, he = r.X[id_ix][ev], r.y[id_ix][ev], hid[ev]
    proj_sd = lambda u: float(np.std(hid[fit] @ u))
    step_w = proj_sd(w)
    eff_probe = steer_effect(model, Xe, ye, tv, r.prof_sel, he, w, step_w)
    rng = np.random.default_rng(sseed + 11)
    ev_, V_ = np.linalg.eigh(np.cov(hid[fit].T))
    ev_ = np.clip(ev_, 1e-12, None)
    pools = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):            # isotropic -> covariance-shaped
        Gc = rng.standard_normal((hid.shape[1], 3000))
        pools.append((V_ @ (ev_[:, None] ** (beta / 2) * Gc)).T)
    Uc = np.vstack(pools); Uc /= np.linalg.norm(Uc, axis=1, keepdims=True)
    sd_c = np.std(hid[fit] @ Uc.T, axis=0)
    Ui = rng.standard_normal((N_NULL, hid.shape[1])); Ui /= np.linalg.norm(Ui, axis=1, keepdims=True)
    keys = ["slope_S", "dS_projout", "slope_loss"]

    def zrow(u, eff):
        """null 1 (norm): isotropic random unit directions, SAME absolute step as the target.
        null 2 (var): covariance-shaped random directions whose projected sd is within 25% of
        the target's, each stepped by its own projected sd."""
        st = proj_sd(u)
        n1 = [steer_effect(model, Xe, ye, tv, r.prof_sel, he, v, st) for v in Ui]
        ok = np.argsort(np.abs(np.log(sd_c / st)))[:N_NULL]   # 50 nearest in projected sd
        n2 = [steer_effect(model, Xe, ye, tv, r.prof_sel, he, Uc[i], sd_c[i]) for i in ok]
        out = {f"z_{k}_normnull": z_of(eff[k], [e[k] for e in n1]) for k in keys}
        out.update({f"z_{k}_varnull": (z_of(eff[k], [e[k] for e in n2]) if len(n2) >= 10 else np.nan)
                    for k in keys})
        out["n_varnull"] = int(len(n2))
        out["varnull_max_sd_ratio_dev"] = float(np.max(np.abs(sd_c[ok] / st - 1)))
        return out
    # positive controls: top PC; decoder direction of MVDF feature with largest U_j
    from sklearn.decomposition import PCA
    pc1 = PCA(1).fit(hid[fit]).components_[0]
    eff_pc1 = steer_effect(model, Xe, ye, tv, r.prof_sel, he, pc1, proj_sd(pc1))
    import json
    cp = os.path.join(HERE, f"runs/phase3_real/{r.tag}/candidates.csv")
    if not os.path.exists(cp):
        cp = os.path.join(R2, "wp6", f"{r.tag}_candidates.csv")
    cand = pd.read_csv(cp)
    cm = cand[cand.in_mvdf]
    jmax = int(cm.loc[cm.U_j.idxmax(), "feature"])

    def dec_dir(j):
        v = sae.W_dec[:, j].detach().numpy().astype(np.float64) * sae._input_std.numpy()
        return v / np.linalg.norm(v)
    u_max = dec_dir(jmax)
    eff_umax = steer_effect(model, Xe, ye, tv, r.prof_sel, he, u_max, proj_sd(u_max))
    rows = [dict(row="probe_direction", **eff_probe, **zrow(w, eff_probe), **probe_pred),
            dict(row="POSCTRL_pc1", **eff_pc1, **zrow(pc1, eff_pc1)),
            dict(row=f"POSCTRL_mvdf_maxU_feat{jmax}", **eff_umax, **zrow(u_max, eff_umax))]
    # ---------------- 6: MVDF decoder directions and probe-selected SAE features ----------
    Zev = Z[ev]
    for grp, feats in (("mvdf", r.mvdf), ("probe_sel", psel)):
        for j in feats:
            u = dec_dir(int(j))
            eff = steer_effect(model, Xe, ye, tv, r.prof_sel, he, u, proj_sd(u))
            a = Z[:, int(j)]
            pr = cv_eval(a[:, None], S, ctrl, emb, sseed) if np.std(a) > 0 else {}
            rows.append(dict(row=f"{grp}_feat{int(j)}", group=grp, **eff, **zrow(u, eff),
                             spearman=pr.get("spearman_within", np.nan),
                             circ_p=pr.get("circ_p", np.nan)))
    for rr in rows:
        rr.update(dataset=ds, seed=seed)
    np.savez_compressed(os.path.join(OUT, f"{r.tag}_probe.npz"), w=w, psel=psel, probe_pred_ev=p_ev,
                        ev_index=id_ix[ev])
    return loc, rows


if __name__ == "__main__":
    runs = [("ETTh2", s) for s in (0, 1, 2)] + [("ETTh1", s) for s in (0, 1, 2)]
    if len(sys.argv) > 1:
        runs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    LOC, ROWS = [], []
    for ds, s in runs:
        print("WP4", ds, s, flush=True)
        l, rw = run(ds, s)
        LOC += l; ROWS += rw
        suf = "" if len(sys.argv) < 2 else "_" + "_".join(a.replace(":", "") for a in sys.argv[1:])
        pd.DataFrame(LOC).to_csv(os.path.join(OUT, f"locate{suf}.csv"), index=False)
        pd.DataFrame(ROWS).drop(columns=["mean_S"]).to_csv(os.path.join(OUT, f"interventions{suf}.csv"), index=False)
    print("WP4 done")
