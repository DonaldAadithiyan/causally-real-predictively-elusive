"""Round 2 WP1: regenerate the paper's tables on ETTh1/ETTh2 on the corrected path.
Outputs runs/p126_round2/wp1/.

Protocols (WP0 task 6):
  P1 chrono_static   ID 50/30/20 chronological, embargo L+H, static bands
  P2 chrono_aci      same split, delayed-feedback ACI (delay H) on every method
  P3 random_blocks   ID cut into 10 contiguous blocks, roles 5/3/2 assigned at random,
                     embargo L+H at role boundaries, static bands, 30 assignments
Shift block: P1/P2 with calibration from the chronological cal segment.
S_t everywhere uses the E_select profile (C3)."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ.setdefault("P126_ARR", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs", "p126_round2", "arrays"))

from p126.common import (Run, REAL, ALPHA, HERE, conf_q, boot_ci, n_eff, spearman,  # noqa
                         matched_width, dump, auroc)
from p126.corrected import online, Scorer, cqr_scores, chrono_split, random_block_split  # noqa
from p126.work_a import feature_sets  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402

OUT = os.path.join(HERE, "runs", "p126_round2", "wp1")
os.makedirs(OUT, exist_ok=True)
MODES = ["dense", "mvdf", "correlated", "random_k", "pca_k"]
GRID = np.linspace(0.01, 0.4, 40)


def method_bases(r, fs, fit, cal, sseed):
    """Return {method: fn(alpha) -> base half-width on ANY index set} fitted on fit/cal."""
    S = r.S
    out = {}
    out["naive_matched"] = lambda al, ix: np.full(len(ix), conf_q(np.r_[S[fit], S[cal]], 1 - al))
    out["naive_no_scorer"] = lambda al, ix: np.full(len(ix), conf_q(S[cal], 1 - al))
    for m in MODES:
        F = fs[m]
        sc = Scorer(seed=sseed).fit(F[fit], S[fit]).conformalize(F[cal], S[cal])
        out[m] = (lambda al, ix, sc=sc, F=F: sc.base(F[ix], al))
    return out


def eval_set(bases, S_te, te, kind, H, base_name="naive_matched"):
    rows = {}
    bfn = lambda al: online(bases[base_name](al, te), S_te, al, delay=H, kind=kind)["hw"]
    bhw = bfn(ALPHA)
    bcov = float(np.mean(S_te <= bhw))
    for m, f in bases.items():
        fn = lambda al, f=f: online(f(al, te), S_te, al, delay=H, kind=kind)["hw"]
        hw = fn(ALPHA)
        c = (S_te <= hw).astype(float)
        Wm, al_m, _ = matched_width(fn, S_te, bcov, grid=GRID)
        clo, chi, _ = boot_ci(lambda ix: c[ix].mean(), len(c), 2 * H, 2000, 0)
        rows[m] = dict(cov=float(c.mean()), cov_ci=[clo, chi], width=float(np.mean(2 * hw)),
                       n_eff=n_eff(c),
                       red_vs_naive_matched=float((np.mean(2 * bhw) - Wm) / np.mean(2 * bhw)),
                       matched_alpha=al_m)
    return rows


def run(ds, seed, n_random=30):
    r = Run(ds, seed)
    H, L = r.H, r.L
    emb = L + H
    sseed = derive_seed(seed, ds)
    id_ix = r.idx_id
    n = len(id_ix)
    rows = []
    # ---------------- P1 / P2 chronological ----------------
    f, c, t = chrono_split(n, emb=emb)
    fit, cal, te_id = id_ix[f], id_ix[c], id_ix[t]
    fs = feature_sets(r, fit, sseed)
    bases = method_bases(r, fs, fit, cal, sseed)
    for seg, te in (("ID", te_id), ("shift", r.idx_sh)):
        S_te = r.S[te]
        for kind, proto in (("static", "P1_chrono_static"), ("aci", "P2_chrono_delayed_aci")):
            for m, d in eval_set(bases, S_te, te, kind, H).items():
                rows.append(dict(dataset=ds, seed=seed, protocol=proto, segment=seg, method=m,
                                 n_test=len(te), n_blocks=len(te) / (2 * H), **d))
    # ---------------- P3 random contiguous blocks (ID only, static) ----------------
    rng = np.random.default_rng(sseed)
    acc = []
    for rep in range(n_random):
        f, c, t = random_block_split(n, rng, emb=emb)
        if len(t) < 50 or len(c) < 150:
            continue
        fit_r, cal_r, te_r = id_ix[f], id_ix[c], id_ix[t]
        fs_r = feature_sets(r, fit_r, sseed)
        b_r = method_bases(r, fs_r, fit_r, cal_r, sseed + rep)
        for m, d in eval_set(b_r, r.S[te_r], te_r, "static", H).items():
            acc.append(dict(rep=rep, method=m, **d))
    a = pd.DataFrame(acc)
    for m, g in a.groupby("method"):
        rows.append(dict(dataset=ds, seed=seed, protocol="P3_random_blocks", segment="ID",
                         method=m, n_test=np.nan, cov=g["cov"].mean(),
                         cov_q05=g["cov"].quantile(.05), cov_q95=g["cov"].quantile(.95),
                         width=g["width"].mean(), n_eff=g["n_eff"].mean(),
                         red_vs_naive_matched=g["red_vs_naive_matched"].mean(),
                         red_q05=g["red_vs_naive_matched"].quantile(.05),
                         red_q95=g["red_vs_naive_matched"].quantile(.95), reps=len(g)))
    a.to_csv(os.path.join(OUT, f"{r.tag}_P3_reps.csv"), index=False)

    # ---------------- standard baselines: naive, PID, CQR static / delayed ACI ----------
    base_rows = []
    E = cqr_scores(r.q_lo.astype(float), r.q_hi.astype(float), r.y.astype(float), r.prof_sel)
    for seg, te in (("ID", te_id), ("shift", r.idx_sh)):
        S_te = r.S[te]
        q80 = lambda al: conf_q(np.r_[r.S[fit], r.S[cal]], 1 - al)
        qc = lambda al: conf_q(E[cal], 1 - al)
        rawW = float((r.q_hi[te] - r.q_lo[te]).mean())
        defs = {
            ("naive_matched", "static"): (lambda al: online(q80(al), S_te, al, delay=H, kind="static")["hw"], S_te, 0.0),
            ("naive_matched", "aci"): (lambda al: online(q80(al), S_te, al, delay=H, kind="aci")["hw"], S_te, 0.0),
            ("pid_conformal", "pid"): (lambda al: online(q80(al), S_te, al, delay=H, kind="pid")["hw"], S_te, 0.0),
            ("cqr", "static"): (lambda al: online(qc(al), E[te], al, delay=H, kind="static")["hw"], E[te], rawW),
            ("cqr", "aci"): (lambda al: online(qc(al), E[te], al, delay=H, kind="aci")["hw"], E[te], rawW),
        }
        for (m, kind), (fn, sc, extra) in defs.items():
            hw = fn(ALPHA)
            c = (sc <= hw).astype(float)
            ref_kind = "static" if kind == "static" else "aci"
            rfn = defs[("naive_matched", ref_kind)][0]
            rhw = rfn(ALPHA)
            rcov = float(np.mean(S_te <= rhw)); rW = float(np.mean(2 * rhw))
            Wm, _, _ = matched_width(lambda al: fn(al), sc, rcov, grid=GRID)
            base_rows.append(dict(dataset=ds, seed=seed, segment=seg, method=m, update=kind,
                                  cov=float(c.mean()), n_eff=n_eff(c),
                                  width=float(np.mean(2 * hw)) + extra,
                                  like_for_like_ref=f"naive_matched/{ref_kind}",
                                  ref_cov=rcov, ref_width=rW,
                                  red_vs_ref_matched=float((rW - (Wm + extra)) / rW),
                                  crossing_rate=float(np.mean(r.q_lo[te] > r.q_hi[te]))
                                  if m == "cqr" else np.nan))

    # ---------------- regime gating (H7-A R_Q) ----------------
    gate_rows = []
    cp = os.path.join(HERE, f"runs/phase3_real/{r.tag}/candidates.csv")
    if not os.path.exists(cp):                      # WP6 datasets: written by r2_wp6.regen
        cp = os.path.join(HERE, "runs", "p126_round2", "wp6", f"{r.tag}_candidates.csv")
    cand = pd.read_csv(cp)
    cellB = cand[cand["type"] == "B"].feature.values.astype(int)
    Zb = r.z_cols(cellB)
    A = (Zb != 0).mean(1)
    tau = float(np.percentile(A[r.idx_sel], 90))
    for seg, ix in (("E_eval_ID", id_ix), ("E_eval_shift", r.idx_sh)):
        s, d = r.S[ix], (A[ix] >= tau)

        def rq(jx):
            dd, ss = d[jx], s[jx]
            if dd.sum() < 10 or (~dd).sum() < 10:
                return np.nan
            return np.quantile(ss[dd], .9) / np.quantile(ss[~dd], .9)
        lo, hi, nbk = boot_ci(rq, len(ix), 2 * H, 2000, sseed)
        gate_rows.append(dict(dataset=ds, seed=seed, env=seg, n_cellB=len(cellB), tau_A=tau,
                              n1=int(d.sum()), R_Q=rq(np.arange(len(ix))), ci_lo=lo, ci_hi=hi,
                              n_blocks=nbk, excludes_1_above=bool(lo > 1)))

    # ---------------- localization (SLCP local variance, +/-40 exclusion) ----------------
    from sklearn.decomposition import PCA
    sel = r.idx_sel
    k = len(r.mvdf)
    zsel = r.z_idx[sel]
    act_rate = np.zeros(int(r.d_sae))
    for cc in np.unique(zsel[zsel >= 0]):
        act_rate[cc] = (zsel == cc).any(1).mean()
    elig = np.where(act_rate >= 0.02)[0]
    rand = np.sort(np.random.default_rng(sseed + 777).choice(elig, size=min(k, len(elig)), replace=False))
    pca = PCA(n_components=k).fit(r.hidden[sel].astype(np.float64))
    spaces = {"dense": r.hidden[id_ix].astype(np.float64), "mvdf": r.z_cols(r.mvdf)[id_ix],
              "correlated": r.z_cols(r.correlated)[id_ix], "random_k": r.z_cols(rand)[id_ix],
              "pca_k": pca.transform(r.hidden[id_ix].astype(np.float64))}
    S_id = r.S[id_ix]
    samp = np.sort(np.random.default_rng(sseed + 999).choice(n, size=min(800, n), replace=False))
    lv = {}
    for sp, Fm in spaces.items():
        Fz = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-8)
        out = np.empty(len(samp))
        for i, t_ in enumerate(samp):
            dd = np.sqrt(((Fz - Fz[t_]) ** 2).sum(1))
            dd[max(0, t_ - 40):min(n, t_ + 41)] = np.inf
            nn = np.argpartition(dd, 50)[:50]
            out[i] = np.var(S_id[nn])
        lv[sp] = out
    loc_rows = []
    pos = np.full(n, -1); pos[samp] = np.arange(len(samp))
    for sp in spaces:
        def ratio(jx):
            m_ = pos[jx]; m_ = m_[m_ >= 0]
            return lv[sp][m_].mean() / lv["dense"][m_].mean() if len(m_) else np.nan
        lo, hi, nbk = boot_ci(ratio, n, 2 * H, 2000, sseed + 3)
        loc_rows.append(dict(dataset=ds, seed=seed, space=sp,
                             ratio_vs_dense=float(lv[sp].mean() / lv["dense"].mean()),
                             ci_lo=lo, ci_hi=hi, n_blocks=nbk))
    return rows, base_rows, gate_rows, loc_rows


def item_a_check():
    """Round-1 item A metrics under the normalisation fix (S with E_select profile)."""
    from p126.work_a import fit_predict, r2_decomp
    rows = []
    for ds, s in REAL:
        r = Run(ds, s)
        sseed = derive_seed(s, ds)
        id_ix = r.idx_id; n = len(id_ix); cpt = int(0.6 * n)
        F = r.z_cols(r.mvdf)[id_ix]
        for sname, S in (("S_orig_blockprof", r.S_local(id_ix)[0]), ("S_corrected", r.S[id_ix])):
            yf, ye = S[:cpt], S[cpt:]
            for est, p in fit_predict(F[:cpt], yf, F[cpt:], sseed).items():
                d = r2_decomp(p, ye, yf)
                rows.append(dict(dataset=ds, seed=s, S=sname, estimator=est, **d,
                                 spearman=spearman(p, ye),
                                 auroc_top10=auroc(p, ye >= np.quantile(ye, .9))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "itemA_normalisation_check.csv"), index=False)
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "itemA":
        print(item_a_check().groupby(["dataset", "S"])[["R2_test", "R2_const", "skill", "spearman"]]
              .agg(["min", "max"]).round(3).to_string())
        sys.exit()
    runs = REAL if len(sys.argv) < 2 else [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    A, B, G, Lc = [], [], [], []
    for ds, s in runs:
        print("WP1", ds, s, flush=True)
        a, b, g, l = run(ds, s)
        A += a; B += b; G += g; Lc += l
    suf = "" if len(sys.argv) < 2 else "_" + "_".join(x.replace(":", "") for x in sys.argv[1:])
    pd.DataFrame(A).to_csv(os.path.join(OUT, f"continuous_scoring{suf}.csv"), index=False)
    pd.DataFrame(B).to_csv(os.path.join(OUT, f"standard_baselines{suf}.csv"), index=False)
    pd.DataFrame(G).to_csv(os.path.join(OUT, f"regime_gating{suf}.csv"), index=False)
    pd.DataFrame(Lc).to_csv(os.path.join(OUT, f"localization{suf}.csv"), index=False)
    print("WP1 done")
