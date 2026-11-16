"""Round 2 WP0: sanity test of the corrected path (task 4) + baseline-parity audit of
round-1 item B (task 5). Outputs results/round2/p126_round2/wp0/."""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore")

from experiments.round2.common import (REAL, ALPHA, HERE, conf_q, profile_of, score_with, boot_ci, n_eff,  # noqa
                         spearman, matched_width, dump)
from experiments.round2.corrected import online, Scorer, cqr_scores, naive_base  # noqa: E402

R2 = os.path.join(HERE, "results", "round2", "p126_round2")
OUT = os.path.join(R2, "wp0")
os.makedirs(OUT, exist_ok=True)
H = 96


# ------------------------------------------------------------------ task 4
def sanity(reps=300, scorer_reps=100, n_fit=983, n_cal=397, n_tests=(394, 1871, 8000), seed=0):
    rng = np.random.default_rng(seed)
    prof_true = np.load(os.path.join(HERE, "results/round2/p126/arrays/ETTh1_seed0.npz"))
    ae = np.abs(prof_true["y"] - prof_true["yhat"])
    prof_true = profile_of(ae[prof_true["m_select"]])
    res = []
    for n_te in n_tests:
        acc = {}
        R = reps if n_te < 5000 else 60
        for rep in range(R):
            N = n_fit + n_cal + n_te
            e = prof_true[None, :] * rng.standard_normal((N, H))
            # calibration profile from fit+cal only (C3), applied unchanged to test
            p_cal = profile_of(np.abs(e[:n_fit + n_cal]))
            S = score_with(np.abs(e), p_cal)
            fit, cal, te = slice(0, n_fit), slice(n_fit, n_fit + n_cal), slice(n_fit + n_cal, N)
            q80 = naive_base(S[:n_fit + n_cal], ALPHA)
            for kind in ("static", "aci", "pid"):
                r = online(q80, S[te], ALPHA, delay=H, kind=kind)
                acc.setdefault(f"naive_{kind}", []).append(r["covered"].mean())
            lo = -1.6 * prof_true[None, :] + 0.1 * rng.standard_normal((N, H))
            hi = 1.6 * prof_true[None, :] + 0.1 * rng.standard_normal((N, H))
            E = cqr_scores(lo, hi, e, p_cal)
            qh = conf_q(E[cal], 1 - ALPHA)
            for kind in ("static", "aci"):
                r = online(qh, E[te], ALPHA, delay=H, kind=kind)
                acc.setdefault(f"cqr_{kind}", []).append(r["covered"].mean())
            if rep < scorer_reps and n_te < 5000:
                F = np.where(rng.random((N, 12)) < 0.03, rng.exponential(1, (N, 12)), 0.0)
                sc = Scorer(seed=rep).fit(F[fit], S[fit]).conformalize(F[cal], S[cal])
                b = sc.base(F[te])
                for kind in ("static", "aci"):
                    r = online(b, S[te], ALPHA, delay=H, kind=kind)
                    acc.setdefault(f"scorer_{kind}", []).append(r["covered"].mean())
        for k, v in acc.items():
            v = np.array(v)
            se = v.std(ddof=1) / np.sqrt(len(v))
            res.append(dict(n_test=n_te, method=k, mean_cov=float(v.mean()), mc_se=float(se),
                            sd=float(v.std()), reps=len(v),
                            within_3se=bool(abs(v.mean() - 0.9) <= 3 * se)))
        print(pd.DataFrame(res[-len(acc):]).round(4).to_string(), flush=True)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "sanity_exchangeable.csv"), index=False)
    return df


# ------------------------------------------------------------------ task 5
def parity(embargo_tag="emb"):
    """Round-1 B compared each predictor (binned/normalised, STATIC, calibrated on the 30%
    conformalize segment) against a STATIC constant band calibrated on fit+cal (80%).
    Recompute like-for-like: (i) static vs static constant band on the SAME cal segment,
    (ii) delayed-ACI predictor vs delayed-ACI constant band (same cal segment).
    Normalised construction (finite at every alpha). ID and shift test segments."""
    rows = []
    for ds, s in REAL:
        z = np.load(os.path.join(HERE, f"results/round2/p126/B/{ds}_seed{s}_{embargo_tag}_preds.npz"))
        S, fit, cal = z["S"], z["fit"], z["cal"]
        preds = [k[2:] for k in z.files if k.startswith("g_")]
        for seg, te in (("ID", z["te_id"]), ("shift", z["te_sh"])):
            S_te, S_cal = S[te], S[cal]
            # baselines
            q30 = lambda al: conf_q(S_cal, 1 - al)
            q80 = lambda al: conf_q(np.r_[S[fit], S_cal], 1 - al)
            base = {}
            for bname, qf in (("const_cal30", q30), ("const_cal80_round1", q80)):
                for kind in ("static", "aci"):
                    fn = (lambda al, qf=qf, kind=kind:
                          online(qf(al), S_te, al, delay=H, kind=kind)["hw"])
                    hw = fn(ALPHA)
                    base[(bname, kind)] = (fn, float(np.mean(S_te <= hw)), float(np.mean(2 * hw)))
            for p in preds:
                g = z[f"g_{p}"].astype(float)
                gc, gt = g[cal], g[te]
                gc = np.where(np.isfinite(gc), gc, np.nanmedian(gc))
                gt = np.where(np.isfinite(gt), gt, np.nanmedian(gc))
                if np.any(gc <= 0):
                    continue
                ratio = S_cal / gc
                for kind in ("static", "aci"):
                    fn = (lambda al, kind=kind: online(conf_q(ratio, 1 - al) * gt, S_te, al,
                                                       delay=H, kind=kind)["hw"])
                    hw = fn(ALPHA)
                    cov = float(np.mean(S_te <= hw))
                    row = dict(dataset=ds, seed=s, segment=seg, predictor=p, update=kind,
                               cov=cov, width=float(np.mean(2 * hw)), n_test=len(te),
                               n_eff=n_eff((S_te <= hw).astype(float)),
                               spearman=spearman(gt, S_te))
                    lo, hi, _ = boot_ci(lambda ix: spearman(gt[ix], S_te[ix]), len(te), 2 * H,
                                        2000, s)
                    row["spearman_ci"] = [lo, hi]
                    for bname in ("const_cal30", "const_cal80_round1"):
                        bfn, bcov, bw = base[(bname, kind)]
                        Wm, al_m, _ = matched_width(fn, S_te, bcov,
                                                    grid=np.linspace(0.01, 0.4, 40))
                        hw_m = fn(al_m)
                        bhw = bfn(ALPHA)
                        red = (bw - Wm) / bw
                        lo, hi, nbk = boot_ci(lambda ix: (np.mean(2 * bhw[ix]) - np.mean(2 * hw_m[ix]))
                                              / np.mean(2 * bhw[ix]), len(te), 2 * H, 2000, s + 1)
                        row[f"{bname}_cov"] = bcov
                        row[f"red_vs_{bname}"] = float(red)
                        row[f"red_vs_{bname}_ci"] = [lo, hi]
                        row["n_blocks"] = nbk
                    rows.append(row)
        print("parity", ds, s, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"parity_{embargo_tag}.csv"), index=False)
    return df


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("all", "sanity"):
        sanity()
    if w in ("all", "parity"):
        parity("emb")
