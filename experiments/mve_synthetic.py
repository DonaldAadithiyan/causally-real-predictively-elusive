"""Controlled sanity check with KNOWN ground truth (spec §4.5).

This is NOT a scientific result. It runs the exact same validation + calibration
machinery as the real experiment, but on the synthetic forecaster/series where we
injected:
  - a MECHANISTIC difficulty coordinate  (head reads it, tracks difficulty)
  - a SPURIOUS coordinate                 (correlates with |e| pre-shift only,
                                           head ignores it)

Expected, if the machinery is correct:
  RQ1  : the two rankings diverge (the spurious feature ranks high on correlation,
         low on |Delta_j|).
  H1   : rq1 case studies surface the spurious feature.
  H3   : under the synthetic shift (spurious coupling switched off), the
         correlation-only condition degrades more than MVDF.

If this does NOT hold, the pipeline has a bug — fix it here before trusting the
real run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from forecasters.synthetic_forecaster import SeriesConfig, generate_series, make_windows
from forecasters.train_forecaster import train_forecaster, forecaster_outputs
from sae.train_sae import train_sae, encode_all
from validation.correlation_screen import screen, top_k_correlated
from validation.rq1 import rq1_diagnostic
from eval import plots


def _syn_collect(model, X, Y):
    """Mirror validation.errors.collect for the synthetic (1-step) forecaster."""
    yhat, hidden = forecaster_outputs(model, X)
    score = np.abs(yhat - Y)
    return dict(yhat=yhat, y=Y, hidden=hidden.astype(np.float32), score=score.astype(np.float32))


def _syn_score_patched(model, X, Y, patched_hidden):
    with torch.no_grad():
        yb = torch.from_numpy(X)
        h = torch.from_numpy(patched_hidden.astype(np.float32))
        yhat, _ = model(yb, h_override=h)
    return np.abs(yhat.numpy() - Y).astype(np.float32)


def _syn_ablate(model, sae, X, Y, hidden, feat_idx, verbose=True):
    with torch.no_grad():
        z = sae.encode(torch.from_numpy(hidden)).numpy()
        h_recon = sae.decode(torch.from_numpy(z)).numpy()
    base = _syn_score_patched(model, X, Y, h_recon)
    delta = np.zeros(len(feat_idx))
    active = np.zeros(len(feat_idx))
    for t, j in enumerate(feat_idx):
        zz = z.copy(); zz[:, int(j)] = 0.0
        with torch.no_grad():
            h_abl = sae.decode(torch.from_numpy(zz)).numpy()
        abl = _syn_score_patched(model, X, Y, h_abl)
        delta[t] = float(np.mean(abl - base))
        active[t] = float((z[:, int(j)] != 0).mean())
        if verbose and t % max(1, len(feat_idx) // 10) == 0:
            print(f"[syn-ablation] {t+1}/{len(feat_idx)} feat {int(j)} Delta={delta[t]:+.5f}")
    return dict(delta=delta, abs_delta=np.abs(delta), active_frac=active,
               feature_indices=np.asarray(feat_idx), order=np.argsort(-np.abs(delta)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="tiny run for a smoke test")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(here, "runs", "mve_synthetic")
    os.makedirs(out, exist_ok=True)
    np.random.seed(0); torch.manual_seed(0)

    scfg = SeriesConfig(n_timesteps=4000 if args.fast else 16000)
    mech_dim, spur_dim = 8, 8
    k_budget = 8 if args.fast else 12

    # ---- data: pre-shift for train/cal, post-shift for the shift eval -------------
    pre = generate_series(scfg, regime="preshift")
    post = generate_series(scfg, regime="postshift")
    Xp, Yp, auxp = make_windows(pre, scfg)
    Xs, Ys, auxs = make_windows(post, scfg)

    n = len(Xp)
    i_tr, i_cal = slice(0, int(0.6 * n)), slice(int(0.6 * n), int(0.8 * n))
    i_te = slice(int(0.8 * n), n)

    # ---- forecaster -------------------------------------------------------------
    model = train_forecaster(
        Xp[i_tr], Yp[i_tr], {k: v[i_tr] for k, v in auxp.items()},
        hidden_dim=256, mech_dim=mech_dim, spur_dim=spur_dim,
        epochs=3 if args.fast else 10, verbose=True,
    )

    tr = _syn_collect(model, Xp[i_tr], Yp[i_tr])
    cal = _syn_collect(model, Xp[i_cal], Yp[i_cal])
    te_id = _syn_collect(model, Xp[i_te], Yp[i_te])
    te_sh = _syn_collect(model, Xs[i_te], Ys[i_te])

    # ---- SAE on training hidden states only -----------------------------------
    sae = train_sae(tr["hidden"], expansion=8, k=64,
                    steps=2000 if args.fast else 12000, verbose=True)

    z_cal = encode_all(sae, cal["hidden"])
    scr = screen(z_cal, cal["score"], min_active_frac=0.02)

    # ---- RQ1: ablate a non-correlation-prefiltered sample --------------------
    rng = np.random.default_rng(0)
    elig = np.where(scr["eligible"])[0]
    samp = rng.choice(elig, size=min(200 if args.fast else 400, len(elig)), replace=False)
    abl = _syn_ablate(model, sae, cal["hidden"][:2048], cal["score"][:2048][:len(cal["score"][:2048])],
                      cal["hidden"][:2048], samp, verbose=True)
    # fix Y alignment for ablation eval set
    abl = _syn_ablate(model, sae, Xp[i_cal][:2048], Yp[i_cal][:2048], cal["hidden"][:2048], samp, verbose=False)

    rq1 = rq1_diagnostic(scr, abl)
    with open(os.path.join(out, "rq1_diagnostic.json"), "w") as f:
        json.dump(rq1, f, indent=2)
    print("\n[RQ1]", json.dumps({k: rq1[k] for k in ("spearman_rho", "top_set_jaccard", "interpretation")}, indent=2))

    plots.delta_j_distribution(abl["abs_delta"], os.path.join(out, "delta_j_distribution.png"))
    plots.rq1_scatter(np.abs(scr["pearson"][abl["feature_indices"]]),
                      abl["abs_delta"], os.path.join(out, "rq1_scatter.png"),
                      rq1.get("spearman_rho", float("nan")))

    # ---- ground-truth check: which SAE feature best matches the mech/spurious blocks
    #      (correlate each SAE code with the mechanistic vs spurious hidden coord)
    h_cal = cal["hidden"]
    mech_sig = h_cal[:, :mech_dim].mean(1)
    spur_sig = h_cal[:, mech_dim:mech_dim + spur_dim].mean(1)
    corr_mech = np.array([np.corrcoef(z_cal[:, j], mech_sig)[0, 1] if z_cal[:, j].std() > 0 else 0
                          for j in range(z_cal.shape[1])])
    corr_spur = np.array([np.corrcoef(z_cal[:, j], spur_sig)[0, 1] if z_cal[:, j].std() > 0 else 0
                          for j in range(z_cal.shape[1])])
    gt = dict(
        top_feature_for_mechanistic_block=int(np.argmax(np.abs(corr_mech))),
        its_abs_corr_with_mech=float(np.abs(corr_mech).max()),
        top_feature_for_spurious_block=int(np.argmax(np.abs(corr_spur))),
        its_abs_corr_with_spur=float(np.abs(corr_spur).max()),
    )
    with open(os.path.join(out, "ground_truth_feature_map.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print("[ground truth]", json.dumps(gt, indent=2))
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
