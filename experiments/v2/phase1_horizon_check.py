"""Pre-run design check for the Phase 1 synthetic control (no forecaster, no SAE).

Diagnosed problem: with regime AR(1) coefficients a_0=0.3, a_1=0.9, an AR forecast
decays toward 0 fast (half-lives ~0.6 and ~6.6 steps). At H=96 both regimes have
converged to noise-dominated dynamics for most of the horizon, so the regime's
difficulty signal is diluted below any detection floor for reasons unrelated to the
pipeline. That would give Phase 1 an ambiguous null.

This script simulates the DGP directly and, for a range of candidate horizons H,
measures how distinguishable the two regimes still are in the quantity the pipeline
actually keys on: the *irreducible one-window nonconformity level* under an optimal
point forecast. Concretely, for a window whose last input step is in regime z:

  - the optimal H-step point forecast of x is the AR mean recursion from x_L
    (using the regime the chain is actually in, which the model can in principle
    infer from local autocorrelation);
  - the residual at horizon step h has variance = AR innovation-accumulation
    variance(h, regime path) + sigma(z_h)^2  (the heteroscedastic term);
  - the window nonconformity score S = max_h |resid_h| / profile_h.

We report, per candidate H, the ratio of median S in hard-started vs easy-started
windows (and a rank-based AUC of S separating the two), from a pure Monte-Carlo of
the frozen DGP. Freeze the smallest H whose separation is clearly sufficient.
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import json
import os

import numpy as np

from mechconf.data.synthetic_dgp import load_config, generate_series, _f_apply

HERE = REPO_ROOT


def incentive_pathway_check(cfg, seed=0, n_steps=40000, horizons=(12, 16, 24, 48, 96)):
    """Is the AR-decay training incentive to encode z alive at each candidate H?

    z affects the data through two channels:
      (1) sigma(z)  — noise VARIANCE only. Under MSE, E[y|x,z] = f(x) is z-independent,
          so this channel gives ZERO direct gradient incentive to encode z.
      (2) a_z        — the AR coefficient. E[x_{t+h}] = a_z^h * x_t, so a multi-step
          point forecast that knows a_z is genuinely better. This is the ONLY real
          MSE incentive for the forecaster to represent the regime.

    This measures channel (2) directly: the achievable test-MSE reduction on y from a
    forecaster that knows the true regime path (oracle AR-mean) vs one that does not
    (uses a single regime-marginal AR coefficient a_bar = stationary-mix average).
    If (mse_no_regime - mse_oracle) is a real, non-trivial quantity at the frozen H,
    the incentive pathway is alive and Option-1's SNR fix is sufficient. If it's ~0,
    a structural DGP change (z affecting f) would be needed.
    """
    d = cfg["dgp"]; L = cfg["windowing"]["lookback"]
    a = np.array([d["a_0"], d["a_1"]])

    rng = np.random.default_rng(seed)
    ser = generate_series(n_steps, cfg, rng)
    x, y, z = ser["x"].astype(np.float64), ser["y"].astype(np.float64), ser["z"]
    # regime-marginal AR coeff: stationary distribution of the 0.97/0.97 chain is 50/50
    p0 = d["markov_p_stay_0"]; p1 = d["markov_p_stay_1"]
    pi1 = (1 - p0) / ((1 - p0) + (1 - p1))          # stationary P(z=1)
    a_bar = (1 - pi1) * a[0] + pi1 * a[1]

    Hmax = max(horizons)
    T = len(x)
    starts = np.arange(0, T - L - Hmax + 1)
    ymean = y.mean(); ystd = y.std() + 1e-8
    out = {}
    for H in horizons:
        se_oracle = np.empty(len(starts)); se_noregime = np.empty(len(starts))
        for i, s in enumerate(starts):
            x_last = x[s + L - 1]
            zp = z[s + L - 1: s + L - 1 + H]
            # oracle: knows regime path
            cur = x_last; xo = np.empty(H)
            for h in range(H):
                cur = a[zp[h]] * cur; xo[h] = cur
            # no-regime: single a_bar
            xn = x_last * a_bar ** np.arange(1, H + 1)
            yt = y[s + L: s + L + H]
            yo = _f_apply(xo, d["f"]); yn = _f_apply(xn, d["f"])
            se_oracle[i] = np.mean((((yo - ymean) / ystd) - ((yt - ymean) / ystd)) ** 2)
            se_noregime[i] = np.mean((((yn - ymean) / ystd) - ((yt - ymean) / ystd)) ** 2)
        mse_o = float(se_oracle.mean()); mse_n = float(se_noregime.mean())
        out[H] = dict(mse_oracle_knows_regime=round(mse_o, 4),
                      mse_no_regime_info=round(mse_n, 4),
                      mse_gain_from_knowing_regime=round(mse_n - mse_o, 4),
                      relative_gain=round((mse_n - mse_o) / (mse_n + 1e-12), 4),
                      a_bar=round(float(a_bar), 4))
    return out


def run_check(cfg, seed=0, n_steps=40000, horizons=(6, 8, 12, 16, 24, 48, 96)):
    d = cfg["dgp"]; L = cfg["windowing"]["lookback"]
    a = np.array([d["a_0"], d["a_1"]])

    rng = np.random.default_rng(seed)
    ser = generate_series(n_steps, cfg, rng)
    x, y, z = ser["x"].astype(np.float64), ser["y"].astype(np.float64), ser["z"]

    out = {}
    Hmax = max(horizons)
    T = len(x)
    starts = np.arange(0, T - L - Hmax + 1)
    N = len(starts)
    # oracle AR-mean forecast for the longest horizon; slice for shorter ones
    xhat_full = np.empty((N, Hmax))
    for i, s in enumerate(starts):
        cur = x[s + L - 1]
        zpath = z[s + L - 1: s + L - 1 + Hmax]
        for h in range(Hmax):
            cur = a[zpath[h]] * cur
            xhat_full[i, h] = cur
    yhat_full = _f_apply(xhat_full, d["f"])
    y_win_full = np.stack([y[s + L: s + L + Hmax] for s in starts])
    z_start = np.array([z[s + L - 1] for s in starts])

    for H in horizons:
        ae = np.abs(y_win_full[:, :H] - yhat_full[:, :H])          # (N, H)
        profile = ae.mean(axis=0); profile = profile / (profile.mean() + 1e-12)
        S = (ae / profile[None, :]).max(axis=1)
        hard_frac = np.stack([z[s + L: s + L + H].mean() for s in starts])
        easy = S[z_start == 0]; hard = S[z_start == 1]
        # rank-AUC of S separating hard-start from easy-start windows
        from scipy import stats
        try:
            u = stats.mannwhitneyu(hard, easy, alternative="greater")
            auc = float(u.statistic / (len(hard) * len(easy)))
        except ValueError:
            auc = float("nan")
        # also separation by horizon hard-fraction (continuous)
        rho = float(stats.spearmanr(S, hard_frac).statistic)
        out[H] = dict(
            median_S_easy=float(np.median(easy)), median_S_hard=float(np.median(hard)),
            median_ratio_hard_over_easy=float(np.median(hard) / (np.median(easy) + 1e-12)),
            mean_ratio_hard_over_easy=float(np.mean(hard) / (np.mean(easy) + 1e-12)),
            rank_auc_hard_vs_easy=auc,
            spearman_S_vs_hardfrac=rho,
            n_easy=int((z_start == 0).sum()), n_hard=int((z_start == 1).sum()),
        )
    return out


def main():
    cfg = load_config()
    res = {s: run_check(cfg, seed=s) for s in (0, 1, 2)}
    inc = {s: incentive_pathway_check(cfg, seed=s) for s in (0, 1, 2)}
    inc_horizons = sorted(next(iter(inc.values())).keys())
    inc_summary = []
    for H in inc_horizons:
        g = np.mean([inc[s][H]["mse_gain_from_knowing_regime"] for s in inc])
        rg = np.mean([inc[s][H]["relative_gain"] for s in inc])
        inc_summary.append(dict(H=H, mse_gain_from_knowing_regime=round(float(g), 4),
                                relative_gain=round(float(rg), 4)))
    print("=== incentive-pathway check (AR-decay channel) ===")
    print(json.dumps(inc_summary, indent=2))
    print("=== regime-separation check (difficulty signal) ===")
    # average across seeds
    horizons = sorted(next(iter(res.values())).keys())
    summary = []
    for H in horizons:
        auc = np.mean([res[s][H]["rank_auc_hard_vs_easy"] for s in res])
        ratio = np.mean([res[s][H]["median_ratio_hard_over_easy"] for s in res])
        rho = np.mean([res[s][H]["spearman_S_vs_hardfrac"] for s in res])
        summary.append(dict(H=H, rank_auc=round(float(auc), 4),
                            median_S_ratio_hard_over_easy=round(float(ratio), 4),
                            spearman_S_vs_hardfrac=round(float(rho), 4)))
    out = dict(per_seed=res, summary=summary,
               note="Oracle-point-forecast S_t separation by regime, pure DGP Monte-Carlo. "
                    "Freeze the smallest H with rank_auc clearly > 0.6 and a median S ratio "
                    "clearly > 1.3 (task guidance). No forecaster / SAE involved.")
    p = os.path.join(HERE, "results", "v2", "phase1_synthetic_gates", "horizon_check.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
