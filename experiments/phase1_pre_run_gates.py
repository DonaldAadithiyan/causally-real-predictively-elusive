"""Pre-run correction 4 — DGP-validity gates A/B/C (no forecaster, no SAE, no pipeline).

All three gates are pre-declared in configs/synthetic_dgp_config.json
_pre_run_correction_4.gates_pre_declared_before_running BEFORE this script is run.
This script only measures them. It is blind to anything downstream of the DGP.

  Gate A — detectability: an IDEALIZED regime detector (hand-crafted autocorrelation
    lags {1,2,3,5,8} + log-variance of the 48-step window -> logistic regression,
    4-fold CV) must reach >= 0.75 accuracy. Baseline ~0.50 at 200-step dwell / 50-50
    stationary mix. This is the check whose failure at the old params (67.6% vs 59.9%)
    triggered correction 4.

  Gate B — episode counts: each of E_train / E_select / E_eval_ID / E_eval_shift must
    contain >= 30 independent regime EPISODES (contiguous runs of z) of EACH regime.

  Gate C — incentive + separation: re-uses experiments/phase1_horizon_check.py at H=24
    (incentive relative_gain > 0.02, oracle-S_t regime rank-AUC > 0.6).

Writes runs/phase1_synthetic_gates/pre_run_gates_v4.json. If A/B/C all clear, the next step
is the blind mu_z grid (experiments/phase1_muz_gate.py); otherwise escalate to Option 2
(do NOT run the full pipeline).
"""
from __future__ import annotations

import json
import os

import numpy as np

from data.synthetic_dgp import load_config, build_split_series, generate_series, make_windows_xy
from experiments.phase1_horizon_check import incentive_pathway_check, run_check

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATE_A_THRESHOLD = 0.75
GATE_B_MIN_EPISODES = 30
GATE_C_MIN_INCENTIVE = 0.02
GATE_C_MIN_AUC = 0.60


def _window_feats(w: np.ndarray) -> list:
    w0 = w - w.mean()
    v = np.var(w0) + 1e-9
    acs = [float(np.sum(w0[:-k] * w0[k:]) / (len(w0) * v)) for k in (1, 2, 3, 5, 8)]
    return acs + [float(np.log(v))]


def gate_A_detectability(cfg, seeds=(0, 1, 2)) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    L = cfg["windowing"]["lookback"]
    accs, bases = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        ser = generate_series(60000, cfg, rng, mu_z_override=0.0)
        x = (ser["x"] - ser["x"].mean()) / (ser["x"].std() + 1e-8)
        z = ser["z"]
        T = len(x)
        starts = np.arange(0, T - L - 1, 2)
        F = np.array([_window_feats(x[s0:s0 + L]) for s0 in starts])
        zw = z[starts + L - 1]
        base = float(max((zw == 0).mean(), (zw == 1).mean()))
        acc = float(np.mean(cross_val_score(LogisticRegression(max_iter=500), F, zw, cv=4)))
        accs.append(acc); bases.append(base)
    return dict(mean_cv_acc=round(float(np.mean(accs)), 4),
               per_seed_acc=[round(a, 4) for a in accs],
               mean_baseline=round(float(np.mean(bases)), 4),
               mean_gain_over_baseline=round(float(np.mean(accs) - np.mean(bases)), 4),
               threshold=GATE_A_THRESHOLD,
               PASS=bool(np.mean(accs) >= GATE_A_THRESHOLD))


def _count_episodes(z: np.ndarray) -> tuple[int, int]:
    """Number of contiguous runs of regime 0 and regime 1."""
    if len(z) == 0:
        return 0, 0
    change = np.where(np.diff(z) != 0)[0]
    starts = np.concatenate([[0], change + 1])
    runs_val = z[starts]
    return int((runs_val == 0).sum()), int((runs_val == 1).sum())


def gate_B_episode_counts(cfg, seeds=(0, 1, 2)) -> dict:
    L, H = cfg["windowing"]["lookback"], cfg["windowing"]["horizon"]
    from shift.shift_windows import three_way_env_masks
    rows = []
    worst = {"E_train": (1e9, 1e9), "E_select": (1e9, 1e9),
             "E_eval_ID": (1e9, 1e9), "E_eval_shift": (1e9, 1e9)}
    for s in seeds:
        S = build_split_series(cfg, s)
        z_tr = S["train"]["z"]
        e0_tr, e1_tr = _count_episodes(z_tr)
        # test split carved 1/3 / 1/3 / 1/3 by window end index -> map back to z indices
        z_te = S["test"]["z"]
        _, _, _, _, end_te = make_windows_xy(
            (S["test"]["x"] - S["train"]["x"].mean()) / (S["train"]["x"].std() + 1e-8),
            S["test"]["y"], z_te, L, H)
        env = three_way_env_masks(end_te, len(z_te))
        # episode count over the z spanned by each env's windows
        seg = {}
        for tag in ("E_select", "E_eval_ID", "E_eval_shift"):
            idx = np.where(env[tag])[0]
            if len(idx) == 0:
                seg[tag] = (0, 0); continue
            lo = idx.min(); hi = idx.max() + L + H
            seg[tag] = _count_episodes(z_te[lo:hi])
        allseg = {"E_train": (e0_tr, e1_tr), **seg}
        rows.append(dict(seed=s, **{k: list(v) for k, v in allseg.items()}))
        for k, v in allseg.items():
            worst[k] = (min(worst[k][0], v[0]), min(worst[k][1], v[1]))
    min_any = min(min(v) for v in worst.values())
    return dict(per_seed=rows,
               worst_case_episodes_per_regime={k: list(v) for k, v in worst.items()},
               min_episodes_any_split_any_regime=int(min_any),
               threshold=GATE_B_MIN_EPISODES,
               PASS=bool(min_any >= GATE_B_MIN_EPISODES))


def gate_C_incentive_separation(cfg) -> dict:
    H = cfg["windowing"]["horizon"]
    inc = {s: incentive_pathway_check(cfg, seed=s, horizons=(H,)) for s in (0, 1, 2)}
    sep = {s: run_check(cfg, seed=s, horizons=(H,)) for s in (0, 1, 2)}
    rel_gain = float(np.mean([inc[s][H]["relative_gain"] for s in inc]))
    auc = float(np.mean([sep[s][H]["rank_auc_hard_vs_easy"] for s in sep]))
    s_ratio = float(np.mean([sep[s][H]["median_ratio_hard_over_easy"] for s in sep]))
    return dict(H=H, incentive_relative_gain=round(rel_gain, 4),
               oracle_S_rank_auc=round(auc, 4),
               oracle_S_median_ratio_hard_over_easy=round(s_ratio, 4),
               thresholds=dict(incentive=GATE_C_MIN_INCENTIVE, auc=GATE_C_MIN_AUC),
               PASS=bool(rel_gain > GATE_C_MIN_INCENTIVE and auc > GATE_C_MIN_AUC))


def main():
    cfg = load_config()
    A = gate_A_detectability(cfg)
    B = gate_B_episode_counts(cfg)
    C = gate_C_incentive_separation(cfg)
    allpass = bool(A["PASS"] and B["PASS"] and C["PASS"])
    out = dict(
        config_dgp={k: cfg["dgp"][k] for k in ("a_0", "a_1", "sigma_easy", "sigma_hard",
                    "markov_p_stay_0", "mean_dwell_steps", "n_timesteps")},
        gate_A_detectability=A, gate_B_episode_counts=B, gate_C_incentive_separation=C,
        ALL_PASS=allpass,
        next_step=("run blind mu_z grid: python -m experiments.phase1_muz_gate"
                   if allpass else
                   "gate(s) failed -> escalate to Option 2 (directly-observable regime "
                   "driver) as a labeled second experiment. DO NOT run the full pipeline."),
    )
    p = os.path.join(HERE, "runs", "phase1_synthetic_gates", "pre_run_gates_v4.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")
    print(f"\n{'ALL GATES PASS' if allpass else 'GATE FAILURE'} -> {out['next_step']}")


if __name__ == "__main__":
    main()
