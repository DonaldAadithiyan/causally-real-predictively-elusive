"""E3 hard leakage rule: no ex-ante feature for row t may use a target value with
timestamp > end_idx[t].

Two checks:
  1. Analytic: every lagged feature's latest target time <= end_idx[t].
  2. Empirical: overwrite every realised target y[t', h] whose time end_idx[t'] + 1 + h
     exceeds end_idx[t] with noise, rebuild S (fixed profile) and all F1/F1p/F2/F3 rows for
     t, and assert they are bit-identical to the clean features.
Run: python -m experiments.camera_ready.test_leakage [ds seed]   (exit 1 on failure)
"""
from __future__ import annotations

import sys

import numpy as np

from experiments.camera_ready.common import Run, score_with
from experiments.camera_ready.features import build_exante, target_times_used, MIN_T


def check(run, rows, rng):
    clean = build_exante(run, rows)
    for t in rows:
        assert target_times_used(run, t).max() <= run.end_idx[t], f"analytic leak at row {t}"
        tt = run.end_idx[:, None] + 1 + np.arange(run.H)[None, :]   # target time of y[t', h]
        y2 = run.y.copy()                                  # same dtype as Run (float32)
        fut = tt > run.end_idx[t]
        y2[fut] = rng.normal(0, 50, fut.sum()).astype(y2.dtype)
        ae2 = np.abs(y2 - run.yhat).astype(np.float64)     # same arithmetic as Run.abs_err
        S2 = score_with(ae2, run.prof_sel)
        pert = build_exante(run, np.array([t]), abs_err=ae2, S=S2)
        for k in clean:
            a = clean[k][0][list(rows).index(t)]
            b = pert[k][0][0]
            if not np.array_equal(a, b):
                bad = [clean[k][1][i] for i in np.where(a != b)[0]]
                raise AssertionError(f"LEAK {run.tag} row {t} set {k}: {bad}")
        # sanity: the perturbation does change the TARGET S_t (so the test has power)
        assert S2[t] != run.S[t], "perturbation did not touch the target; test has no power"
    return True


def main():
    cells = [(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) > 2 else [("ETTh1", 0), ("traffic", 0)]
    rng = np.random.default_rng(0)
    for ds, s in cells:
        run = Run(ds, s)
        cand = np.concatenate([run.idx_id[:3], run.idx_id[-2:], run.idx_sh[:2], run.idx_sh[-2:],
                               rng.choice(run.idx_id, 5, replace=False)])
        cand = np.unique(cand[cand >= MIN_T])
        check(run, cand, rng)
        print(f"[leakage test] {run.tag}: PASS on {len(cand)} rows", flush=True)


if __name__ == "__main__":
    main()
