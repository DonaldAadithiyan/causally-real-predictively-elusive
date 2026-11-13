"""V2 spec §4.7 (reuses v1 §4.6) — cross-environment invariance.

The measurement Run 5 skipped. For a feature set, we ask: does the conditional
relationship between the feature signal and the nonconformity score STAY THE SAME
across disjoint deployment environments? A merely-correlated feature's relationship
is a property of the calibration-time distribution and drifts; a mechanistic one is
stable.

  conditional_difficulty_curve(activation, score, n_bins) -> Curve
      bin the feature signal into n_bins quantile bins; Curve[k] = mean S_t in bin k.

  invariance_score({env: Curve}) -> {
      I            : mean pairwise L2 distance between per-env curves (LOWER = more invariant)
      spearman_stability : mean pairwise Spearman rho of the curves (HIGHER = more stable)
      conditional_error_drift : max_k |Curve_env1[k] - Curve_env2[k]| averaged over pairs
  }

HARD LEAKAGE RULE (v1 §8.5, spec §4.7, §10): environment tags used here must be
disjoint from the tag used to SELECT the feature set. `assert_no_leakage` raises
if they overlap; call it before any invariance number is trusted, and there is a
deliberately-overlapping test in test_invariance_leakage().
"""
from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    pass


def assert_no_leakage(selection_env_tag: str, eval_env_tags: list[str]) -> None:
    """Raise if the feature-selection environment tag appears among the evaluation
    environment tags (spec §4.7 hard requirement)."""
    if selection_env_tag in set(eval_env_tags):
        raise LeakageError(
            f"LEAKAGE: feature selection used env {selection_env_tag!r}, which also "
            f"appears in the invariance eval envs {eval_env_tags}. E_select must be "
            f"strictly disjoint from every E_eval_* (spec §4.7, v1 §8.5)."
        )


def conditional_difficulty_curve(activation: np.ndarray, score: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Curve[k] = mean S_t among windows whose feature signal falls in quantile bin k.
    Returns (n_bins,) with NaN for empty bins."""
    a = np.asarray(activation, float); s = np.asarray(score, float)
    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    b = np.digitize(a, edges)
    out = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = b == k
        if m.sum() >= 5:
            out[k] = s[m].mean()
    return out


MIN_SHARED_BINS = 3   # a scale-normalized 2-point curve is always [+1,-1] -> trivial I~0.
                      # Require >=3 shared populated bins for a pair to count; else the
                      # metric is degenerate (this exact edge case produced a spurious
                      # I(MVDF_v2)=0 on ETTh1 seed 0 that inflated the seed sd).


def invariance_score(curves_by_env: dict[str, np.ndarray]) -> dict:
    from scipy import stats
    envs = [e for e, c in curves_by_env.items() if np.isfinite(c).sum() >= MIN_SHARED_BINS]
    if len(envs) < 2:
        return {"I": float("nan"), "spearman_stability": float("nan"),
                "conditional_error_drift": float("nan"), "n_env_pairs": 0, "envs": envs,
                "degenerate": True,
                "note": f"need >=2 environments with >={MIN_SHARED_BINS} populated bins; "
                        f"the feature signal barely varies so its conditional curve is "
                        f"degenerate (I is not meaningful)."}
    l2s, rhos, drifts = [], [], []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            ci, cj = curves_by_env[envs[i]], curves_by_env[envs[j]]
            ok = np.isfinite(ci) & np.isfinite(cj)
            if ok.sum() < MIN_SHARED_BINS:
                continue
            ni = (ci[ok] - np.nanmean(ci[ok])) / (np.nanstd(ci[ok]) + 1e-9)
            nj = (cj[ok] - np.nanmean(cj[ok])) / (np.nanstd(cj[ok]) + 1e-9)
            l2s.append(float(np.sqrt(np.mean((ni - nj) ** 2))))
            rhos.append(float(stats.spearmanr(ci[ok], cj[ok]).statistic))
            drifts.append(float(np.max(np.abs(ci[ok] - cj[ok]))))
    if not l2s:
        return {"I": float("nan"), "spearman_stability": float("nan"),
                "conditional_error_drift": float("nan"), "n_env_pairs": 0, "envs": envs,
                "degenerate": True,
                "note": f"no env pair shares >={MIN_SHARED_BINS} populated bins."}
    return dict(
        I=float(np.mean(l2s)),
        spearman_stability=float(np.nanmean(rhos)),
        conditional_error_drift=float(np.mean(drifts)),
        n_env_pairs=len(l2s), envs=envs, degenerate=False,
    )


def compute_for_condition(
    signal_by_env: dict[str, np.ndarray],
    score_by_env: dict[str, np.ndarray],
    selection_env_tag: str,
    n_bins: int = 5,
) -> dict:
    """signal_by_env / score_by_env keyed by E_eval_* tags. Runs the leakage check,
    builds per-env conditional curves, returns invariance_score(...)."""
    eval_tags = list(signal_by_env.keys())
    assert_no_leakage(selection_env_tag, eval_tags)
    curves = {e: conditional_difficulty_curve(signal_by_env[e], score_by_env[e], n_bins)
              for e in eval_tags}
    out = invariance_score(curves)
    out["curves"] = {e: c.tolist() for e, c in curves.items()}
    return out


def test_invariance_leakage() -> bool:
    """Deliberately-overlapping case MUST raise (spec §10 acceptance)."""
    try:
        assert_no_leakage("E_select", ["E_eval_ID", "E_select", "E_eval_shift"])
    except LeakageError:
        return True
    return False
