"""RQ1 diagnostic (spec §6.5) — run first, it's cheap and it's a go/no-go gate.

Spearman rank correlation between:
  - the correlation-based ranking  (|s_j|)
  - the ablation-magnitude ranking (|Delta_j|)

over a feature sample that is NOT pre-filtered by correlation (spec pitfall §9.5).

Low rank correlation  -> the two criteria are genuinely different, H2/H3 have
                          something to test. Also surfaces H1 case-study features:
                          high |s_j|, low |Delta_j| = 'correlated but not mechanistic'.
High rank correlation -> H2/H3 have little to test; report honestly, don't bury.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rq1_diagnostic(screen_result: dict, ablation_result: dict, top_frac: float = 0.25) -> dict:
    feats = ablation_result["feature_indices"]
    absd = ablation_result["abs_delta"]
    abss = np.abs(screen_result["pearson"][feats])
    eligible = screen_result["eligible"][feats]

    fj, aj, sj = feats[eligible], absd[eligible], abss[eligible]
    if len(fj) < 5:
        return dict(n=len(fj), note="too few eligible features for RQ1")

    rho = stats.spearmanr(sj, aj).statistic
    tau = stats.kendalltau(sj, aj).statistic
    pear = np.corrcoef(sj, aj)[0, 1]

    # top-set overlap: do the two rankings pick the same features?
    n_top = max(1, int(round(top_frac * len(fj))))
    top_corr = set(fj[np.argsort(-sj)[:n_top]].tolist())
    top_abl = set(fj[np.argsort(-aj)[:n_top]].tolist())
    jaccard = len(top_corr & top_abl) / len(top_corr | top_abl)

    # H1 case studies: high correlation rank, low ablation rank
    corr_rank = np.argsort(np.argsort(-sj))       # 0 = most correlated
    abl_rank = np.argsort(np.argsort(-aj))        # 0 = largest |Delta|
    divergence = abl_rank - corr_rank             # large positive = correlated but not mechanistic
    worst = np.argsort(-divergence)[:10]
    spurious_candidates = [
        dict(feature=int(fj[i]), abs_corr=float(sj[i]), abs_delta=float(aj[i]),
             corr_rank=int(corr_rank[i]), abl_rank=int(abl_rank[i]))
        for i in worst
    ]
    # and the converse: validated (large |Delta|) that also correlate
    best = np.argsort(divergence)[:10]
    validated_candidates = [
        dict(feature=int(fj[i]), abs_corr=float(sj[i]), abs_delta=float(aj[i]),
             corr_rank=int(corr_rank[i]), abl_rank=int(abl_rank[i]))
        for i in best if corr_rank[i] < len(fj) // 2
    ]

    return dict(
        n=int(len(fj)),
        spearman_rho=float(rho),
        kendall_tau=float(tau),
        pearson=float(pear),
        top_frac=top_frac,
        top_set_jaccard=float(jaccard),
        interpretation=(
            "LOW rank agreement -> correlation and ablation are different criteria; "
            "H2/H3 have something to test"
            if abs(rho) < 0.6 else
            "HIGH rank agreement -> criteria mostly coincide; H2/H3 have limited headroom (report honestly)"
        ),
        h1_spurious_case_studies=spurious_candidates,
        h1_validated_case_studies=validated_candidates,
    )
