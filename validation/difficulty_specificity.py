"""V2 spec §4.4, §10 — causal specificity R_j and the feature-type table.

  R_j = U_j - E[U_j | P_j]

residualize the interventional uncertainty-effect (U_j) against the interventional
prediction-effect (P_j), via a fitted regression over the >=100-candidate pool.
BOTH inputs MUST be interventional (from dose_response.sweep_pool). Passing the
observational C_j here is a contract violation (spec §10, §15) and is guarded.

High R_j: intervening on the feature moves the nonconformity score by MORE than its
general prediction-importance predicts -> the feature's causal role is specific to
DIFFICULTY, not prediction content. These are the type-B features MVDF_v2 selects.

Feature-type table (median split on P_j x U_j):
  A: high P, low U   -> prediction/content feature
  B: low P,  high U  -> uncertainty-specific  (the target type)
  C: high P, high U  -> mixed
  D: low P,  low U   -> irrelevant to both
"""
from __future__ import annotations

import numpy as np


def residualize(p_j: np.ndarray, u_j: np.ndarray, method: str = "isotonic",
                provenance_p: str | None = None, provenance_u: str | None = None) -> np.ndarray:
    """R_j = U_j - E[U_j | P_j]. `method` in {'isotonic', 'local_linear', 'linear'}.
    Both provenance tags, if given, must contain 'interventional' (spec §10 guard)."""
    for name, prov in (("p_j", provenance_p), ("u_j", provenance_u)):
        if prov is not None and "interventional" not in prov:
            raise ValueError(
                f"difficulty_specificity.residualize: {name} provenance {prov!r} is not "
                f"interventional. R_j requires BOTH terms from the dose-response sweep "
                f"(spec §4.4, §15). Observational C_j is NOT a valid input here."
            )
    p = np.asarray(p_j, float); u = np.asarray(u_j, float)
    ok = np.isfinite(p) & np.isfinite(u)
    R = np.full_like(p, np.nan)
    if ok.sum() < 10:
        return R
    pp, uu = p[ok], u[ok]
    order = np.argsort(pp)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        # fit E[U|P] as a monotone function of P in whichever direction fits better
        best_pred, best_sse = None, np.inf
        for inc in (True, False):
            ir = IsotonicRegression(increasing=inc, out_of_bounds="clip")
            pred = ir.fit(pp, uu).predict(pp)
            sse = np.sum((uu - pred) ** 2)
            if sse < best_sse:
                best_sse, best_pred = sse, pred
        resid = uu - best_pred
    elif method == "local_linear":
        # LOWESS
        try:
            import statsmodels.api as sm
            sm_pred = sm.nonparametric.lowess(uu, pp, frac=0.4, return_sorted=False)
            resid = uu - sm_pred
        except Exception:
            method = "linear"
    if method == "linear":
        A = np.vstack([pp, np.ones(len(pp))]).T
        coef = np.linalg.lstsq(A, uu, rcond=None)[0]
        resid = uu - A @ coef
    R[ok] = resid
    _ = order  # kept for readability of the isotonic path
    return R


def feature_type_table(p_j: np.ndarray, u_j: np.ndarray,
                       feature_indices: np.ndarray | None = None) -> dict:
    """Median split -> feature identifiers per cell. If `feature_indices` is given,
    cells hold those identifiers (feature IDs); otherwise they hold local positions.
    Returns {'A':[...], 'B':[...], ...} plus counts and the median thresholds."""
    p = np.asarray(p_j, float); u = np.asarray(u_j, float)
    ids = np.asarray(feature_indices) if feature_indices is not None else np.arange(len(p))
    ok = np.isfinite(p) & np.isfinite(u)
    pm, um = np.median(np.abs(p[ok])), np.median(np.abs(u[ok]))
    ap, au = np.abs(p), np.abs(u)
    cells = {"A": [], "B": [], "C": [], "D": []}
    for i in np.where(ok)[0]:
        hi_p, hi_u = ap[i] >= pm, au[i] >= um
        key = "C" if (hi_p and hi_u) else "A" if hi_p else "B" if hi_u else "D"
        cells[key].append(int(ids[i]))
    n = int(ok.sum())
    return dict(
        cells=cells,
        counts={k: len(v) for k, v in cells.items()},
        fractions={k: len(v) / max(n, 1) for k, v in cells.items()},
        median_abs_P=float(pm), median_abs_U=float(um), n=n,
        interpretation="type-B mass (low |P_j|, high |U_j|) = uncertainty-specific features; "
                       "MVDF_v2 should be enriched for these vs correlated_set / MVDF_v1 (H1c)",
    )


def enrichment(selected_idx: np.ndarray, type_table: dict, all_pool_idx: np.ndarray) -> dict:
    """What fraction of `selected_idx` are type B, vs the pool base rate (H1c)."""
    B = set(type_table["cells"]["B"])
    sel = set(int(i) for i in selected_idx)
    pool = set(int(i) for i in all_pool_idx)
    sel_in_pool = sel & pool
    if not sel_in_pool:
        return {"note": "no selected features in the characterized pool"}
    sel_B_frac = len(sel_in_pool & B) / len(sel_in_pool)
    base_B_frac = len(B) / max(len(pool), 1)
    return dict(selected_type_B_fraction=sel_B_frac, pool_type_B_fraction=base_B_frac,
                enrichment_ratio=sel_B_frac / (base_B_frac + 1e-12),
                n_selected_in_pool=len(sel_in_pool))


def select_mvdf_v2(dose: dict, u_threshold: float, k: int,
                   residualize_method: str = "isotonic") -> dict:
    """MVDF-v2 gate (spec §4.6): a feature qualifies if
      (1) |U_j| >= u_threshold  AND its D_j^unc(lambda) curve is ~monotone, AND
      (2) it ranks top-k by R_j.
    Returns {selected, R_j, gate1_pass, ...}."""
    P, U = dose["P_j"], dose["U_j"]
    feats = dose["feature_indices"]
    R = residualize(P, U, method=residualize_method,
                    provenance_p="interventional_dose_response",
                    provenance_u="interventional_dose_response")

    # gate 1: |U_j| threshold + monotone unc curve
    mono_ok = np.array([_curve_monotone(dose["unc_curves"][int(j)]) for j in feats])
    gate1 = (np.abs(U) >= u_threshold) & mono_ok & np.isfinite(R)

    cand = np.where(gate1)[0]
    cand = cand[np.argsort(-R[cand])]        # gate 2: rank by R_j
    selected = feats[cand[:k]]
    return dict(selected=selected, R_j=R, gate1_pass=gate1,
                n_gate1=int(gate1.sum()), u_threshold=u_threshold,
                residualize_method=residualize_method)


def _curve_monotone(curve: dict, tol: float = 0.0) -> bool:
    v = np.array([curve[key] for key in sorted(curve)], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return False
    d = np.diff(v)
    return bool(np.all(d >= -1e-9 - tol) or np.all(d <= 1e-9 + tol))
