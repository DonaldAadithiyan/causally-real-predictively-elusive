"""PRE-REGISTERED distribution-shift windows (spec §6.3, pitfall §9.6).

Fixed here, in code, BEFORE the main MVDF-vs-correlated comparison is run. Do not
edit a window after seeing results.

The standard LTSF test split is the IN-DISTRIBUTION evaluation. The SHIFT evaluation
is the final contiguous fraction of the test split (`tail_frac`), which is the block
furthest in calendar time from the train+val calibration window and therefore where
temporal-dependence similarity to calibration is weakest and a seasonal/regime
difference is most likely. `tail_frac` (not an absolute step count) is used so the
definition is robust to the exact series length each data adapter produces
(the attached Kaggle bundle is hourly-resampled and shorter than the canonical CSVs).

`describe_shift` runs a KS two-sample test at run time; a window with KS p >= 0.05 is
reported as "did not verify as a shift", never silently swapped.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass(frozen=True)
class ShiftWindow:
    dataset: str
    description: str
    rationale: str
    tail_frac: float = 0.20          # final 20% of the test split
    require_ks_p_below: float = 0.05


_COMMON_RATIONALE = (
    "Final {pct:.0f}% of the test split: the block furthest in calendar time from the "
    "train+val calibration window, so autocorrelation-driven similarity to calibration "
    "is weakest and a seasonal/regime difference is most likely. Pre-registered before "
    "results were seen; KS-checked at run time."
)


def _mk(name, extra):
    return ShiftWindow(
        dataset=name,
        description=f"Final 20% of the {name} test split.",
        rationale=_COMMON_RATIONALE.format(pct=20) + " " + extra,
        tail_frac=0.20,
    )


REGISTERED = {
    "ETTh1": _mk("ETTh1", "ETT has a strong annual cycle; the tail is a different season from calibration."),
    "ETTh2": _mk("ETTh2", "Same construction and reasoning as ETTh1."),
    "ETTm1": _mk("ETTm1", "Same as ETTh1 at finer (15-min / hourly-resampled) resolution."),
    "ETTm2": _mk("ETTm2", "Same as ETTh1 at finer resolution."),
    "weather": _mk("weather", "Jena/Beutenberg meteorology has a strong seasonal component."),
    "traffic": _mk("traffic", "SF Bay Area road occupancy has weekly regime structure and holiday anomalies."),
    "electricity": _mk("electricity", "Client consumption has weekday/weekend and seasonal regimes."),
    "solar": _mk("solar", "Solar power has a strong seasonal envelope over the one-year record."),
    "exchange": _mk("exchange", "Exchange rates drift with macro regimes over the multi-decade record."),
    "QPS": _mk("QPS", "Query load has weekly and release-driven regime structure."),
}


def split_test_windows(end_idx: np.ndarray, test_len: int, tail_frac: float):
    """Boolean masks over sliding-window end indices into the test series:
      in_distribution : all windows
      shift           : windows whose end index is within the final `tail_frac`
    """
    in_dist = np.ones(len(end_idx), dtype=bool)
    cut = int(round(test_len * (1.0 - tail_frac)))
    shift = end_idx >= cut
    return in_dist, shift


# ---- V2 spec §9: three-way chronological carve-out of the TEST split ----------------
#   E_select    : first third   — rank/threshold features, fix MVDF_v2_set. Nothing
#                 downstream may re-touch this block (hard leakage rule, spec §4.7).
#   E_eval_ID   : middle third  — in-distribution invariance + calibration/eval.
#   E_eval_shift: final third   — the shift block (aligns with, and widens, Run 5's
#                 final-20% window).
# Environment tags are threaded explicitly through the pipeline; validation/invariance.py
# asserts E_select never appears among the E_eval_* tags.

ENV_TAGS = ("E_select", "E_eval_ID", "E_eval_shift")


def three_way_env_masks(end_idx: np.ndarray, test_len: int,
                        fracs=(1 / 3, 1 / 3, 1 / 3)) -> dict:
    """Return {tag: bool mask over sliding-window end indices} for the three
    chronological blocks of the test split. Windows are assigned by the block their
    END index falls in, so blocks are disjoint at the window level."""
    assert abs(sum(fracs) - 1.0) < 1e-6
    c1 = int(round(test_len * fracs[0]))
    c2 = int(round(test_len * (fracs[0] + fracs[1])))
    masks = {
        "E_select":     end_idx < c1,
        "E_eval_ID":    (end_idx >= c1) & (end_idx < c2),
        "E_eval_shift": end_idx >= c2,
    }
    return masks


def describe_env_split(raw_test_target: np.ndarray, fracs=(1 / 3, 1 / 3, 1 / 3)) -> dict:
    """KS tests: E_eval_ID vs E_select (should be similar), E_eval_shift vs E_select
    (should differ — that is the pre-registered shift)."""
    n = len(raw_test_target)
    c1 = int(round(n * fracs[0])); c2 = int(round(n * (fracs[0] + fracs[1])))
    sel, mid, shf = raw_test_target[:c1], raw_test_target[c1:c2], raw_test_target[c2:]
    ks_mid = stats.ks_2samp(mid, sel)
    ks_shf = stats.ks_2samp(shf, sel)
    return dict(
        n_select=len(sel), n_eval_id=len(mid), n_eval_shift=len(shf),
        ks_evalID_vs_select=dict(stat=float(ks_mid.statistic), p=float(ks_mid.pvalue)),
        ks_evalShift_vs_select=dict(stat=float(ks_shf.statistic), p=float(ks_shf.pvalue)),
        shift_verified=bool(ks_shf.pvalue < 0.05),
    )


def describe_shift(raw_test_target: np.ndarray, tail_frac: float, cal_target: np.ndarray) -> dict:
    n_tail = max(1, int(round(len(raw_test_target) * tail_frac)))
    tail = raw_test_target[-n_tail:]
    ks = stats.ks_2samp(tail, cal_target)
    return dict(
        n_tail=int(n_tail),
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        tail_mean=float(np.mean(tail)),
        cal_mean=float(np.mean(cal_target)),
        tail_std=float(np.std(tail)),
        cal_std=float(np.std(cal_target)),
        is_shift=bool(ks.pvalue < 0.05),
    )
