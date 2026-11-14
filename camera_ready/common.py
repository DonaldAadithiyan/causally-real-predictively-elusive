"""Shared loading + statistics for the camera-ready experiments (E1-E6).

Artifact resolution for (dataset, seed):
  arrays : runs/camera_ready/arrays/<tag>.npz  if present, else runs/p126_round2/arrays/<tag>.npz
  weights: runs/camera_ready/weights/<tag>_*.pt if present, else runs/p126_round2/weights/
  meta   : seeds 0-2 ETTh from runs/phase3_real (the paper's stored selection),
           electricity/traffic 0-2 from runs/p126_round2/wp6, new seeds from runs/camera_ready/E2/meta
Everything downstream works from these frozen artifacts; no retraining.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
CR = os.path.join(HERE, "runs", "camera_ready")
R2 = os.path.join(HERE, "runs", "p126_round2")
ALPHA = 0.10
N_BOOT = 2000
BLOCK = 96
BOOT_SEED = 0
ETT = ("ETTh1", "ETTh2")


# ------------------------------------------------------------------ provenance
def git_hash():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.check_output(["git", "-C", HERE, "status", "--porcelain", "--",
                                       "camera_ready", "p126", "experiments", "validation"],
                                      text=True).strip()
        return bool(out)
    except Exception:
        return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def provenance(cmd, seeds, datasets, t0, device="cpu"):
    fps = {}
    for ds in datasets:
        p = os.path.join(HERE, "data", "raw", f"{ds}.csv")
        if os.path.exists(p):
            fps[ds] = dict(file=f"data/raw/{ds}.csv", sha256=sha256(p))
    copy = {"ETTh1": "HTTP ETDataset mirror (edge fp c6d1709fd0693b1a)",
            "ETTh2": "HTTP ETDataset mirror (edge fp c022db5521d23684)",
            "electricity": "Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical",
            "traffic": "Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical"}
    lines = ["## Provenance", "",
             f"- Git: `{git_hash()}`{' (+uncommitted changes)' if git_dirty() else ''}",
             f"- Command: `{cmd}`",
             f"- Seeds: {seeds}",
             f"- Datasets / copy used: " + "; ".join(f"{d}: {copy.get(d, '?')}" for d in datasets),
             "- Raw-file SHA-256: " + "; ".join(f"{d} `{v['sha256']}`" for d, v in fps.items()),
             f"- Hardware: {platform.machine()} {platform.platform()}, no CUDA GPU; analysis on {device} "
             f"(training device per run in E2/meta)",
             f"- Wall-clock: {time.time() - t0:.0f} s",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ artifacts
ARR_OVERRIDE = os.environ.get("CR_ARR_DIR")      # E6: TimeMixer arrays + meta live here


def arr_path(tag):
    if ARR_OVERRIDE:
        return os.path.join(ARR_OVERRIDE, f"{tag}.npz")
    p = os.path.join(CR, "arrays", f"{tag}.npz")
    return p if os.path.exists(p) else os.path.join(R2, "arrays", f"{tag}.npz")


def weights_dir(tag):
    if os.path.exists(os.path.join(CR, "weights", f"{tag}_sae.pt")):
        return os.path.join(CR, "weights")
    return os.path.join(R2, "weights")


def meta(ds, seed):
    tag = f"{ds}_seed{seed}"
    if ARR_OVERRIDE:
        return json.load(open(os.path.join(ARR_OVERRIDE, f"{tag}_meta.json")))
    p = os.path.join(CR, "E2", "meta", f"{tag}_meta.json")
    if os.path.exists(p) and not (ds in ETT and seed <= 2):
        return json.load(open(p))
    if ds in ETT:
        import pandas as pd
        a = pd.read_csv(os.path.join(HERE, "runs", "phase3_real", "AGG_phase3.csv"))
        r = a[(a.dataset == ds) & (a.seed == seed)].iloc[0]
        sel = json.load(open(os.path.join(HERE, "runs", "phase3_real", tag, "mvdf_selection.json")))
        return dict(tag=tag, val_mse=float(r.forecaster_val_mse), sae_fvu=float(r.sae_fvu),
                    mvdf=sel["MVDF_v2"], correlated=sel["correlated"],
                    enrichment_typeB=sel["enrichment_v2"]["enrichment_ratio"], device="cpu")
    return json.load(open(os.path.join(R2, "wp6", f"{tag}_meta.json")))


def candidates_path(ds, seed):
    tag = f"{ds}_seed{seed}"
    if ds in ETT and seed <= 2:
        return os.path.join(HERE, "runs", "phase3_real", tag, "candidates.csv")
    p = os.path.join(CR, "E2", "meta", f"{tag}_candidates.csv")
    return p if os.path.exists(p) else os.path.join(R2, "wp6", f"{tag}_candidates.csv")


def available_seeds(ds):
    out = []
    for s in range(20):
        if os.path.exists(arr_path(f"{ds}_seed{s}")):
            out.append(s)
    return out


def profile_of(abs_err):
    p = abs_err.mean(0)
    return p / (p.mean() + 1e-12)


def score_with(abs_err, prof):
    return (abs_err / prof[None, :]).max(1)


class Run:
    """All test windows of one (dataset, seed). Window index t = origin hour offset
    (stride 1). S = E_select-profile score (the dose-response profile)."""

    def __init__(self, ds, seed, load_X=True):
        tag = f"{ds}_seed{seed}"
        self.ds, self.seed, self.tag = ds, seed, tag
        a = np.load(arr_path(tag))
        for k in a.files:
            if k == "X" and not load_X:
                continue
            setattr(self, k, a[k])
        self.L, self.H = int(a["L"]), int(a["H"])
        self.tv = int(a["target_var"])
        self.meta = meta(ds, seed)
        mv = self.meta["stored_mvdf"] if self.meta.get("mvdf_used") == "stored" else self.meta["mvdf"]
        self.mvdf = np.asarray(mv, int)
        self.correlated = np.asarray(self.meta.get("correlated", []), int)
        self.abs_err = np.abs(self.y - self.yhat).astype(np.float64)
        self.idx_sel = np.where(self.m_select)[0]
        self.idx_id = np.where(self.m_id)[0]
        self.idx_sh = np.where(self.m_shift)[0]
        self.prof_sel = profile_of(self.abs_err[self.idx_sel])
        self.S = score_with(self.abs_err, self.prof_sel)
        assert np.all(np.diff(self.end_idx) == 1), "stride must be 1"

    def z_cols(self, cols):
        cols = np.asarray(cols, int)
        out = np.zeros((len(self.z_idx), len(cols)), np.float64)
        for j, c in enumerate(cols):
            out[:, j] = (self.z_val * (self.z_idx == c)).sum(1)
        return out

    def z_dense(self, rows=None):
        zi = self.z_idx if rows is None else self.z_idx[rows]
        zv = self.z_val if rows is None else self.z_val[rows]
        d = int(self.d_sae)
        out = np.zeros((len(zi), d), np.float32)
        r = np.repeat(np.arange(len(zi)), zi.shape[1])
        c = zi.reshape(-1).astype(int)
        v = zv.reshape(-1)
        ok = c >= 0
        out[r[ok], c[ok]] = v[ok]
        return out

    def S_local(self, idx):
        ae = self.abs_err[idx]
        return score_with(ae, profile_of(ae))


def phase3_split(run):
    n = len(run.idx_id)
    cpt = int(0.6 * n)
    return run.idx_id[:cpt], run.idx_id[cpt:]


def conformal_split(run):
    n = len(run.idx_id)
    c1, c2 = int(0.5 * n), int(0.8 * n)
    return run.idx_id[:c1], run.idx_id[c1:c2], run.idx_id[c2:]


# ------------------------------------------------------------------ statistics
def block_idx(n, block=BLOCK, n_boot=N_BOOT, seed=BOOT_SEED):
    """Moving-block bootstrap index sets, shape (n_boot, n)."""
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]


def ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def r2_vs(y, pred, ref_mean):
    ss = np.sum((y - ref_mean) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else np.nan


def spearman(a, b):
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def fast_spearman_rows(P, Y):
    """Row-wise Spearman between P[i] and Y[i] (2-D arrays), via ranks."""
    rp = stats.rankdata(P, axis=1)
    ry = stats.rankdata(Y, axis=1)
    rp -= rp.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rp ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rp * ry).sum(1) / den, 0.0)


def boot_r2_rho(y, pred, ref_mean, block=BLOCK, n_boot=N_BOOT, seed=BOOT_SEED):
    """Point + block-bootstrap CI for R2 (vs fixed ref_mean) and Spearman rho."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    idx = block_idx(len(y), block, n_boot, seed)
    Y, P = y[idx], pred[idx]
    ss = ((Y - ref_mean) ** 2).sum(1)
    r2b = 1 - ((Y - P) ** 2).sum(1) / np.where(ss > 0, ss, np.nan)
    rhob = fast_spearman_rows(P, Y) if np.std(pred) > 1e-15 else np.zeros(n_boot)
    return dict(R2=r2_vs(y, pred, ref_mean), R2_lo=ci(r2b)[0], R2_hi=ci(r2b)[1],
                rho=spearman(pred, y), rho_lo=ci(rhob)[0], rho_hi=ci(rhob)[1],
                _r2b=r2b, _rhob=rhob, _idx=idx)


def boot_mean(d, block=BLOCK, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, float)
    idx = block_idx(len(d), block, n_boot, seed)
    b = d[idx].mean(1)
    lo, hi = ci(b)
    return float(d.mean()), lo, hi, b


def two_level(per_seed_boot, seed=BOOT_SEED, n_boot=N_BOOT):
    """Pooled CI across seeds: resample seeds with replacement, then for each picked seed
    take one of its within-seed block-bootstrap replicates; statistic = mean over seeds."""
    rng = np.random.default_rng(seed + 1)
    arrs = [np.asarray(b, float)[np.isfinite(b)] for b in per_seed_boot]
    arrs = [a for a in arrs if len(a)]
    if not arrs:
        return np.nan, np.nan
    k = len(arrs)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        out[i] = np.mean([arrs[p][rng.integers(0, len(arrs[p]))] for p in pick])
    return ci(out)


def sign_test(successes, n):
    """One-sided binomial P(X >= successes | n, 0.5)."""
    return float(stats.binom.sf(successes - 1, n, 0.5)) if n else np.nan


def pass_threshold(n):
    """Pre-registered seed-count rule: 3/3 for N=3, 4/5 for N=5, else ceil(0.8 N)."""
    if n <= 3:
        return n
    return int(np.ceil(0.8 * n))


def conf_q(scores, level=1 - ALPHA):
    """Project conformal quantile (calibration/aci.empirical_quantile): rank ceil((n+1)level)
    clipped to n."""
    from calibration.aci import empirical_quantile
    return empirical_quantile(np.asarray(scores, float), level)


def summ(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(median=np.nan, min=np.nan, max=np.nan)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def md_table(df, fmt="{:.3f}"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(fmt.format(v) if np.isfinite(v) else "–")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ enrichment with uncertainty
def enrichment_stats(selected, ftab, pool, n_boot=N_BOOT, seed=BOOT_SEED):
    """validation.difficulty_specificity.enrichment + (added) a bootstrap CI and a one-sided
    hypergeometric p. CI: resample the selected features' B-membership and the pool's
    B-membership independently (2000x), ratio of fractions, percentile 95%."""
    from validation.difficulty_specificity import enrichment
    e = enrichment(np.asarray(selected, int), ftab, np.asarray(pool, int))
    B = set(ftab["cells"]["B"])
    selected = sorted(int(j) for j in selected)        # CI must depend on the set, not its order
    pool = sorted(int(j) for j in pool)
    selb = np.array([int(j) in B for j in selected if int(j) in set(pool)], float)
    poolb = np.array([int(j) in B for j in pool], float)
    rng = np.random.default_rng(seed)
    num = selb[rng.integers(0, len(selb), (n_boot, len(selb)))].mean(1)
    den = poolb[rng.integers(0, len(poolb), (n_boot, len(poolb)))].mean(1)
    ratio = num / np.where(den > 0, den, np.nan)
    lo, hi = ci(ratio)
    k, n, K, N = int(selb.sum()), len(selb), int(poolb.sum()), len(poolb)
    p = float(stats.hypergeom.sf(k - 1, N, K, n))
    return dict(enrichment=float(e.get("enrichment_ratio", np.nan)), enr_lo=lo, enr_hi=hi,
                enr_p_hypergeom=p, n_sel_B=k, n_sel=n, pool_B=K, pool_n=N,
                enr_ci_excludes_1=bool(lo > 1))


def jaccard(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / len(a | b) if (a | b) else np.nan
