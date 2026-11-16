# CQR + PID-conformal vs `naive_matched` — the literature baseline

**Run:** `experiments/baseline_comparison.py`, full fidelity, 2026-09-05. Outputs
`runs/baseline_comparison/{AGG_baselines.csv, AGG_baselines_tests.csv}`.

**Scope: ETTh1 + ETTh2, seeds 0/1/2 (6 cells).** electricity and traffic could NOT be
run: their raw source files (`LD2011_2014.csv` ~700 MB, `PeMS.csv` ~102 MB) were only
ever Kaggle-mounted for the earlier phases and are not in the local checkout; the
LTSF-standard preprocessed versions are different data (different preprocessing →
different fingerprint → a non-comparable forecaster/SAE vs every prior H6 / Task G /
H7-A result), so substituting them would break the same-data comparison. ETTh1/ETTh2 are
also exactly the two datasets H7-A ran on, and the two with clean coverage throughout
this project — the conclusion below is clear on them.

## What this answers

Not "does our method beat the literature" — no SAE-conditioned method survived to make
that claim (Task G and H7-A both falsified). It answers whether **the literature itself
beats `naive_matched`**: if standard adaptive-CP methods also lose to a constant-width
band under this forecasting-under-shift regime, the negative result is general to
adaptive conformal conditioning, not specific to SAE-conditioning.

## Method (each baseline in its own standard form)

- **`naive_matched`** — reused unchanged (constant-width split-conformal, 80 %-budget
  quantile; already validated in Task G's budget audit).
- **CQR** (Romano, Patterson & Candès 2019) — a genuine quantile-regression iTransformer
  (`baselines/quantile_itransformer.py`: same trunk as the point forecaster, two linear
  heads, pinball loss at α/2 and 1−α/2, trained fresh per (dataset, seed) on the same
  E_train/E_val split) + the standard split-conformal correction, with the conformity
  score scale-normalized by the calibration `step_profile` (standard heteroscedastic-CQR
  practice, same normalization this project's own `S_t` uses). No SAE, no MVDF.
- **PID-conformal** (Angelopoulos, Candès & Tibshirani 2023) — the P+I online update
  (no Scorecaster) on the **point** forecaster's raw residual stream. `η` set to the
  project's `gamma_global` (0.03), matched to `naive_matched`'s adaptation speed so the
  PID-vs-single-gain comparison isn't confounded by tuning.

**Known CQR caveat (diagnosed, reported not fixed):** the quantile forecaster converges
(val pinball plateaus by ~epoch 5; 60 epochs + patience 10 gives the identical result)
but its raw per-step coverage falls from ~0.94 at horizon step 1 to ~0.60 by mid/late
horizon — a genuine capacity limit of one linear head tracking 96 quantile positions,
not undertraining. The split-conformal step then restores per-step marginal validity
(post-correction per-step coverage 0.97–0.99) but must inflate `q_hat` to cover the
worst steps, so **simultaneous** coverage still lands a few points off nominal. Iterating
on the quantile architecture to close this was declined — it would be tuning a baseline
to look better; the residual gap is itself informative about how hard well-calibrated
multi-horizon quantile regression is with a standard architecture.

## Result

### Means over ETTh1 + ETTh2 (6 cells), α = 0.10

| mode | ID coverage | ID width | shift coverage | shift width |
|---|---|---|---|---|
| **`naive_matched`** | 0.836 | 2.757 | 0.896 | 2.411 |
| `pid_conformal` | 0.830 | 2.749 | 0.897 | 2.405 |
| `cqr` | 0.801 | 2.893 | 0.938 | 2.771 |

### `pid_conformal − naive_matched`, shift env, per cell (coverage-equivalent = |Δcov| ≤ 1.5 pt)

| dataset | seed | Δ shift width | Δ shift coverage | coverage-equiv? | narrower & valid? |
|---|---|---|---|---|---|
| ETTh1 | 0 | **−0.061** | +0.002 | ✅ | ✅ |
| ETTh1 | 1 | **−0.066** | +0.002 | ✅ | ✅ |
| ETTh1 | 2 | **−0.056** | +0.003 | ✅ | ✅ |
| ETTh2 | 0 | +0.045 | −0.001 | ✅ | ✗ (wider) |
| ETTh2 | 1 | +0.066 | +0.001 | ✅ | ✗ (wider) |
| ETTh2 | 2 | +0.043 | 0.000 | ✅ | ✗ (wider) |

### `cqr − naive_matched`, shift env, per cell

| dataset | seed | Δ shift width | shift coverage (CQR) | narrower & valid? |
|---|---|---|---|---|
| ETTh1 | 0 | −0.209 | 0.879 (borderline undercover) | ✗ |
| ETTh1 | 1 | −0.127 | 0.905 | ✗ (ID coverage 0.77) |
| ETTh1 | 2 | −0.151 | 0.889 | ✗ (ID coverage 0.72) |
| ETTh2 | 0 | +0.702 | 0.988 (massive overcover) | ✗ |
| ETTh2 | 1 | +1.084 | 0.985 | ✗ |
| ETTh2 | 2 | +0.866 | 0.982 | ✗ |

### Read

- **PID-conformal splits by dataset.** On **ETTh1** it is genuinely **narrower under
  shift at equivalent valid coverage — 3/3 seeds, significant** (~3 % tighter: shift
  width 1.82–1.88 vs `naive_matched` 1.87–1.94, Δcov ≤ 0.003). On **ETTh2** it is
  **wider** at equivalent coverage — 3/3 seeds. Net over the 6 cells: **−0.005 shift
  width, a wash.** So PID-conformal is *competitive with, and on one of the two datasets
  modestly better than,* `naive_matched` — not uniformly indistinguishable, and not
  beaten either. Both hold ID coverage ~0.83 (the shared short-`E_eval_ID` effect) and
  shift coverage ~0.90.
- **CQR is not competitive on either dataset.** ETTh1: undercovers ID (0.71–0.77) and its
  narrower shift width comes with borderline shift coverage (0.88–0.90). ETTh2: both
  wider *and* massively over-covering (shift 0.98–0.99). Never narrower at equivalent
  valid coverage. (The quantile-forecaster caveat above is the ETTh1 ID undercoverage;
  the ETTh2 over-cover is the same undertrained mid/late-horizon quantiles producing an
  inflated `q_hat`.)

## Interpretation (pre-committed, from the wrap-up spec)

The wrap-up spec pre-committed two readings. The data lands **between** them, and the
honest statement is the more precise one:

> **No evaluated adaptive-CP method — learned difficulty scorer (Task G), discrete regime
> gate (H7-A), CQR, or PID-conformal — is uniformly narrower than a properly-calibrated
> constant-width split-conformal band at equivalent valid coverage on ETTh1/ETTh2 under
> our shift protocol.** The strongest showing is PID-conformal, which ties on aggregate
> and is ~3 % tighter on ETTh1 while ~2 % wider on ETTh2 — a dataset-dependent wash, not
> a win. CQR is worse on both. The SAE-conditioned methods (Task G, H7-A) are therefore
> **not uniquely poor** — they sit in the same band as the standard literature baselines,
> all of which fail to establish a robust improvement over the constant band here.

This makes the paper's downstream claim **general, with a stated ceiling**: on these
data, adaptive conformal conditioning of any evaluated kind buys little-to-nothing over
a constant band, and where it helps at all (PID/ETTh1) it is small and does not
generalize to the sibling dataset. Task G and H7-A are instances of this pattern, not
isolated SAE-specific failures.

**Claims discipline:** say *"on ETTh1/ETTh2 under our shift protocol, no evaluated
adaptive-CP method (learned-scorer family, regime gate, CQR, PID-conformal) is uniformly
narrower than a constant-width split-conformal baseline at equivalent valid coverage;
the best case, PID-conformal, ties on aggregate and shows a small, non-transferring
ETTh1-only advantage."* Do **not** say "adaptive conformal prediction doesn't work" —
the scope is these methods, this regime, two datasets. A dataset with genuinely
predictable heteroscedastic difficulty remains the natural boundary condition;
electricity/traffic under the same protocol are untested here (data-availability, see
below) — a documented limitation, not a silent gap.

## Standing status of the exploitation question — COMPLETE

| route | verdict |
|---|---|
| **H1c** — causal difficulty features exist | **DISCOVERY** (12/12) |
| **H3** — are they shift-invariant? | **MEASUREMENT BOUNDARY** (sparse activation → metric degenerate; not askable here) |
| **Task G / 3a** — continuous learned scorer | **FALSIFIED** (loses to `naive_matched`, budget-matched) |
| **H7-A / 3b** — discrete regime gate | **FALSIFIED** (no regime; `R_Q ≈ 1.0 ± 0.06`) |
| **SLCP / 3c** — similarity-weighted localization | **CONFOUNDED** (temporal autocorrelation; §2 check confirms; not testable here) |
| **CQR, PID-conformal** — literature adaptive-CP baselines | **also do not beat `naive_matched`** (ETTh1/ETTh2) → the failure is general, not SAE-specific |

The exploitation phase is closed. Per the wrap-up spec's stop condition: **done,
regardless of outcome. No further candidate mechanisms.**
