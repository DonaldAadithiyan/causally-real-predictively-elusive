# H7-A — NEGATIVE. MG-CP does not proceed.

**Run:** `experiments/h7a_gate.py`, full fidelity (20-epoch iTransformer + 20k-step SAE),
ETTh1 + ETTh2 × seeds 0/1/2, 2026-09-04. SAE quality FVU **0.0042–0.0095** (inside the
project's H1c trust bar of 0.003–0.013). Outputs: `runs/h7a_gate/{H7A.csv,
H7A_decision.json}`.

## The question

Does high co-activation of the interventionally-validated Type-B (cell-B) feature set
induce a separated empirical tail-error regime — i.e. is
`R_Q = Q_.90(S_t | D_t=1) / Q_.90(S_t | D_t=0)` reliably > 1?

- `D_t = 1` iff `A_t` (fraction of the ~24 cell-B features active in window `t`) is
  ≥ the 90th percentile of `A_t` computed on `E_select` only (leakage-safe). This lands
  at a ~11–32% regime per seed — `A_t` is lumpy-discrete (≤24 features, each ~3% active),
  so a p90 threshold cannot split at exactly 10%. Definition frozen before computing
  `R_Q`; no re-tuning.
- Cell-B set = `feature_type_table.json['cells']['B']`, fresh per seed (24–27 features).
- `S_t` = the frozen nonconformity score `max_h |e_h| / profile_h`.

## Pre-committed stop rule (MG-CP spec §1)

Proceed to MG-CP-1 only if the paired-bootstrap CI on `R_Q` excludes 1 in the `R_Q > 1`
direction, on **E_eval_ID**, on **both datasets**, **all 3 seeds**. `E_eval_shift` is a
secondary, non-gating readout.

## Result — FAILS

**GATE (E_eval_ID):**

| dataset | seed | R_Q | 95% CI | n₁ | n₀ | Q₁ | Q₀ | excl. 1 above? |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 0 | 1.028 | [0.927, 1.156] | 340 | 1626 | 0.901 | 0.877 | ✗ |
| ETTh1 | 1 | 0.912 | [0.862, 1.004] | 550 | 1416 | 0.797 | 0.873 | ✗ |
| ETTh1 | 2 | 1.059 | [0.912, 1.205] | 237 | 1729 | 0.911 | 0.860 | ✗ |
| ETTh2 | 0 | 0.971 | [0.918, 1.019] | 215 | 1751 | 1.644 | 1.694 | ✗ |
| ETTh2 | 1 | 1.045 | [1.009, 1.093] | 626 | 1340 | 1.753 | 1.678 | ✓ |
| ETTh2 | 2 | 1.003 | [0.946, 1.051] | 225 | 1741 | 1.689 | 1.683 | ✗ |

**1 / 6 seed-runs** clears the bar (ETTh2 s1). Rule requires 6 / 6. → **STOP.**

**E_eval_shift (secondary, non-gating):** same picture — R_Q ∈ [0.94, 1.06], only
ETTh2 s1 separates (1.063, CI [1.022, 1.090]). No shift-side regime.

## Why this is a clean negative, not an underpowered one

- **Well-powered.** n₁ ranges 215–626 (not the handful the spec worried about). The
  absolute tail quantiles Q₁ and Q₀ are near-identical in every cell (largest gap:
  ETTh1 s1, 0.797 vs 0.873, and that one points the *wrong* way). This is not a
  noisy-tail-quantile artefact — it is an absence of separation.
- **No consistent direction.** R_Q straddles 1.0 across the 6 runs (0.91–1.06);
  `P(R_Q > 1)` spans 0.04–0.99. Not a suppressed positive, not a reversal — just noise
  around 1.
- **SAE quality is not the story.** The `--fast` smoke (FVU ~0.028) had suggested a mild
  *reversal* (R_Q ~0.94–0.98). The trust-bar SAE (FVU ~0.005) shows *even less* signal,
  scattered around 1.0. The one outcome that would have reopened this — full fidelity
  reversing the smoke pattern in a way that implicated smoke SAE quality — did not occur.
- **Two prior `D_t` operationalizations agreed.** `D_t = ≥1 cell-B feature active` (~55%
  regime) gave R_Q ≈ 0.98 at smoke fidelity; the fraction-threshold gives R_Q ≈ 1.0 ± 0.06
  at full fidelity. Three operationalizations, no regime.

## Decision

**H7-A negative. MG-CP-1 is not built.** No §2, no MG-CP-2, no further `D_t` redefinition
(pre-committed: three consistent operationalizations is the answer). The MG-CP spec is
closed.

## What this adds to the paper — H1c sharpened, not weakened

> The model contains interventionally-validated uncertainty-specific features (H1c
> stands, 12/12). But activation of those features — whether any single feature, or
> high co-activation of the set — does **not** induce a separated empirical tail-error
> regime: the 90th-percentile nonconformity score conditioned on high Type-B activation
> is within noise of the score conditioned on low activation (R_Q ∈ [0.91, 1.06] across
> ETTh1/ETTh2 × 3 seeds, 5/6 CIs contain 1), in-distribution and under shift.

This is the **second of two falsifications** in the chain (H3 is a measurement boundary,
not a falsification — see below):

- **H3 — measurement boundary.** The invariance question cannot be cleanly *asked* of
  these features: sparse activation collapses the aggregate difficulty signal below any
  conditional-drift metric's resolution (four aggregations confirm). Not "tested and
  rejected" — "not measurable at this sparsity".
- **Task G — falsification.** A *continuous* learned scorer on the features (or any
  representation) loses to a constant-width band; the learned-conditioning paradigm has
  a net cost. Well-powered, pre-registered, budget-matched control.
- **H7-A — falsification.** A *discrete* regime gate on the features has no regime to
  gate: Type-B activation and tail-error magnitude are statistically unrelated.
  Well-powered, pre-registered, three converging operationalizations.

**Boundary claim for the discussion:** *identifiability of an internal uncertainty
mechanism (H1c) does not imply that mechanism is statistically exploitable for
uncertainty quantification — neither by continuous conditioning (Task G) nor by discrete
regime-gating (H7-A).* Both cleanly-testable exploitation strategies were pre-registered,
tested, and falsified, with the mechanistic discovery itself untouched.
