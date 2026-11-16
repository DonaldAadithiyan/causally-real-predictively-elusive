# F — Follow-up items 1–5: pre-registration

Committed before any F computation.

## F1 Regime gating (H7-A), ETTh1/ETTh2 × seeds 0–9
- **Deviation from "re-run h7a_gate.py unchanged" (cannot be literal):** `h7a_gate.py` retrains a local SAE (20 000 steps, `configs/v2_phase2.yaml`) but takes the cell-B feature IDs from `v2_confirm/v2_phase1_seed{s}/…/feature_type_table.json`, which were produced by a *different* SAE (Kaggle runs). SAE feature indices are not comparable across SAEs, and those tables exist only for seeds 0–2. The construction therefore follows round 2's corrected path (`p126/r2_wp1.py` L142–163), applied to the camera-ready pipelines:
  - cell-B = type-B features of the seed's own candidate pool (E5a features.csv, which reproduces the stored type tables);
  - A_t = fraction of cell-B features active;
  - τ = 90th percentile of A_t on E_select;
  - D_t = 1[A_t ≥ τ];
  - S_t with the E_select profile;
  - R_Q = Q.90(S | D=1) / Q.90(S | D=0); undefined if either group < 10.
- CI: moving-block bootstrap, block 96 (primary) and 192 (sensitivity), 2000 resamples, seed 0.
- Pass rule (H7-A's own pre-committed rule): CI excludes 1 from above on E_eval_ID. Report k/10 per dataset; E_eval_shift is a secondary readout. Seeds 0–2 at block 192 are compared with round 2's `wp1/regime_gating.csv` as a reproduction check.

## F2 Pooled type-B enrichment test, 30 runs
1. Two-level bootstrap CI of the mean enrichment ratio over all 30 runs (resample runs, then the per-run feature bootstrap replicate), plus per dataset.
2. Fisher and Stouffer combination of the 30 one-sided hypergeometric p-values (null: the 12 selected features are a random draw from the pool).
3. **Added because (2) uses the wrong null** (round 2 showed the selection rule itself produces enrichment from shuffled inputs): the perm-U selection null of round-2 WP3, run on all 30 runs. U_j and its monotonicity flag are permuted across pool features with P_j fixed, then the unchanged selection and type split are re-run (1000 permutations, rng seed = derive_seed + 2026). Per-run p, then Fisher and Stouffer over the 30. This is the test the paper's population-level claim should rest on.

## F3 ACI in learned scorers — code reading only, no run.

## F4 Profile locality — code reading only, with window counts.

## F5 Shift-block gap recovery — read from the existing E3/E1 aggregates (no recomputation): per dataset × seed × feature set, coverage, validity (|cov − 0.90| ≤ 0.02), GRR. Also checked: "the learned scorer never gives a narrower band at valid coverage on any of 30 runs", from `E2/AGG_E2_scorer_vs_naive.csv` (point width < naive and |cov − 0.90| ≤ 0.02, ID and shift, dense and mvdf).
