# E0.2 — Definitions as implemented

No new runs. Every formula below is read from the code at commit `fb7e43f` (baseline snapshot).
Paths are relative to `mechanistic-conformal/`.

## 1. Windows, splits, stride

| Item | As implemented | Where |
|---|---|---|
| Window | `X = series[s : s+96]`, `Y = series[s+96 : s+192]`, `end_idx = s + 95` (window origin = last input step). All channels in, all channels out. | `data/datasets.py:360-366` |
| **Stride** | **1** (`make_windows(..., stride=1)` default; never overridden). Adjacent windows share 95 of 96 target steps. | `data/datasets.py:360` |
| Train/val/test rows (ETTh) | `[0, 8640)`, `[8640, 11520)`, `[11520, 17420)` → 5709 test windows. Non-ETT: 70/10/20 of rows. | `data/datasets.py:339-343` |
| Per-window normalisation (RevIN-style) | `x̃ = (x − mean_L(x)) / (std_L(x) + 1e-6)` per channel over the 96-step input; forecast de-normalised with the same stats. | `forecasters/itransformer.py:155-156` |
| Test carve-out | Windows assigned by `end_idx` relative to `c1 = round(T_test/3)`, `c2 = round(2T_test/3)`: `E_select = end_idx < c1`, `E_eval_ID = c1 ≤ end_idx < c2`, `E_eval_shift = end_idx ≥ c2`. ETTh: 1872 / 1966 / 1871 windows. | `shift/shift_windows.py:88-101` |
| Phase-3 fit/eval split | Chronological 60/40 of `E_eval_ID`: `cpt = int(0.6·n_ID)` → fit 1179, eval 787 (ETTh). | `experiments/phase3_real_predictability.py:173` |
| Conformal split (`naive_matched`, learned scorers, CQR, PID) | Chronological 50/30/20 of `E_eval_ID`: `c1 = int(0.5n)`, `c2 = int(0.8n)` → scorer-train 983 / conformalize 589 / width-eval 394 (ETTh). `naive_matched` calibrates on the first 80% (train ∪ conformalize). | `experiments/phase3_scorer_h6.py:103`, `:172-175`; `experiments/baseline_comparison.py:146` |

## 2. The difficulty score `S_t`

```
abs_err[t, h] = | ŷ[t, h, OT] − y[t, h, OT] |              h = 1..96, target channel only
profile[h]    = mean_t abs_err[t, h]  /  mean_h( mean_t abs_err[t, h] )      (mean-1 shape)
S_t           = max_h  abs_err[t, h] / profile[h]
```
`validation/errors.py:58-61`.

- **Channel aggregation: none.** Only the target channel (`target_var = −1`, OT for ETT; last column for electricity/traffic) enters `S_t`. The other channels are inputs only.
- **Horizon aggregation: max** over the 96 steps of the profile-normalised absolute error. `S_t` is the smallest scalar half-width multiplier that would cover all 96 steps of the target channel.
- **Which split `profile` comes from: whatever block `collect()` is handed.** `collect()` recomputes the profile from the same windows it scores (`errors.py:59-60`). Consequences:
  - Dose-response (`P_j`, `U_j`) uses the **E_select** profile (`phase3_real_predictability.py:116-117`, passed as `prof` to `sweep_pool`).
  - The Phase-3 R² battery scores `S_t` on E_eval_ID with the **E_eval_ID** profile (`:171`), i.e. the normaliser is computed from the evaluation outcomes.
  - Phase 4 and `phase3_scorer_h6` give each block (scorer-train, conformalize, width-eval, shift) **its own** profile.
- Units: normalised-data units (inputs are z-scored on the train split; `data/datasets.py:get_splits`). Because `mean_h profile = 1`, `S_t` is on the scale of a per-step absolute error.

## 3. The "90% interval" and coverage

Band for window t (all methods, `calibration/difficulty_scorer.py:105-126`):
```
half-width scalar  q_t            (constant for naive; per-window for scorers/oracle)
band[t, h]       = ŷ[t, h, OT] ± q_t · profile[h]
covered_t        = max_h |y[t,h,OT] − ŷ[t,h,OT]| / profile[h]  ≤  q_t      ⇔  S_t ≤ q_t
```
- **Covered = all 96 horizon steps of the target channel inside the band** (joint over horizon, single channel). Not per-channel, not per-step.
- `naive_matched`: `q = empirical_quantile(S_cal80, 0.90)` with the conformal `+1` correction, `rank = ceil((n+1)·0.9)` clipped to n (`calibration/aci.py:17-25`), then adapted online by `GlobalACI` with immediate feedback (`aci.py:28-52`, `gamma = gamma_global`). `naive_no_scorer` is the same on the 30% conformalize block only.
- Phase-4 oracle: `q_t` = conformal 0.90-quantile of `S` over the *other* windows in t's own true-`S` decile (leave-one-out), whole block; naive = block-wide constant quantile (`phase4_oracle_ceiling.py:69-90`, `:126-138`).

## 4. Interval width as reported

`width_t = mean_h (upper − lower) = 2 · q_t · mean_h profile[h] = 2 · q_t` (`difficulty_scorer.py:126`; Phase 4 reports `2·half_width` directly, `phase4_oracle_ceiling.py:93-97`). Tables report the mean over evaluated windows. **Units: normalised target units** (train-split z-score of the target channel); not divided by anything further. Oracle "% narrower" = `(W_naive − W_oracle) / W_naive`.

## 5. Feature-level quantities

All computed on E_select only (`phase3_real_predictability.py:113-158`).

| Symbol | As implemented | Where |
|---|---|---|
| Eligibility screen | SAE feature eligible if active (≠0) on ≥ 2% of E_select windows. | `validation/correlation_screen.py:24-25` |
| Candidate pool | 120 features drawn uniformly without replacement from eligible ones, `rng = default_rng(derive_seed(seed, ds))`; then dose windows `wsel` = min(2048, n_select) windows drawn with the same rng (ETTh: all 1872). | `phase3_real_predictability.py:135-140` |
| `C_j` | `Cov(a_j, S_t)` over E_select windows where `a_j > 0` (observational). Not used for selection. | `validation/importance_and_difficulty.py:37-48` |
| Intervention | Target-variate token at tap `pre_head` replaced by `unstd(decode(z'))` where `z'_j = λ·z_j`, other codes unchanged. **The SAE reconstruction error is not added back**: the patched token is the SAE reconstruction (also at λ=1). | `validation/dose_response.py:42-72`, `:110-116` |
| `D^pred_j(λ)`, `D^unc_j(λ)` | Means over **active** windows (`z_j ≠ 0`, ≥5 needed) of horizon-MSE (target channel) and of `S_t` (E_select profile), λ ∈ {0, .25, .5, .75, 1, 1.25}. | `dose_response.py:75-106` |
| `P_j`, `U_j` | OLS slope of `D^pred_j` (resp. `D^unc_j`) on λ over the 6 points. Fit quality was **not** logged. | `dose_response.py:119-128`, `:158-163` |
| `R_j` | `U_j − Ê[U_j | P_j]`, Ê = isotonic regression of U on P over the pool, direction (increasing/decreasing) chosen by lower SSE. | `validation/difficulty_specificity.py:25-68` |
| Types A/B/C/D | Median split on **absolute values**: hi_P = `|P_j| ≥ median|P|`, hi_U = `|U_j| ≥ median|U|`. C = hi_P∧hi_U, A = hi_P∧¬hi_U, B = ¬hi_P∧hi_U, D = neither. | `difficulty_specificity.py:71-94` |
| MVDF selection | Gate 1: `|U_j| ≥ 0` (threshold 0, i.e. no-op) **and** `D^unc_j(λ)` monotone (all diffs ≥ −1e-9 or all ≤ 1e-9) **and** `R_j` finite. Gate 2: top **k = 12** by `R_j` (signed, descending). No sign condition on `U_j` or `P_j`. | `difficulty_specificity.py:112-142`; `k`, `u_threshold` at `phase3_real_predictability.py:75` |
| Correlated set | Top-12 pool features by `|Pearson(a_j, S_t)|` on E_select. | `phase3_real_predictability.py:155-156` |
| Type-B enrichment | `(#MVDF in B / #MVDF in pool) / (#B / #pool)`. **No p-value or CI is computed** in the code or in `AGG_h1c.csv`. | `difficulty_specificity.py:97-109` |

## 6. Discrepancies between text-level descriptions and code (for the camera-ready)

1. **Phase-3 R² denominator** uses the **evaluation-set mean** (`experiments/phase1_synthetic_control.py:76-78`), not the fit-split mean. E3 reports both (primary = fit-split mean, as the camera-ready spec states).
2. **`profile` is block-local** (point 2 above): the Phase-3 target and every conformal block are normalised by a profile computed from their own outcomes. E3 uses the E_select profile throughout, and reproduces the original definition for the F0 reference row.
3. **Interventions drop the SAE reconstruction error** (point 5). The spec for E5c says the existing code "keeps the reconstruction-error term fixed"; it does not. E5c follows the existing code for comparability with `U_j`.
4. **"Covered" is joint over the horizon on one channel** — any text implying multivariate or per-step coverage should be corrected.
5. **`naive_matched` includes online ACI with immediate feedback**, which is not available in deployment at horizon 96 (round-2 finding). E3 gap recovery uses static split conformal for every method so the comparison is like-for-like.
6. **Phase 5 headline threshold is 0.25 SD**, not 0.5 SD (fixed in E0.1), and its "signature-shuffle floor" is identical to the match rate by construction (see E0 RESULT).
