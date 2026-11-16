# Paper frame — LOCKED 2026-09-04, EXTENDED 2026-09-09 (Phase 3 + Phase 4)

## 2026-09-09 extension — what Phase 3 and Phase 4 add, in one place

Two new pieces of evidence, on **ETTh1 + ETTh2** (HTTP ETDataset mirror — same split
sizes as the Kaggle bundle used for H1c, different edge fingerprint; Phase 1 retrained
locally for self-consistent MVDF selections; electricity/traffic raw data Kaggle-only,
unavailable locally — the same 2-dataset scope as H7-A and Result 4). Neither reopens a
closed question; both **sharpen** the central claim.

- **Phase 3 (Result 3-prime — direct feature-level causal→predictive dissociation).**
  The real-data version of the multi-estimator natural-activation predictability battery
  (`experiments/phase3_real_predictability.py`), previously run only inside Phase 1's
  synthetic environment. Question: do the **natural, non-intervened** activations of the
  L2-validated MVDF features predict `S_t` out-of-sample? Four held-out estimators
  (OLS, isotonic, gradient boosting, pinball MLP), same E_train/E_select/E_eval
  discipline. **Result: negative on every (dataset, seed).** Best held-out R² is
  **negative** — ETTh1 −0.58 to −0.72, ETTh2 −0.89 to −0.92 (every estimator predicts
  `S_t` worse than its own training mean), and stays negative when restricted to windows
  where an MVDF feature is active (ETTh1 −0.48 to −0.70, ETTh2 −0.80 to −0.89). The
  `R_j`-vs-natural-predictive-utility rank correlation is ≈ 0 (−0.10 to +0.13). The
  identical battery recovered R² 0.26–0.48 on Phase 1's designed-in signal, so the
  pipeline finds natural predictive signal when it exists. **This makes Result 3a's
  downstream conformal failure a *consequence* of a cleaner upstream fact, measured
  directly:** a feature validated by L2 (interventional) dose-response evidence carries
  essentially no L1 (associational) natural predictive information about difficulty here.

- **Phase 4 (Result 5 — oracle ceiling).** Mondrian-on-truth
  (`experiments/phase4_oracle_ceiling.py`): bin E_eval_ID / E_eval_shift windows into
  deciles of their **true realized `S_t`**; a window's oracle half-width is the (1−α)
  conformal quantile of `S_t` over the *other* windows in its own decile (leave-one-out;
  own value never sets own width). `naive` = the constant (1−α) block quantile. Both sit
  at ~0.90 coverage by construction, so `W_naive − W_oracle` is a clean heterogeneity
  measurement. **Result: a large, statistically clean gap on every dataset and block,
  all seeds** — `(W_naive − W_oracle)/W_naive` = ETTh1 24.5% (ID) / 25.4% (shift),
  ETTh2 29.2% (ID) / 24.5% (shift); bootstrap CIs tight and far from zero;
  precondition (gap meaningfully positive) satisfied everywhere.

**What the two together establish, and how it changes the framing:**

The paper's central tension is now **quantified from both ends**. An oracle that knows
true per-window difficulty could tighten intervals ~25–29% at the same coverage (Phase
4) — so ETT difficulty heterogeneity is **real and substantial**, not a near-
homoskedastic setting. Yet **none of that ~25% is accessible** through the forecaster's
internal representation, causally-validated or otherwise (Phase 3: zero natural
predictive signal in four estimator classes), nor through any adaptive-conformal
machinery tested (Task G, H7-A, SLCP, CQR/PID). The negatives are therefore **genuine
method failures against a real, quantified opportunity** — not "correctly finding
nothing to find."

**Reconciles with Phase 0** (`corr(hidden, |e|)` ≈ 0.08–0.21, previously read as "ETT is
near-homoskedastic"): that measured *linear decodability from the hidden state*, not
whether heterogeneity *exists in the errors*. Phase 4 shows the latter clearly does. The
gap between "heterogeneity exists" and "nothing recovers it" is the paper's contribution,
now bounded on both sides.

**Causal-ladder vocabulary (Joshi et al. 2026, arXiv:2602.16698) — adopt as the paper's
formal frame.** `P_j`, `U_j`, `R_j` are **L2 (interventional)** estimands — dose-response
slopes under a controlled perturbation. `C_j` and the Phase 3 predictability battery are
**L1 (associational)** — natural activation vs `S_t`, no intervention. The paper's
central finding is a **rung mismatch**: L2 evidence (causal specificity) does not license
an L1 claim (natural predictive information), and the downstream exploitation tests
further show it does not license L3-style decision utility. This replaces the
project's earlier homemade `C`/`P`/`U` notation attempts — same quantities, established
citable framework. Language discipline: use "encodes / represents / the mechanism for /
controls" only for the L2 dose-response results; use "predicts / correlates with / is
decodable from" for Phase 3, `C_j`, and any observational statement. Elazar et al. 2021
("amnesic probing") is the direct precedent that decodability (L1) ≠ the model's
computation using a property (L2/L3).

- **Architecture generalization (Result 6 — the findings are not attention-specific).**
  `experiments/arch_gen_timemixer.py` + `forecasters/timemixer.py`. The project's
  forecaster has been iTransformer (attention) throughout; this re-runs the pipeline on a
  **structurally different architecture** — MLP-based, channel-independent,
  multi-resolution mixing, no attention (a faithful minimal TimeMixer with the SAE
  tap/patch interface). ETTh1 + ETTh2 × 3 seeds. Forecast quality matches iTransformer
  (ETTh1 val_mse ~0.70, ETTh2 ~0.21); SAEs healthy (FVU 0.0008–0.0027). **Three gate
  checks, reported per (dataset, seed), all 6/6:** (1) **H1c** — MVDF type-B enrichment
  1.2–2.5× (iTransformer 2.0–4.6×; same regime, lower end); (2) **Task G** —
  `naive_matched` narrower than the learned dense scorer under shift by −0.24 to −0.28,
  all significant; (3) **Phase 3** — best held-out natural-activation R² −0.49 to −1.16,
  marginal *and* activation-conditioned. No layer sweep on the new architecture
  (`pre_head` re-used by design, §3.1). → **The central findings — L2 discovery, the
  learned-scorer failure, and the L1 dissociation — reproduce on a non-attention
  forecaster. They are not an iTransformer artifact.**

- **Phase 2 (bounded — necessity / denoising-noising test).**
  `experiments/phase2_necessity_noising.py`. Dose-response shows `U_j ≠ 0` under
  intervention → *sufficiency*. This adds the other direction (Heimersheim & Nanda
  denoising/noising; Joshi et al. Case Study I "Sufficient ≠ Necessary"): NOISING (zero
  the MVDF codes in a clean run) vs DENOISING (corrupt all codes to their mean, restore
  only MVDF), each with matched-size random-feature-set and matched-L2-norm
  random-direction controls, plus a natural-range on-manifold check. *[Result folded in
  from `runs/phase2_necessity/PHASE2_NECESSITY_RESULT.md` when the run completes — early
  signal: `delta_noise` is small and near zero, consistent with the features being
  causally difficulty-*specific* under a dose-response slope but sparse enough that
  zeroing them outright barely moves `S_t`; i.e. sufficiency holds, necessity is not
  strongly established — H1c's causal claim should be stated as sufficiency.]*

- **Phase 5 (SAE functional reproducibility — H1c is a POPULATION claim).**
  `experiments/phase5_sae_reproducibility.py`, ETTh1 + ETTh2. Forecaster trained once
  per dataset; **3 independent SAEs (seeds 10/20/30)** on the same frozen hidden states
  (all healthy, FVU 0.007–0.012). Match run-0's validated MVDF features to another run by
  **causal signature** — standardized Euclidean distance in `(P_j, U_j, R_j)` space,
  mutual nearest neighbour — *not* decoder-weight correlation (permutation/rotation
  freedom makes raw-coordinate matching a poor criterion; the reason the consistency
  literature moved past feature-ID matching). **Result: low functional reproducibility of
  individual feature identity** — mean cross-seed MVDF match rate **0.21–0.29** at a
  0.5-SD threshold, *at or below* the pool base rate (0.30–0.39) and never above the
  signature-shuffle floor. MVDF membership buys **no** extra cross-seed reproducibility
  over a generic feature. → **H1c must be stated purely at the population level** (type-B
  enrichment 2.0–4.6× is a property of the SAE feature *space*, reproducible in
  aggregate); **no claim about specific feature identities.** This does *not* touch
  H1c's causal validity — the dose-response intervention is real for whatever features a
  given SAE recovers — it bounds what can be said about their identity. A clean, in-
  setting confirmation of "Low Loss → Identifiability is false" (Joshi et al. 2026;
  Leask et al. 2025 SAE-canonical-units).

**Artifacts:** `runs/phase3_real/PHASE3_RESULT.md`, `runs/phase4_oracle/PHASE4_RESULT.md`,
`runs/phase5_sae_repro/PHASE5_RESULT.md`, and the matching `AGG_*.csv`.
`eval/consolidate_results.py` has loaders + sections for all three (and Phase 1).

---

# Paper frame — LOCKED 2026-09-04 (post Task G + matched-budget confirmation)

**The matched-budget rerun confirmed the result** (`v2_h6_topup/`, 2026-09-04). At the
SAME 80%-of-E_eval_ID calibration budget the learned modes get, `naive_matched` shift
width 3.345 vs `naive_no_scorer` (30%) 3.331 — a **+0.015** difference (P(narrower)=0.00,
wrong sign for "more data helps", pure noise). `naive_matched − dense` shift width
**−1.009, 12/12 significant, P=1.00**; `naive_matched − mvdf` **−0.461, 12/12 sig**. The
budget imbalance was doing nothing; the scorer penalty is real. Frame is final as
written below.

**This supersedes the "freeze the framework, then beat current methods" plan.** Task G
answered the question that plan was built to answer, and the answer kills SAE-CP as a
proposed method. Do not build a method around SAE conditioning. Do not run a
"beat-the-baselines" leaderboard. The paper is
**discovery (H1c) → measurement boundary (H3) → two falsifications (Task G continuous,
H7-A discrete) → one route not cleanly testable here (SLCP — temporal-autocorrelation
confound) → a literature-baseline check (CQR, PID-conformal) that puts Task G / H7-A in
context: no evaluated adaptive-CP method, standard or SAE-conditioned, is uniformly
narrower than a constant-width band at equivalent valid coverage on the two datasets
tested**.

**Wording discipline on H3:** H3 is *not* a falsification. The degeneracy audit + four
aggregation choices established that the invariance question **cannot be cleanly asked**
of these features (sparse activation collapses the aggregate signal below the metric's
resolution) — not that it was asked and answered "no". Call it a measurement-boundary
result. Task G and H7-A *are* falsifications: well-powered, pre-registered, controlled
comparisons that came back negative.

---

## The one-line claim

> Neural forecasters contain causally identifiable (L2 / interventional) uncertainty-
> specific internal features, and their target — genuine per-window difficulty
> heterogeneity — is real enough that an oracle could tighten intervals ~25% at matched
> coverage; yet those same features carry essentially no natural (L1 / associational)
> predictive information about difficulty, and no learned or standard adaptive-conformal
> method recovers the oracle gap. The mechanistic signal is causally real but
> observationally and practically inert — a rung mismatch, not a missing signal.

*(Earlier one-liner, still valid for the downstream half: "a learned difficulty-
conditioning scorer built on these features, or on any other representation we tried,
fails to beat a parameter-free constant-width split-conformal baseline — and the scorer,
not the representation, is where the failure lives.")*

## Title

**When Prediction Is Not Difficulty: Mechanistic Probing of Uncertainty Representations
in Neural Forecasters**

## Abstract (working)

> We investigate whether neural forecasters encode predictive difficulty in identifiable
> internal representations and whether such representations can support sharper conformal
> prediction under distribution shift. Using sparse autoencoders and causal dose-response
> interventions, we identify features whose manipulation selectively changes predictive
> difficulty across forecasting environments — an interventional (L2) result. We then
> show this does not transfer to an associational (L1) one: across four held-out
> estimator classes, the *natural* activations of these causally-validated features
> predict the deployed nonconformity score no better than its mean, both marginally and
> when restricted to windows where the features are active. This is not for want of a
> target — a retrospective oracle that bins by true realized difficulty attains ~25–29 %
> narrower intervals than a constant-width band at matched coverage on both datasets, so
> the difficulty heterogeneity is real and substantial. Nor does any tested method
> recover it. A controlled comparison shows *continuous* learned difficulty scorers are
> consistently wider and less calibrated than a parameter-free constant-width
> split-conformal predictor, regardless of whether the scorer operates on dense hidden
> states, SAE-selected features, random SAE subsets, or PCA representations — and the
> severe shift-coverage failures of SAE-conditioned scorers disappear under the
> constant-width control, locating the failure in the learned conditioning mechanism
> rather than the representation. A *discrete* alternative — gating the conformal quantile
> on whether the difficulty features are highly co-active — fares no better: the
> tail-error distribution does not separate by regime. Standard adaptive-CP baselines
> (CQR, PID-conformal) likewise fail to establish a robust improvement over the
> constant-width band. Our synthetic positive control confirms the discovery-through-
> exploitation pipeline recovers a causal→predictive→decision chain when one is built in
> by construction, so the real-data negatives reflect the data, not the method. These
> results expose a precise rung mismatch: a mechanistically identified, causally real
> uncertainty representation that is neither observationally predictive nor practically
> exploitable, against a difficulty signal that demonstrably exists.

---

## The results — one discovery; one measurement boundary; two falsifications; one route confounded

Downstream-exploitation routes tried: continuous scoring (3a, falsified), discrete
regime-gating (3b, falsified), similarity-weighted localization (3c, premise check
confounded — not a falsification). H3 (Result 2) is a measurement boundary on the
invariance question, also not a falsification. Only 3a and 3b are clean negatives.

### Result 1 — Mechanistic discovery (POSITIVE)
Causal dose-response + isotonic residualization (`R_j`) identifies SAE features whose
intervention selectively changes predictive **difficulty** independent of ordinary
predictive importance. Type-B enrichment 2.0–4.6×, **12/12 runs**, 4 datasets. Neither
correlation screening nor single-point ablation finds this feature type. **This is an L2
(interventional) result, and — per Phase 5 — a POPULATION-level one:** the type-B
*enrichment* reproduces across independent SAE seeds, but individual feature *identity*
does not (cross-seed causal-signature match rate 0.21–0.29, at the pool base rate). State
it as "the SAE feature space reliably contains an enriched population of causally
difficulty-specific features", not "feature 1648 is the difficulty feature".

### Result 2 — Measurement boundary, NOT a falsification (H3)
The invariance question — are the difficulty features *more shift-invariant* than
correlation-selected ones — **cannot be cleanly asked** of these features. The
aggregation diagnostic (four aggregation choices, all confirmed) shows why:
difficulty-specificity forces sparse/localized activation (~4% of windows), which
collapses the aggregate difficulty signal below the resolution of any conditional-drift
metric. The nominal comparison (`I(MVDF_v2) < I(correlated)` on 2/10 non-degenerate
runs) is a coin flip *because the metric is degenerate for signals this sparse*, not
because invariance was tested and rejected. Difficulty is *represented*, but not as a
globally smooth invariant scalar subspace that this class of metric can score. **Do not
call this a falsification** — it is a statement about what these features' sparsity
permits measuring.

### Result 3-prime — Direct feature-level causal→predictive dissociation (NEGATIVE) — Phase 3

Before the downstream conformal question (3a): do the **natural, non-intervened**
activations of the L2-validated MVDF features carry *any* associational (L1) predictive
information about `S_t`? This is the more basic prior question, and answering it directly
removes the "maybe the MLP just wasn't powerful enough" objection that 3a alone leaves.

`experiments/phase3_real_predictability.py`, **ETTh1 + ETTh2 × 3 seeds** (HTTP mirror,
Phase 1 retrained locally for self-consistent selections; electricity/traffic Kaggle-only
— same scope as H7-A / Result 4). Four held-out estimators under the E_train/E_select/
E_eval discipline: OLS (no regularization; n ≈ 1170 ≫ p = 12), 1-D isotonic (monotone-
only), gradient boosting (depth-3, lr 0.05, 0.8 subsample — deliberately conservative),
pinball MLP (2×64 GELU, 300 epochs, no weight decay/dropout, median target).

- **Marginal battery (all E_eval_ID windows): best held-out R² is NEGATIVE on every
  (dataset, seed)** — ETTh1 −0.58 to −0.72, ETTh2 −0.89 to −0.92. Every estimator,
  including the two most flexible (GBM, MLP — the MLP is the single worst on every run),
  predicts `S_t` worse than its training-set mean.
- **Activation-conditioned battery (windows where ≥1 MVDF feature fires): still
  negative** — ETTh1 −0.48 to −0.70, ETTh2 −0.80 to −0.89. This settles the "no signal
  because rarely active" vs "no signal despite being active" question: it is the
  **stronger dissociation** — the features do not predict difficulty even on the windows
  where they are actually active.
- **`R_j`-vs-natural-predictive-utility rank correlation ≈ 0** (−0.10 to +0.13 across
  runs). Direct feature-level evidence for the dissociation, stronger than inferring it
  from 3a's downstream conformal failure.
- **Positive control:** the *identical* battery recovers R² 0.26–0.48 on Phase 1's
  synthetic designed-in signal (`runs/phase1_synthetic/`). The pipeline finds natural
  predictive signal when it exists.
- → **A feature validated by L2 (dose-response, interventional) evidence carries
  essentially no L1 (natural, associational) predictive information about difficulty on
  this data.** 3a's conformal failure is a *consequence* of this cleaner upstream fact.
  The regularization caveat runs the right way: the flexible models don't overfit to a
  spurious positive R², they are flat/negative like OLS; a tuning sweep could only pull
  GBM/MLP toward 0, never toward real signal that OLS and isotonic already show is absent.

### Result 3a — Downstream falsification, continuous conditioning (NEGATIVE, controlled, budget-matched)
The learned pinball-loss MLP difficulty scorer — the FFDCI-style adaptive-conditioning
paradigm — is **wider and less calibrated** than a constant-width split-conformal band,
ID and under shift, on all 4 datasets. This is **not representation-specific**:
`dense ≈ SAE-MVDF ≈ SAE-correlated ≈ SAE-random ≈ PCA`, and all lose to the parameter-
free baseline. It is **not a calibration-budget artefact**: `naive_matched` (given the
same 80%-of-E_eval_ID calibration data the learned modes use) reproduces
`naive_no_scorer` (30%) to within +0.015 shift width — the imbalance changed nothing.

Key numbers (12 runs, α=0.10; `v2_h6_topup/`):
- shift width: `naive_no_scorer` **3.33**, `naive_matched` **3.35** vs dense 4.35,
  mvdf 3.81, correlated 3.89, random_k 3.94, pca_k 4.36
- ID coverage: `naive_no_scorer` 0.908, `naive_matched` 0.895 vs every MLP mode ~0.87–0.88
- `naive_matched − dense` shift width **−1.009, 12/12 significant, P=1.00**
- `naive_matched − mvdf` shift width **−0.461, 12/12 significant**
- `naive_matched − naive_no_scorer` shift width **+0.015** (budget-sensitivity control:
  the 30→80% calibration change is noise)
- **electricity control**: `mvdf` shift coverage 0.827 / 0.630 / 0.811 →
  `naive_matched` 0.894 / 0.895 / 0.894 (and `naive_no_scorer` 0.895 flat). The collapse
  is introduced by the learned conditioning mechanism, not by the SAE representation.

### Result 3b — Downstream falsification, discrete regime-gating (NEGATIVE) — H7-A

If a *continuous* scorer on the features fails (3a), does a *discrete* regime gate on
them work? Test (`experiments/h7a_gate.py`, full fidelity, SAE FVU 0.004–0.010): define
`D_t = 1` for windows in the top decile of Type-B (cell-B) co-activation `A_t`; ask
whether the 90th-percentile nonconformity score separates by regime,
`R_Q = Q_.90(S|D=1) / Q_.90(S|D=0)`.

**Scope: ETTh1 + ETTh2 only** (traffic excluded per Task A — pinned ID coverage;
electricity excluded per Task B — confounded shift coverage), seeds 0/1/2. Same 2-of-4
restriction as the H6/Task G evidence base; the two remaining datasets are the only ones
with valid coverage in this project.

- Pre-committed gate: `R_Q` CI excludes 1 (above) on E_eval_ID, both datasets, all 3
  seeds. **Result: 1/6 seed-runs clears it.** `R_Q ∈ [0.91, 1.06]`, 5/6 CIs contain 1;
  `P(R_Q>1)` spans 0.04–0.99. No consistent direction. E_eval_shift: same (only ETTh2 s1).
- Well-powered (n₁ 215–626), absolute tail quantiles near-identical in every cell — an
  *absence of separation*, not a noisy estimate. Three `D_t` operationalizations agree.
- → **Type-B activation and tail-error magnitude are statistically unrelated.** MG-CP-1
  was not built (`H7A_RESULT.md`). The MG-CP spec is closed.

### Result 3c — Similarity-weighted localization (SLCP) — NOT CLEANLY TESTABLE (methodological note)

A third exploitation route: weight the conformal quantile by feature-space similarity
(LCP/RLCP). The pre-committed premise check (`experiments/slcp_gate.py`, full fidelity):
does any SAE space give k-NN neighbourhoods with lower local `S_t` variance than dense /
PCA, on all 6 dataset-seed cells?

- **No SAE space passes** the `ratio < 0.80 on 6/6 cells` bar at trust-bar SAE fidelity
  (`correlated`/`mvdf` mean ~0.89 with cells > 1.0; `random_k` mean 0.79 but one cell
  0.97).
- **The temporal-adjacency control passes decisively** (ratio 0.34–0.39 vs dense, all 6
  cells): `S_t` is strongly serially autocorrelated, so k-NN in *any* temporally-smooth
  space recovers temporally-near windows and inherits low local score variance for free.
  The random-Gaussian-dense-projection control does *not* pass (1.04), ruling out generic
  linear compression.
- **Temporal-exclusion check (§2):** rerun the metric excluding the ±40 nearest windows
  in time from each candidate pool — structurally removing the confound. Still no SAE
  space clears the 6/6 bar (`random_k` best at 4/6, `correlated`/`mvdf` 2/6). "Not enough
  signal once you control for time" is now a number, not an argument.
- → **Not a falsification of SAE structure.** The premise check fails on a
  temporal-autocorrelation confound, not on the representation. **Methodological finding:**
  local-structure claims in time-series CP must residualize `S_t` against its own serial
  correlation (or restrict neighbours to be temporally distant) — an LCP/RLCP variant
  this project's data cannot cleanly support at n ≈ 2000 ID windows. SLCP §2 not built
  (`SLCP_RESULT.md`).

### Result 4 — Literature-baseline check (CQR, PID-conformal) — CONTEXT for 3a/3b

Do standard adaptive-CP methods beat `naive_matched` where the SAE-conditioned ones did
not? Built each in its own standard form (`baselines/`): **CQR** = a genuine
quantile-regression iTransformer (pinball loss, same trunk, no SAE) + Romano et al.
split-conformal; **PID-conformal** = Angelopoulos et al. P+I online update on raw point
residuals. Scope: **ETTh1 + ETTh2 × 3 seeds** (electricity/traffic raw files were
Kaggle-only, not in the local checkout, and the LTSF-standard substitutes are
non-comparable data — a documented limitation; these are also the two datasets H7-A ran
on and the two with clean coverage throughout).

- **PID-conformal** ties `naive_matched` on aggregate (shift width −0.005 over 6 cells):
  **narrower & valid on ETTh1** (3/3 seeds, ~3 % tighter, significant), **wider & valid
  on ETTh2** (3/3 seeds). A dataset-dependent wash — the best showing of any adaptive
  method, and still not a robust win.
- **CQR** is not competitive on either dataset: ETTh1 undercovers ID (0.71–0.77);
  ETTh2 both wider and over-covers under shift (0.98–0.99). Never narrower at equivalent
  valid coverage. (Its quantile forecaster converges but its per-step coverage decays at
  long horizons — a standard-architecture capacity limit, reported not patched.)
- → **The SAE-conditioned methods (Task G, H7-A) are not uniquely poor.** They sit in
  the same band as the standard literature baselines, none of which establishes a robust
  improvement over a constant-width band on these data. This makes the paper's downstream
  claim **general with a stated ceiling**, not an SAE-specific miss. (`BASELINE_RESULT.md`)

### Result 5 — Oracle ceiling: the difficulty heterogeneity is real (Phase 4)

Every negative result above is diagnosed by mechanism (scorer overfits; no regime;
temporal artifact). One question a careful reviewer still asks: **was there ever enough
exploitable heterogeneity for *any* method to find?**

`experiments/phase4_oracle_ceiling.py`, **ETTh1 + ETTh2 × 3 seeds**. Mondrian-on-truth:
bin E_eval_ID / E_eval_shift windows into deciles of their **true realized `S_t`**; a
window's oracle half-width = the (1−α) conformal quantile of `S_t` over the *other*
windows in its own decile (leave-one-out — own value never sets own width). `naive` =
the constant (1−α) block quantile. α = 0.10, 10 deciles. Both at ~0.90 coverage by
construction (naive 0.901, oracle 0.903) → `W_naive − W_oracle` is a clean heterogeneity
measurement, not coverage-confounded.

- **`(W_naive − W_oracle) / W_naive`, per dataset (mean over seeds):**
  ETTh1 ID **24.5 %**, ETTh1 shift **25.4 %**, ETTh2 ID **29.2 %**, ETTh2 shift **24.5 %**.
- Bootstrap CIs on the gap are tight and far from zero on every run; the precondition
  (gap meaningfully positive before any gap-recovery ratio is interpreted) holds
  everywhere.
- → **ETT difficulty heterogeneity is real and substantial** — an oracle could tighten
  intervals ~25–29 % at matched coverage. This is *not* a near-homoskedastic setting.
- **Non-negotiable:** the oracle is retrospective and non-deployable (it uses the true
  realized score). It is reported once as a ceiling; it never enters a deployable
  comparison table.
- **Reconciles Phase 0** (`corr(hidden, |e|)` ≈ 0.08–0.21): that measured linear
  decodability *from the hidden state*, not heterogeneity *in the errors*. The errors are
  heteroskedastic; the hidden state just doesn't expose it linearly — and Phase 3 shows
  the SAE features don't expose it non-linearly either.

**3-prime + 5 together:** ~25 % is on the table (5); zero of it is reachable through the
representation (3-prime) or any tested adaptive-CP method (3a/3b/3c/4-baselines). The
negatives are method failures against a quantified opportunity, not the absence of one.

---

## The decomposed causal chain (the paper's spine)

```
SAE
 │
 ▼  H1c: YES (12/12)      L2 / interventional          [DISCOVERY]
causally difficulty-specific features
 │
 ▼  H3: invariance question NOT CLEANLY ASKABLE        [MEASUREMENT BOUNDARY, not a falsification]
 │      (sparse activation → aggregate signal below metric resolution; 4 aggs confirm)
 │
 ▼  Phase 3 (3′): natural activation → S_t?   L1 / associational   [FALSIFICATION, feature-level]
 │      best held-out R² NEGATIVE, 4 estimators, marginal AND activation-conditioned
 │      (ETTh1/ETTh2 × 3 seeds; synthetic control recovers R² 0.26–0.48)
 │      → L2 causal specificity does NOT carry L1 natural predictive information
difficulty representation  (causally real, observationally inert)
 │
 ├─▼  Task G (3a): CONTINUOUS learned scorer  →  worse than constant-width split CP   [FALSIFICATION]
 │                 (not representation-specific: dense ≈ SAE ≈ PCA; budget-matched)
 │                 — now a *consequence* of 3′, not an independent surprise
 │
 ├─▼  H7-A (3b):    DISCRETE regime gate  →  no regime to gate                        [FALSIFICATION]
 │                 (R_Q ≈ 1.0 ± 0.06; Type-B activation ⊥ tail-error magnitude)
 │
 ├─▼  SLCP (3c):    SIMILARITY-WEIGHTED localization  →  premise check fails on a      [NOT TESTABLE HERE]
 │                 temporal-autocorrelation confound, not on SAE structure
 │
 └─▼  CQR / PID (4): STANDARD adaptive-CP baselines  →  also not uniformly narrower     [CONTEXT]
 │                  than the constant band at equal valid coverage (ETTh1/ETTh2);
 │                  PID ties on aggregate, small non-transferring ETTh1 edge
 ▼
better conformal prediction  →  NO  (no evaluated adaptive method, SAE-conditioned or
                                      standard, robustly beats a constant-width band here)

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Phase 4 (Result 5): was there anything to find?  →  YES.                 │
    │  Oracle binning by TRUE realized S_t is 25–29% narrower than the constant │
    │  band at matched coverage (ETTh1/ETTh2, ID + shift, all seeds, tight CIs).│
    │  Retrospective, non-deployable — a ceiling, never a baseline.             │
    │  ⇒ the negatives are method failures against a real, quantified gap.      │
    └──────────────────────────────────────────────────────────────────────────┘
```

Controls localize each result. **3a:** `naive_matched` (same data, framework, calibration
budget; scorer vs no scorer) — the failure disappears when the scorer disappears. **3b:**
three independent `D_t` operationalizations plus the absolute-quantile and n₁ diagnostics
— the null is an *absence of a regime*, not an underpowered test. **3c:** the
temporal-adjacency and random-Gaussian-projection controls — the apparent local structure
is serial correlation of `S_t`, not the representation, so the premise check itself is
not valid here. All are mechanistic control experiments, not just extra baselines.

---

## Claims discipline — say exactly this, not more

| DO NOT say | DO say |
|---|---|
| "Constant-width conformal beats adaptive conformal prediction." | "On ETTh1/ETTh2 under our shift protocol, no evaluated adaptive-CP method (learned-scorer family, regime gate, CQR, PID-conformal) is uniformly narrower than a constant-width split-conformal baseline at equivalent valid coverage; the best case, PID-conformal, ties on aggregate with a small non-transferring ETTh1-only edge." |
| "Learned scorers don't work." | "The evaluated pinball-loss MLP difficulty scorers introduce a sharpness/calibration penalty that is not recovered by changing their input representation." |
| "SAE features fail to transfer under shift." | "The electricity shift-coverage collapse was introduced by the learned difficulty-conditioning mechanism; it disappears under the constant-width control." |
| "MVDF is a better conformal method." | "Causal selection is at best marginally better than random selection *within* an otherwise inadequate learned-scorer family (Task D)." |
| "SAE-conditioning specifically fails downstream." | "SAE-conditioned methods sit in the same performance band as standard adaptive-CP baselines (CQR, PID-conformal); none robustly beats a constant band on the datasets tested — the ceiling is on adaptive conditioning here generally, not on the SAE." |
| "Adaptive conformal prediction doesn't work." | "The scope of the negative result is these methods, this shift protocol, and two datasets (ETTh1/ETTh2); electricity/traffic under the same protocol and datasets with genuinely predictable heteroscedastic difficulty are untested and remain the natural boundary conditions." |
| "ETT difficulty is homoskedastic, so there was nothing to find." | "An oracle binning by true realized difficulty is 25–29 % narrower than a constant band at matched coverage (Phase 4) — the heterogeneity is real and substantial; it is just not accessible through the representation or any tested adaptive method. Phase 0's low `corr(hidden,|e|)` measured linear decodability from the hidden state, not heterogeneity in the errors." |
| "The MVDF features encode difficulty but it doesn't transfer to prediction." (mixing rungs) | "The MVDF features are causally difficulty-specific under intervention (L2); their natural activations do not predict difficulty (L1) — best held-out R² is negative across four estimators, marginally and conditioned on activation (Phase 3). Reserve 'encode / represent' for the interventional result." |
| "The learned scorer / MLP just wasn't powerful enough (Phase 3)." | "Four estimator classes were tested — OLS, isotonic, gradient boosting, a pinball MLP; the two most flexible are among the worst, and OLS/isotonic (nothing to tune) already put the ceiling at ≈ 0. A regularization sweep could only move GBM/MLP toward 0, not toward absent signal." |
| "We ran the pipeline on synthetic data instead of real data." | "The synthetic run is a positive control only — it confirms the discovery→exploitation pipeline recovers a causal→predictive→decision chain built in by construction (links 1–3, 3/3 seeds), so the real-data ETTh1/ETTh2 negatives are a property of the data, not a pipeline artifact." |
| "This is an iTransformer / attention finding." | "All three central results (H1c discovery, the learned-scorer failure, the L1 dissociation) reproduce on a structurally different MLP-based non-attention forecaster (TimeMixer), 6/6 dataset-seed runs — Result 6." |
| "The MVDF features are *necessary* for difficulty tracking." | "The dose-response result establishes *sufficiency* (`U_j ≠ 0` under intervention). The bounded noising/denoising test [Phase 2] does not strongly establish necessity — zeroing the MVDF set barely moves `S_t` — so state H1c as sufficiency, not necessity." |
| "Feature 1648 is *the* difficulty feature." | "H1c is a population-level claim: the SAE feature space reliably contains an enriched type-B population, but individual feature *identity* does not survive an SAE reseed (Phase 5 cross-seed causal-signature match ≈ pool base rate)." |

---

## Claim-language audit — evidence rung vs verb choice (Joshi et al. 2026 §2.4 method)

Every claim sentence classified by the rung its **evidence** rests on vs the rung its
**verb** implies. L1 = associational (`p(y|h)`), L2 = interventional (`p(y|do(h))`),
L3 = counterfactual (per-instance). A mismatch (verb implies a stronger rung than the
evidence) is the failure mode to avoid.

| Claim (as stated in the paper) | Evidence rung | Verb rung | OK? |
|---|---|---|---|
| "Dose-response + `R_j` **identifies** SAE features whose intervention **selectively changes** predictive difficulty" (Result 1) | **L2** — `P_j`/`U_j` are dose-response slopes under a controlled perturbation | L2 ("changes", "intervention") | ✅ verb matches |
| "…features that **encode** difficulty" — *avoid this phrasing for Result 1's population*; the L2 evidence is about *effect of intervening*, and Phase 5 shows identity isn't stable | L2 (effect) but NOT identity (Phase 5) | "encode" leans L2-representational + implies a stable unit | ⚠️ **say "the feature space contains a causally difficulty-specific population", not "feature X encodes difficulty"** |
| "The **natural activations** of these features **do not predict** `S_t`" (Result 3-prime) | **L1** — held-out regression on non-intervened activations, no `do()` | L1 ("predict", "natural activation") | ✅ verb matches; keep "predict / decodable from / correlates with" |
| "L2 causal specificity **does not license** an L1 claim (natural predictive information)" (central) | comparison of an L2 result to an L1 test | meta-claim about rungs | ✅ this *is* the rung-mismatch thesis, stated as such |
| "A learned scorer on these features **fails to beat** a constant band" (Result 3a) | **L1→decision** — the scorer conditions on natural activations; validity by conformal calibration | decision-level ("fails to beat", widths/coverage) | ✅ verb matches; not an L2 claim |
| "The electricity shift-coverage collapse **is introduced by** the learned conditioning mechanism" (Result 3a control) | **L2-style control** — swap the scorer for `naive_matched` on identical data, collapse disappears | causal ("introduced by") | ✅ this is a genuine intervention (ablate the scorer), verb matches |
| "An oracle **could tighten** intervals ~25 %" (Result 5) | retrospective construction on realized `S_t` | modal / retrospective ("could", "ceiling") | ✅ explicitly hypothetical and non-deployable, never stated as achievable |
| "The findings **reproduce** on a non-attention architecture" (Result 6) | same three tests on TimeMixer, 6/6 | replication ("reproduce") | ✅ verb matches |
| "H1c **is a population-level** claim" (Phase 5) | cross-seed causal-signature matching at the pool base rate | scoping statement | ✅ correctly downgrades an identity claim to a population one |
| "The MVDF set is **necessary**" — *do not say* | noising barely moves `S_t` (Phase 2) | "necessary" implies L2-necessity | ⚠️ **say "sufficient" only** |

**Editing pass rule for the manuscript:** search for `encode`, `represent`, `the
mechanism for`, `controls`, `necessary`, `uses` — each is only licensed where a
dose-response intervention or an ablate-the-component control backs it (Results 1, 3a's
scorer-ablation control, Result 6's replication of those). Everywhere the evidence is a
regression on natural activations (Result 3-prime, `C_j`, the `R_j`-vs-predictive-utility
figure) use `predict`, `correlates with`, `is decodable from`, `is associated with`.
Elazar et al. 2021 (amnesic probing) is the citation for why the distinction is not
pedantic: decodability (L1) and computational use (L2/L3) can be uncorrelated.

---

## What Task D means now (de-loaded)

Task D (mvdf vs random_k direction) is **no longer load-bearing**. It is reported as a
within-family curiosity, not a method decision:
- Even if `MVDF > random_k`, still `MVDF-MLP < naive` on the central comparison.
- Current direction: `mvdf − random_k` −0.138, CI [−0.28, −0.02], 3/4 datasets by Δ_d,
  reverses on ETTh2, confounded with coverage on electricity/traffic.
- **No SAE-CP method ships.** No `configs/frozen/sae_cp_default.yaml`. No Task D gate.

---

## What is NOT in scope anymore

- ❌ SAE-CP / SAE-conditioned adaptive conformal prediction as a proposed method
  (continuous — Task G; discrete regime-gating / MG-CP — H7-A; both falsified)
- ❌ The freeze-doc baseline suite as a "beat current methods" leaderboard
  (CQR/RLCP/PID/EVT/Mondrian) — building it to win would be fighting the evidence
- ❌ `configs/frozen/sae_cp_default.yaml`, the Task D decision gate
- ❌ MG-CP §2 (MG-CP-1 build), MG-CP-2, any further `D_t` redefinition — H7-A gate failed
- ❌ SLCP §2 (LCP/RLCP weighted-quantile build) — §1 premise check fails on a temporal-
  autocorrelation confound; not cleanly testable on this data
- ❌ Any framing where H1c "propagates" downstream

## What a reviewer might still ask for (rebuttal budget, not v1)

- A stronger scorer (GBM quantile / deeper MLP / longer training) to confirm it's the
  *paradigm* not this MLP. Prior: `naive` still wins, gap is −1.02 / 12-of-12.
- A dataset with genuinely predictable heteroscedastic difficulty, where a conditional
  scorer *should* help — as a boundary condition on Results 3a/3b.
- 1–2 standard CP baselines (plain split-CP is already `naive`; CQR on the forecaster's
  own quantiles) purely to place `naive` on the existing CP map — NOT as a horse race.
- n=5 seeds for the H1c error bars.

---

## Artifact status

- `V2_FINAL_REPORT.md` — rewritten around Task G (Finding 3 = negative). `--verify` clean.
- `RESULTS_FINAL.md` — regenerated; H6 from `v2_h6_topup/`, H1c/H3 from `v2_confirm/`.
- `FINISHING_TASKS_STATUS.md` — Tasks A–G done; Task G section carries the inversion.
- `H7A_RESULT.md` — H7-A negative (`runs/h7a_gate/`, full-fidelity, SAE FVU 0.004–0.010).
  Result 3b. MG-CP spec closed.
- `SLCP_RESULT.md` — SLCP §1 premise check STOP + §2 temporal-exclusion confirmation
  (`runs/slcp_gate/`, full-fidelity). Result 3c: temporal-autocorrelation confound,
  methodological note not a falsification. SLCP spec closed at the gate.
- `BASELINE_RESULT.md` — CQR + PID-conformal vs `naive_matched`, **ETTh1/ETTh2 × 3
  seeds** (`runs/baseline_comparison/`, done 2026-09-06). electricity/traffic
  data-availability-limited (raw files Kaggle-only). Result 4: no adaptive method
  robustly beats the constant band; PID ties on aggregate, CQR worse; SAE-conditioned
  methods not uniquely poor.
- `baselines/` — `quantile_itransformer.py` (CQR's quantile forecaster),
  `cqr.py`, `pid_conformal.py`. `experiments/baseline_comparison.py` — the driver.
- `eval/consolidate_results.py` — `--h6-root` flag; `naive_matched`/`naive_no_scorer` in `MODES`;
  Phase 1/3/4 loaders + sections added 2026-09-09.
- `runs/phase1_synthetic/PHASE1_RESULT.md` — synthetic positive control (config frozen
  `ec9470b5f521616c`; 4 pre-run DGP corrections, all gated on pre-declared manipulation
  checks). Links 1–3 recover 3/3 seeds; link 4 mixed 1/3 (reproduces Task G instability).
- `runs/phase3_real/PHASE3_RESULT.md` — Result 3-prime. Natural-activation predictability
  battery, real ETTh1/ETTh2 × 3 seeds. Best held-out R² negative, 4 estimators, marginal
  and activation-conditioned. `experiments/phase3_real_predictability.py`.
- `runs/phase4_oracle/PHASE4_RESULT.md` — Result 5. Oracle ceiling, ~25–29 % gap on
  ETTh1/ETTh2 (ID + shift, all seeds). Retrospective/non-deployable.
  `experiments/phase4_oracle_ceiling.py`.
- `runs/phase5_sae_repro/PHASE5_RESULT.md` — SAE functional reproducibility. Individual
  MVDF feature identity does not survive an SAE reseed (cross-seed match 0.21–0.29 ≈ pool
  base rate); H1c is a population claim. `experiments/phase5_sae_reproducibility.py`.
- Memory: `project_mechanistic_conformal.md` — 2026-09-04 section + 2026-09-09 Phase 1
  and Phase 3/4 sections.
