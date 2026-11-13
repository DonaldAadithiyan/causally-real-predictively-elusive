# Finishing-tasks — status

Against `mechanistic-conformal-finishing-tasks.md` (Tasks A–G).

**All tasks A–G complete, and the Task G budget confound is closed.** Task G ran
(`v2_h6_topup/`, 2026-09-04); a follow-up matched-budget rerun (`naive_matched`, same
folder) confirmed the result is not a calibration-data-allocation artefact. Net: H6
inverts and the SAE-CP method track ends. See `PAPER_FRAME.md` (**LOCKED**) — the "freeze
the framework, then beat current methods" plan is superseded: no SAE-CP method ships, no
frozen config, no Task D gate, no baseline leaderboard. The paper is discovery (H1c) →
attempted mechanism (H3) → controlled falsification (Task G).

Task F: not done — Tasks A/B are structural, not cheap fixes; the cells are
directional/coverage-qualified in `RESULTS_FINAL.md`, and Task G's `naive_no_scorer`
control makes the electricity coverage question moot anyway (the collapse is a
learned-scorer artefact; `naive` holds 0.895 flat).

---

## Task A — traffic pinned ID coverage (1.000)  ✅ DIAGNOSED

**Fact:** traffic ID coverage = 1.000 on **all 18 H6 runs** (6 modes × 3 seeds) — so it
is structural to the dataset, not seed noise and not `mvdf`-specific.

**Root cause:** the H6 in-distribution split (`E_eval_ID` → 50% MLP-train / 30%
conformalize / 20% ID-width-eval) is **chronological**. traffic's `E_eval_ID` is only
**1169 windows** (vs ~1966 ETT, ~2334 electricity). The conformalize block (middle 30%,
~350 windows) and the ID-width-eval block (last 20%, ~235 windows) are disjoint time
periods; under traffic's strong daily/weekly cycle the residual scale drifts within
`E_eval_ID`, so the split-conformal `q_resid` calibrated on the middle block over-covers
the last block. Result: ID coverage 1.000, ID width inflated (~11 vs shift ~8). The
online ACI term then adapts `q` *down* on the shift stream, which is why traffic shift
coverage is ~0.91 (near nominal) despite ID being pinned.

**This is not a score/profile bug** — the same chronological-split mismatch shows the
*opposite* sign on ETTh1/ETTh2 (ID coverage 0.79–0.81, i.e. undercoverage). electricity
(0.91) is closest to nominal because its `E_eval_ID` is largest.

**Resolution:** traffic's H6 row is **directional only** (explicitly flagged, not
dropped) — the shift-width ranking `mvdf/correlated/random_k < dense < pca_k` matches the
other 3 datasets, but the scorer is not tightly ID-calibrated on traffic at this split
budget. A real fix needs a larger `E_eval_ID` (a different env carve-out + a re-run),
which is out of scope for this phase per the finishing-tasks §1 discipline. Reported in
`RESULTS_FINAL.md`.

---

## Task B — electricity `mvdf` shift undercoverage (mean 0.756 vs 0.90 target)  ✅ DIAGNOSED

**Fact, per seed:** `mvdf` electricity shift coverage = 0.827 / **0.630** / 0.811
(seeds 0/1/2). Seed 1 is the outlier. `dense` holds ~0.876 all seeds.

**It is not `mvdf`-only.** `random_k` also drops (seed 1: 0.744). All three
sparse-SAE-subset modes (`mvdf`, `correlated`, `random_k`; per-feature activation
~2–9%) lose more shift coverage on electricity than `dense`/`pca_k`. `mvdf` is worst
because it is the sparsest (median per-feature activation 0.037 vs correlated 0.045).

**The 47-vs-48-channel quirk is ruled out** — the selected feature indices are SAE
latent dims (0–2047), not channel indices, and the iTransformer channel path is
identical for every mode. Checked the `mvdf` indices for electricity all 3 seeds:
nothing anomalous (range 147–2045, spread evenly).

**Mechanism:** consistent with the H3 sparse-activation finding — a ~4%-active feature
signal characterizes the shift distribution poorly, so the split-conformal `q_resid`
transfers badly from the conformalize block to the shift block, and the online ACI can't
fully compensate on electricity's shift block within one seed's draw (seed 1's draw is
adverse).

**Resolution:** report `mvdf`'s electricity shift-width advantage (≈23%, 2.55 vs dense
3.31) **with its coverage cost stated inline** (mean shift coverage 0.756 vs dense
0.876) — it is partly bought with coverage, not a clean win. `correlated` (0.836) and
`random_k` (0.800) are less affected. This does **not** overturn the H6 conclusion (SAE
basis, not selection) — all three sparse modes show the same shift-width benefit — it
qualifies electricity specifically. Reported in `RESULTS_FINAL.md`.

---

## Task C — MVDF_v1 enrichment tally  ✅ RESOLVED

Recomputed from raw `v2_phase1_seed*/AGG_h1c.csv`:

| criterion | ≥ 1.0× enrichment | range | mean |
|---|---|---|---|
| **MVDF_v2** | **12 / 12** | 2.00 – 4.62 | 3.02 |
| MVDF_v1 | **4 / 12** (ETTh1 s0/s1/s2, ETTh2 s0) | 0.00 – 1.85 | 0.69 |
| correlated | **2 / 12** | 0.00 – 1.20 | 0.64 |

**Corrections vs earlier reports:** MVDF_v1 was reported as **5/12** — it is **4/12**.
`correlated` was reported as "~4/12" / "3/12" — it is **2/12**. `--verify` flags both.

---

## Task D — `mvdf` vs `random_k` direction  ✅ RESOLVED (gate did not certify MVDF → `correlated` frozen as cheap default)

**Resolution (reached before Task G, unchanged):** only 2 of 4 datasets had valid
coverage for this comparison — traffic excluded (Task A, ID coverage 1.000) and
electricity excluded (Task B, `mvdf` shift undercoverage). The two remaining valid
datasets (ETTh1, ETTh2) split **1–1** on the `mvdf`-vs-`random_k` direction, so the
pre-committed **3-of-4 consistency rule could not certify MVDF**. Decision at the time:
**freeze `correlated` as the cheap default scorer input.**

Direction numbers for the record (all 4 datasets, coverage-confounded on 2):
- Aggregate paired bootstrap `mvdf − random_k` shift width: mean −0.138, 95% CI
  [−0.28, −0.02]; per-run `mvdf` narrower 8/12.
- Per-dataset `Δ_d`: ETTh1 −0.008, ETTh2 **+0.007**, electricity −0.132, traffic −0.418.
  On the two *valid* datasets: ETTh1 favours `mvdf` (barely), ETTh2 favours `random_k`
  (barely) → 1–1.

**Post-Task-G status:** moot for the paper. No SAE-CP scorer ships at all (Task G), so
neither `correlated` nor `mvdf` is a frozen method component. Task D is reported as a
within-family curiosity: causal selection is, at best, marginally better than random
selection *inside* a learned-scorer family that itself loses to `naive_no_scorer`.
`configs/frozen/sae_cp_default.yaml` is NOT generated.

---

## Task E — `eval/consolidate_results.py`  ✅ BUILT

- Default: regenerates `RESULTS_FINAL.md` from raw `v2_confirm/` outputs — H1c
  enrichment + tallies, H1/H1b ρ, H3 invariance + degeneracy + aggregation diagnostic,
  H6 coverage/width/bootstrap, Task A/B notes, Task D direction + `Δ_d`, and a
  machine-readable headline JSON block.
- `--verify`: diffs the headline numbers against the existing `*_REPORT.md` files. On
  first run it flagged: MVDF_v1 5→4, correlated 3→2, H3 valid-runs "/11"→"/10". No other
  new mismatches — the script is working.
- No `tabulate` dependency (local `md()` helper).

**`RESULTS_FINAL.md` is now the single source of truth. Nothing in the paper's tables
should be copied from any earlier `*_REPORT.md`.**

---

## Task F — targeted recalibration  ⏸ CONDITIONAL, deferred

Tasks A and B both identified the root cause as **`E_eval_ID` being too short for a
stationary conformalize→eval transfer** (traffic) / **sparse features transferring badly
to the shift block** (electricity `mvdf`). Neither has a *cheap* fix within the current
env carve-out:
- traffic: needs a larger `E_eval_ID` → a different 3-way split → a re-run.
- electricity `mvdf`: the sparse-transfer issue is intrinsic to the H3 finding; no
  calibration knob fixes it without changing what `mvdf` *is*.

Per finishing-tasks §7 ("if no cheap fix exists, do not force one") and §1 (don't expand
scope): **both cells are marked directional / coverage-qualified in `RESULTS_FINAL.md`,
not recalibrated.** If the freeze-doc's Task D gate needs a clean electricity coverage
number to certify MVDF-default, that is a scoped re-run decision for the user — not
something to force now.

---

## Task G — naive no-scorer baseline  ✅ RAN (`v2_h6_topup/`, 2026-09-04). **Result inverts H6.**

`calibration/difficulty_scorer.py::naive_no_scorer_run` — global split-conformal with a
**constant** half-width (the (1−α) quantile of the conformalize-set scores) plus the
online global-ACI term. No MLP, no feature input. 7th mode `naive_no_scorer`, 12 runs.

### `naive_no_scorer` beats every learned scorer on BOTH axes

| mode | ID cov | ID width | shift cov | shift width |
|---|---|---|---|---|
| **naive_no_scorer** | **0.908** | **4.08** | **0.903** | **3.33** |
| correlated | 0.874 | 4.68 | 0.890 | 3.89 |
| mvdf | 0.872 | 4.72 | 0.870 | 3.81 |
| random_k | 0.873 | 4.73 | 0.881 | 3.94 |
| dense | 0.882 | 4.58 | 0.896 | 4.35 |
| dense+mvdf | 0.884 | 4.57 | 0.897 | 4.33 |
| pca_k | 0.879 | 4.57 | 0.894 | 4.36 |

- `naive − dense` shift width **−1.023, 12/12 significant, P(narrower)=1.00**. Per-dataset
  (shift, all significant): ETTh1 −0.18, ETTh2 −0.38, electricity −0.18, traffic −3.36.
- `naive − mvdf` shift width **−0.475, 12/12 significant**. The constant band is narrower
  than the MVDF scorer too.
- **`naive` is the only mode that hits nominal ID coverage** (0.908; every MLP mode
  undercovers ID at 0.87–0.88, even after split-conformal).
- **`naive` fixes electricity (Task B).** `naive` shift coverage on electricity = 0.895
  flat across all 3 seeds (vs `mvdf` 0.827 / 0.630 / 0.811). The sparse-feature
  conformalize→shift transfer failure simply doesn't occur without a learned scorer.
- **traffic ID coverage is still 1.000 for `naive`** — confirms Task A: the pin is the
  chronological `E_eval_ID` split, not the scorer.

### Consequence for the H6 write-up — the conclusion inverts

Old H6: "an SAE-space projection is a more shift-robust scorer input than the dense
vector." New, honest H6:

> The pinball-loss MLP conformal scorer — in **any** input space (dense, SAE-selected,
> SAE-random, PCA) — produces **wider and less-calibrated** intervals than a
> no-parameter constant-width split-conformal band, on all four datasets under shift.
> The `dense → {mvdf, correlated, random_k}` width reductions were the SAE-space
> variants *partially recovering ground the MLP itself lost* — none of them reach the
> constant baseline. `mvdf`'s electricity shift-coverage collapse is a learned-scorer
> artefact that the baseline does not have.

This is a **negative result for the learned scorer family (FFDCI-style)**, not only for
MVDF. It does **not** touch H1c or H3.

### Calibration-budget audit (2026-09-04) — done by code-read, no rerun

Per `experiments/h6_and_diagnostic.py`: `n = len(id_idx); c1, c2 = int(0.5n), int(0.8n)`
→ `tr_i = id_idx[:c1]` (50%), `conf_i = id_idx[c1:c2]` (30%), `idw_i = id_idx[c2:]` (20%).

| mode | data used to set the interval scale | share of E_eval_ID |
|---|---|---|
| learned modes | MLP-train `tr_i` **+** conformalize `conf_i` | 50% + 30% = **80%** |
| `naive_no_scorer` | conformalize `conf_i` **only** (`naive_no_scorer_run(b_conf["score"], …)`) | **30%** |

**`naive` uses LESS calibration data, not more** — it ignores the 50% MLP-train block
entirely. So the "more data beats less data" alternative explanation runs *against*
`naive` here: it wins on 30% of the block while the MLP modes lose on 80%. The result's
direction is, if anything, conservative — a matched rerun (give `naive` the same
`tr_i ∪ conf_i` pool for its quantile) would most likely *widen* `naive`'s margin.

**Two things this does NOT fully clear:**
1. It is a non-favorable-to-`naive` imbalance, not a matched design. Clean version =
   every mode gets the identical calibration set. Worth one matched rerun to remove the
   asterisk (cheap: change one slice in the naive call). Expected outcome: `naive`'s
   margin grows or holds.
2. "The MLP is underfit at 50%-of-E_eval_ID train data" is a real narrower reading — but
   it is a statement about the *learned scorer's data hunger*, which is part of the
   paradigm's cost here, not a confound that rescues it.

### Matched-budget rerun — DONE, CONFIRMS (`v2_h6_topup/`, 2026-09-04, 96 H6 rows)

Added `naive_matched`: identical to `naive_no_scorer` but its quantile is over the full
`tr_i ∪ conf_i` pool (80%), the SAME calibration data the learned modes use.

| mode | ID cov | ID width | shift cov | shift width |
|---|---|---|---|---|
| `naive_no_scorer` (30%) | 0.908 | 4.08 | 0.903 | **3.33** |
| `naive_matched` (80%) | 0.895 | 4.07 | 0.901 | **3.35** |
| dense | 0.882 | 4.58 | 0.896 | 4.35 |
| mvdf | 0.872 | 4.72 | 0.870 | 3.81 |

- `naive_matched − naive_no_scorer` shift width **+0.015** (P(narrower)=0.00, wrong sign
  for "more data helps") → **the calibration budget was doing nothing.**
- `naive_matched − dense` shift width **−1.009, 12/12 sig, P=1.00**.
- `naive_matched − mvdf` shift width **−0.461, 12/12 sig**.
- electricity shift coverage `naive_matched` **0.894 / 0.895 / 0.894** by seed (vs `mvdf`
  0.827 / 0.630 / 0.811) — the stability pattern holds at matched budget.
- `naive_matched` ID coverage 0.895 (vs `naive_no_scorer` 0.908) — the 80% pool's
  quantile is pulled slightly down on the short ETT ID blocks; still far above every
  learned mode (~0.87).

**The scorer penalty is real, not a data-allocation artefact. `PAPER_FRAME.md` is
LOCKED.** Both narrower-claims safe: constant-width dominates every learned scorer on
width AND coverage, at matched budget.

---

## Freeze-doc §5 baseline suite  ❌ CANCELLED (2026-09-04)

Superseded by Task G (+ matched-budget confirmation) + `PAPER_FRAME.md`. There is no
SAE-CP method to defend, so there is no leaderboard to run. Building CQR/RLCP/PID/EVT/
Mondrian to "win" would be fighting the evidence. If a reviewer asks, the rebuttal budget
is: plain split-CP (= `naive_matched` already), CQR on the forecaster's own quantiles,
and a stronger scorer (GBM/deeper MLP) to confirm Result 3 is about the paradigm not this
MLP — as *placement*, not a race.

---

## Deliverables checklist (finishing-tasks §10)

- [x] Traffic ID-coverage diagnosis + resolution (directional-only) — this file + `RESULTS_FINAL.md`
- [x] Electricity `mvdf`-shift undercoverage diagnosis + resolution (coverage-qualified) — same
- [x] MVDF_v1 tally recomputed from raw (4/12) and reconciled (was 5/12)
- [x] `mvdf` vs `random_k` direction stated with signed difference (−0.138, 8/12, `Δ_d` 3/4)
- [x] `eval/consolidate_results.py` built; `--verify` run and reviewed (3 mismatches, all expected)
- [x] Naive no-scorer baseline coded (`naive_no_scorer_run`, wired into both H6 scripts)
- [x] `RESULTS_FINAL.md` generated entirely by the script
- [ ] *(next Kaggle run)* Task G number populated; optionally the coverage-fix re-run for Task D certification
