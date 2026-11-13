# SLCP §1 gate — STOP. Methodological note, not a falsification.

**Run:** `experiments/slcp_gate.py`, full fidelity (20-epoch iTransformer + 20k-step SAE,
SAE FVU 0.004–0.010), ETTh1 + ETTh2 × seeds 0/1/2, 2026-09-04. Outputs
`runs/slcp_gate/{SLCP_gate.csv, SLCP_gate_decision.json}`.

## The question (SAE-localized CP spec §1)

Before building any localized-CP (LCP/RLCP) weighted-quantile machinery, check the
premise: is there local structure in the nonconformity score `S_t` for a
similarity-weighted method to exploit? For a sample of E_eval_ID windows, in each of 5
representation spaces, take each window's 50 nearest neighbours and measure the variance
of `S_t` among them.

  `ratio_space = mean_local_var(S_t | space 50-NN) / mean_local_var(S_t | dense 50-NN)`

**Pre-committed rule (locked before running):** proceed only if ≥1 of {mvdf, correlated,
random_k} has `ratio < 0.80` AND bootstrap-CI-excludes-1 vs **both** dense and pca_k, on
**all 6** dataset-seed cells. Plus two confound controls with interpretation
pre-committed:
- **Control A — temporal-adjacency:** neighbours = the 50 calendar-nearest windows, zero
  feature information. If this alone clears the bar → STOP, premise compromised.
- **Control B — random Gaussian dense projection:** a fixed random 12×256 Gaussian
  projection of the dense vector. If this clears the bar → STOP, effect is generic
  linear compression.

## Result

| space | mean ratio vs dense (6 cells) | by-cell range | passes base rule (6/6)? |
|---|---|---|---|
| **`temporal_adj`** (control A) | **0.362** | 0.34 – 0.39 | — (the confound itself) |
| `random_k` | 0.792 | 0.57 – 0.97 | ✗ |
| `correlated` | 0.888 | 0.62 – 1.36 | ✗ |
| `mvdf` | 0.891 | 0.56 – 1.24 | ✗ |
| `pca_k` | 0.981 | — | ✗ |
| `rand_gauss_dense` (control B) | 1.040 | 1.00 – 1.09 | ✗ |

- **Control A clears the bar on all 6 cells, decisively** (ratio 0.34–0.39 vs dense; the
  50 calendar-nearest windows have ~⅓ the local `S_t` variance of dense-space k-NN).
  `S_t` is strongly serially autocorrelated — hard-to-forecast stretches span
  consecutive windows.
- **No SAE space passes the base rule.** At trust-bar SAE fidelity, `correlated` and
  `mvdf` mean ~0.89 with several cells above 1.0; `random_k` mean 0.79 but ETTh1 s0 =
  0.97. `sae_spaces_passing_base = []`.
- **Control B does not clear the bar** (1.04) — the effect is not generic to any 12-d
  linear compression of the dense vector; a dense random projection has no temporal
  structure to leak.

## Verdict (pre-committed)

> **STOP.** Control A (temporal-adjacency) alone clears the bar, so the premise is
> compromised. k-NN in any temporally-smooth representation incidentally recovers
> temporally-nearby windows, and because `S_t` is serially autocorrelated, those
> neighbours share low `S_t` variance *for free* — independent of feature content. The
> apparent "local structure" that `random_k`/`correlated` showed in the fast-mode smoke
> (ratios 0.63/0.79) was largely this, plus a fast-SAE artefact; at FVU ~0.005 it does
> not survive the 6/6-cell bar anyway.

**This is NOT a third falsification of SAE structure's downstream usefulness.** It is a
methodological finding:

> Local-structure claims in time-series conformal prediction must control for temporal
> autocorrelation of the nonconformity score. Any neighbourhood construction that
> incidentally recovers temporally-adjacent windows inherits low local score variance
> from serial correlation, not from the representation.

## §2 — temporal-exclusion check (grounds the "not enough signal" claim in a number)

Rerun the same k-NN gate metric but **exclude the ±40 nearest windows in time** from
each point's candidate neighbour pool before computing feature-space k-NN — structurally
removing the serial-correlation confound rather than only diagnosing it. Pre-committed
reading: if no SAE space clears the `<0.80 on 6/6 cells` bar once time is controlled,
"not enough signal once you control for time" becomes a number, not an argument.

**Result — CONFIRMS.** With ±40-window temporal exclusion:

| space | mean ratio vs dense | ratio vs dense by cell (6) | cells passing 6/6 bar |
|---|---|---|---|
| `random_k` | 0.730 | 0.79 / 0.55 / 0.69 / 0.59 / 0.92 / 0.84 | **4 / 6** (2 cells > 0.80) |
| `correlated` | 0.805 | 0.61 / 0.78 / 0.65 / 0.94 / 0.88 / 0.97 | **2 / 6** |
| `mvdf` | 0.943 | 0.70 / 1.36 / 0.72 / 0.90 / 0.91 / 1.08 | **2 / 6** |
| `pca_k` | 0.973 | — | — |

No SAE space clears the pre-committed bar once temporally-near windows are excluded —
`random_k` gets closest (4/6) but still fails two cells and its passing cells are noisy
(0.55 one seed, 0.92 another). **The "not enough signal once you control for time"
claim is now empirically grounded.** A valid test of representation-driven localization
would need to residualize `S_t` against its own temporal autocorrelation first — an
LCP/RLCP variant this project's data cannot cleanly support at n≈2000 ID windows.

### The `mvdf` vs `random_k` comparison (per the MVDF-LCP gate spec — "the comparison
that actually matters, not just pass/fail")

Even setting the 6/6 bar aside and reading the raw margins: **`random_k` (mean 0.730,
4/6 cells < 0.80) does *better* than `mvdf` (mean 0.943, 2/6 cells < 0.80)** under
temporal exclusion. Causally-validated selection is not carrying its own weight in the
localization layer — the opposite. This is the **same pattern Task G and H7-A already
showed**: whatever weak residual local structure survives temporal exclusion is a
property of taking *any* sparse SAE-space subset, not of causal specificity, and even
that does not reach the pass bar. There is therefore **no data-motivated case for the
`R_j`-weighted kernel** the MVDF-LCP spec §2.2 describes — its precondition ("build the
weighted kernel only if `mvdf` clears §1 by a materially larger margin than `random_k`")
is not met; it is contradicted.

## Decision

**SLCP-§2 / MVDF-LCP (the LCP/RLCP weighted-quantile build, the `R_j`-weighted kernel,
Kish-shrinkage) is NOT built.** Both the original SLCP spec's gate and the MVDF-LCP
spec's corrected temporal-exclusion re-test return the same verdict, now with a number:
no SAE space (causal or otherwise) shows local `S_t` structure that survives removing
the temporal shortcut from its own neighbour pool. Per the MVDF-LCP spec §1 first branch
and §4 stop rule: **this closes rung 3 of the exploitation ladder.** `PAPER_FRAME.md`
records it as a methodological note (Result 3c), distinct from the two falsifications
(Task G, H7-A) and the one measurement boundary (H3).

## Standing status of the exploitation question — LADDER CLOSED

- **Task G / rung 1** — continuous learned scorer: FALSIFIED (loses to `naive_matched`,
  budget-matched).
- **H7-A / rung 2** — discrete regime gate: FALSIFIED (no regime; `R_Q ≈ 1.0 ± 0.06`).
- **SLCP / MVDF-LCP / rung 3** — similarity-weighted localization: CLOSED at the gate.
  No SAE space shows local `S_t` structure that survives temporal exclusion (§2 above);
  `random_k` > `mvdf`, so no `R_j`-weighting is motivated. Not a falsification of SAE
  structure — a temporal-autocorrelation confound the data cannot see past.
- **CQR, PID-conformal** — literature adaptive-CP baselines: also do NOT robustly beat
  `naive_matched` (ETTh1/ETTh2; `BASELINE_RESULT.md`).
- **H3** — invariance: MEASUREMENT BOUNDARY (sparse activation → metric degenerate).

**All three designed exploitation mechanisms + two standard literature baselines have
failed to beat a constant-width band. Per the MVDF-LCP spec §4 stop rule: this is the
final state of the exploitation ladder. No fourth mechanism.** H1c (the discovery) is
untouched throughout and remains the paper's positive result.
