# Follow-up F — request to the coding agent (verbatim)

Answered in `results/camera_ready/F/RESULT.md` (pre-registration `results/camera_ready/F/PREREG.md`).

> 1. Re-run `h7a_gate.py` on ETTh1/ETTh2 seeds 3–9 with unchanged settings. Report R_Q per seed with block-bootstrap CI and the pass count k/10.
> 2. Give one pooled test for type-B enrichment across all 30 runs (the two-level bootstrap CI the spec asked for, plus a Fisher or Stouffer combination of the 30 hypergeometric p-values).
> 3. Confirm whether the learned-scorer modes also use immediate-feedback ACI, like naive_matched, so I can state whether the comparison is like-for-like.
> 4. Explain "profile is block-local": which windows are used to compute profile_h for each block, and does S_t on an evaluation block use that same block's errors?
> 5. Report the shift-block (E_eval_shift) gap-recovery ratio for each dataset and the cells with valid coverage, so the paper sentence matches exactly.

Item 3 was flagged as the most important: at a 96-step horizon, immediate feedback uses errors that have not yet been observed when the forecast is made.
