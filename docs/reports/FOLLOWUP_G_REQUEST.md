# Follow-up G — request to the coding agent (verbatim)

Pre-registration: `results/camera_ready/G/PREREG.md`. **Round G was stopped on request before it
finished**: G2, G4 and G5 completed; G1 and G3 are partial (`results/camera_ready/G/STATUS.md`).

> **G — final round before the camera-ready.** Same rules as before: pre-register in `runs/camera_ready/G/PREREG.md`, never overwrite, block bootstrap with block 96, report per-seed values, median [min, max] and k/N. No new training except where stated.
>
> **G1. Table 7 on one data copy.** On the 30 local pipelines (ETTh1/ETTh2 × 10 seeds, electricity/traffic × 5), run every downstream mode the paper reports: naive_matched, naive_no_scorer, scorer modes dense / mvdf / correlated / random_k / pca_k, CQR and PID. Use the existing protocol unchanged. For each dataset × mode × block (ID, shift), report coverage and width as median [min, max], plus the number of runs where the mode is narrower than naive_matched with coverage in [0.88, 0.92]. Give me a compact replacement for Table 7: one row per mode, shift block, four datasets. The paper's Table 7 currently comes from Kaggle runs, and I need everything on the local copy.
>
> **G2. Oracle gap with a pre-evaluation profile.** Recompute the Phase 4 oracle gap (ID and shift) and the Phase 3 MVDF best R² using the **E_select** profile for S_t instead of the block-local one. Report both versions side by side for all 30 runs, plus the rank correlation between the two S_t definitions on each block.
>
> **G3. Synthetic positive control on 10 seeds.** Re-run `runs/phase1_synthetic/` (Links 1–4: causal sign, enrichment, predictability R², downstream gain) on seeds 0–9 with unchanged settings and the paper's existing pass rules. Report k/10 for each link and the positive-control R² as median [min, max]. Link 4 was 1/3 in the paper, and it is the result reviewers attack most.
>
> **G4. The direct L2 evidence per run.** For each of the 30 runs, give the fraction of MVDF features whose U_j has a bootstrap CI excluding 0 (bootstrap over active windows, the same dose-response as E5a), against the same fraction for the rest of the pool and for a random draw of 12 features. Report per dataset as median [min, max] and k/N, where k counts runs in which the MVDF fraction exceeds the pool fraction. This replaces enrichment as the main "causally real" evidence.
>
> **G5. One sentence check.** Confirm that under static calibration (round 2, seeds 0–2), no learned scorer is consistently narrower than naive_matched on the shift block, and give the exact range, so I can quote it.
>
> Deliver `G/RESULT.md` with a "For the paper" section, and update `CAMERA_READY_SUMMARY.md` §3 with any further number changes.
