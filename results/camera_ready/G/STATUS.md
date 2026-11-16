# G — stopped on request (2026-11-16)

Round G was stopped before completion. All running jobs were killed and nothing further was started. Final status:

| item | status | output |
|---|---|---|
| G2 E_select-profile oracle + Phase 3 (30 runs) | **finished** | `AGG_G2_select_profile.csv` |
| G4 direct L2 evidence (30 runs, in-sample + out-of-sample) | **finished** | `AGG_G4_L2_evidence.csv`, `G4_per_feature_U_ci.csv` |
| G5 round-2 static check | **finished** | `AGG_G5_static_shift.csv` |
| G1 Table 7 on local copy | **not run.** Quantile models exist for 12/30 runs (`quantile/`); the script `camera_ready/g1_table7.py` reproduces the stored ETTh1 s0 naive_matched/CQR/PID exactly, but the 30-run table was never produced | partial: `quantile/`, `quantile_weights/` |
| G3 synthetic control, 10 seeds | **partial**: seeds 0 and 1 finished (seed 0 links 1–4 all pass; seed 1 links 1–3 pass, 4 fails, both as in the paper); seed 2 was killed mid-run (empty dir removed); seeds 3–9 not started. No AGG file. | `phase1_synthetic/seed0`, `seed1` |

No G/RESULT.md was written, and CAMERA_READY_SUMMARY.md §3 was not updated with G numbers.
