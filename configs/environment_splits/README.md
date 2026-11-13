# Committed environment splits (V2 spec §9, §12, §15)

The three-way chronological carve-out of each dataset's TEST split:

| Block | Fraction of test split | Role |
|---|---|---|
| `E_select` | first 1/3 | rank/threshold features, fix `MVDF_v2_set`. Nothing downstream re-touches it. |
| `E_eval_ID` | middle 1/3 | in-distribution invariance + calibration/evaluation |
| `E_eval_shift` | final 1/3 | the pre-registered distribution-shift block |

`E_train` = the existing forecaster/SAE training split (unchanged from Run 5).

Fractions are `(1/3, 1/3, 1/3)` for every dataset — committed here, in this file,
**before Phase 2 runs**. Boundaries are derived at runtime from each dataset's test
length by `shift/shift_windows.py::three_way_env_masks`. The KS check
(`describe_env_split`) is logged per dataset in every Phase 2 run; a dataset whose
`E_eval_shift vs E_select` KS p ≥ 0.05 is flagged, never swapped (spec §6.3, §15).

Do not edit fractions after seeing Phase 2 results (spec §12 item 12).
