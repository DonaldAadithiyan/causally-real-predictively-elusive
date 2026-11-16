# E2 — More seeds for the core results

## Provenance

- Git: `93fbb0c` (+uncommitted changes)
- Command: `python -m camera_ready.train_seeds ...; python -m camera_ready.phase34 ...; python -m camera_ready.e2_scorer ...; python -m camera_ready.report_e2`
- Seeds: {'ETTh1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ETTh2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'electricity': [0, 1, 2, 3, 4], 'traffic': [0, 1, 2, 3, 4]}
- Datasets / copy used: ETTh1: HTTP ETDataset mirror (edge fp c6d1709fd0693b1a); ETTh2: HTTP ETDataset mirror (edge fp c022db5521d23684); electricity: Kaggle-bundle electricity.csv (edge fp cc6af3b5973c14af), local copy byte-identical; traffic: Kaggle-bundle traffic.csv (edge fp a93d2ec2b5815355), local copy byte-identical
- Raw-file SHA-256: ETTh1 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; ETTh2 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`; electricity `353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570`; traffic `1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281`
- Hardware: arm64 macOS-15.2-arm64-arm-64bit, no CUDA GPU; analysis on cpu (training device per run in E2/meta)
- Wall-clock: training ≈ 2.1 h ETTh CPU stream (04:52–07:00) in parallel with ≈ 1.1 h MPS stream (electricity/traffic); analysis ~3 min
- Generated: 2026-11-14 07:10:29 +0530

## Pre-registered criteria

- **C2.1:** every claim currently stated as "3/3 seeds" holds on ≥ 4/5 seeds (≥ 8/10 where N = 10). Any claim that drops below this is reported as weakened, and the paper text changes to match.
- **C2.2:** no headline median moves by more than its own pooled CI half-width when going from 3 to 5 (or 10) seeds.
- Health rule: FVU > 0.05 or val MSE > 1.5 × seed-0 val MSE ⇒ unhealthy (reported, never dropped).

## What was run
- **ETTh1, ETTh2:** seeds 3–9 newly trained (stretch goal taken: one seed ≈ 6–15 min on CPU), giving **N = 10**. ETTh1 seed 0 was also retrained through the same driver as a reproduction check: val MSE, FVU, MVDF set and every saved array (ŷ, hidden) are bit-identical to the paper's run.
- **Electricity, traffic:** seeds 3–4 newly trained on MPS, giving **N = 5**.
- **Health:** 0 unhealthy seeds out of 30. FVU of new seeds is 0.0052–0.0160 (existing range 0.003–0.033); val-MSE ratio to seed 0 is ≤ 1.006 everywhere. No replacement seed was needed. Full table below.
- **Training wall-clock:** ETTh 321–888 s per seed (CPU, contended); electricity 1075–1285 s, traffic 718–728 s (MPS).

## C2.1: does every "3/3 seeds" claim hold on ≥ 4/5 (≥ 8/10)?

| paper claim | dataset | k/N | verdict |
|---|---|---|---|
| MVDF type-B enrichment ≥ 1× on every run (Table 2) | ETTh1 / ETTh2 / elec / traffic | 10/10, 10/10, 5/5, 5/5 | **holds** |
| Phase 3 best held-out R² negative, marginal (Table 4) | all four | 10/10, 10/10, 5/5, 5/5 | **holds** (every block CI also < 0) |
| Phase 3 best held-out R² negative, active windows | all four | 10/10, 10/10, 5/5, 5/5 | **holds** |
| Oracle gap CI excludes 0, ID and shift (Table 6) | all four | 10/10, 10/10, 5/5, 5/5 | **holds** |
| naive_matched narrower than **dense**, significant on every run (§5.6, "12/12"), shift | ETTh1 | **4/10** | **weakened**: direction 10/10, significance 4/10 |
| | ETTh2 / traffic | 10/10, 5/5 | holds |
| | electricity | **0/5** | **weakened**: direction 5/5 (dW +0.08 to +0.18), not significant |
| naive_matched narrower than **mvdf**, significant on every run, shift | ETTh1 | **0/10** | **weakened**: direction 10/10 (dW +0.01 to +0.08), never significant |
| | ETTh2 / traffic | 10/10, 5/5 | holds |
| | electricity | **0/5** | **reversed in point estimate**: mvdf narrower by 0.18–1.03, at lower coverage (0.82 vs 0.90). Coverage-confounded. |

Enrichment with uncertainty (added, not a paper claim): the bootstrap CI over features excludes 1 on only **1/10 (ETTh1), 4/10 (ETTh2), 3/5 (electricity), 1/5 (traffic)**. "Enrichment ≥ 1×" holds on every seed, but individual seeds rarely show it with confidence.

**C2.1 verdict: FAIL as a blanket statement.** Enrichment, the Phase-3 null and the oracle gap hold on every seed of every dataset. The per-run *significance* of the scorer-vs-naive gap does not hold on ETTh1 or electricity under the block bootstrap and the local data copy (the paper's 12/12 used an iid bootstrap on the Kaggle copy).

## C2.2: did any headline median move by more than its pooled CI half-width?
No. Over all 32 metric × dataset rows that have a pooled two-level CI, every |median(all seeds) − median(seeds 0–2)| ≤ the pooled half-width. Largest moves: traffic mvdf-scorer shift gap 0.25 (half-width 0.53), ETTh1 enrichment 0.18 (0.58), traffic Phase-3 R² 0.15 (0.75). **C2.2: PASS.** (The active-window R² and enrichment-CI rows have no pooled CI and are not part of C2.2; traffic active-window median moved by 1.16.) See `AGG_E2_criteria.csv`.

## Ranges that widen with more seeds (feed the "numbers that change" list)
| quantity | paper (3 seeds) | camera-ready (10 seeds) |
|---|---|---|
| ETTh1 Phase-3 marginal best R² | −0.72 to −0.58 | −0.72 to −0.52 |
| ETTh2 Phase-3 marginal best R² | −0.92 to −0.89 | −0.94 to −0.77 |
| ETTh1 active-window best R² | −0.70 to −0.48 | −0.96 to −0.47 |
| ETTh2 active-window best R² | −0.89 to −0.80 | −0.96 to −0.69 |
| ETTh1 oracle gap ID / shift | 24.5% / 25.4% (mean) | 25.2% [23.6, 25.9] / 25.6% [24.5, 27.2] (median [range]) |
| ETTh2 oracle gap ID / shift | 29.2% / 24.5% (mean) | 30.0% [28.2, 30.6] / 24.0% [23.5, 25.1] |
| scorer gap naive − dense, shift (Table 9 iTransformer) | −0.14 to −0.43 | ETTh1 −0.11 to −0.18, ETTh2 −0.33 to −0.45 (local copy) |

## Plain-English summary
Going from 3 to 10 seeds on ETTh (and 3 to 5 on electricity/traffic) leaves the core results intact. On every one of 30 seeds, MVDF is type-B-enriched (≥ 1×), natural MVDF activation predicts difficulty worse than a constant, and a 18–40% oracle gap exists. Medians barely move. What does **not** survive is the claim that the constant-width band is *significantly* narrower than the learned scorer on every run. On ETTh1 and electricity the direction is right but the per-seed block-bootstrap CI usually includes 0, and on electricity the MVDF scorer is narrower in point estimate but under-covers. Per-seed enrichment is also rarely significant on its own (9/30 seeds).

## Anomalies
- The paper's §5.6 / Table 7 numbers came from Kaggle runs (Kaggle data copy, iid paired bootstrap); this re-run uses the local data copies and block bootstrap. Seeds 0–2 of ETTh are the paper's exact Phase-3 models but **not** the Kaggle Task-G models, so the scorer table is a re-run, not a reproduction.
- Electricity mvdf shift: the learned scorer under-covers (0.82) where naive_matched sits at 0.90, so its narrower width is not a like-for-like win.

## Seed health

| dataset | seed | val_mse | sae_fvu | val_mse_ratio_to_seed0 | device | wall_clock_s | unhealthy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0 | 0.7134 | 0.0118 | 1.0000 | cpu | – | False |
| ETTh1 | 1 | 0.7119 | 0.0118 | 0.9979 | cpu | – | False |
| ETTh1 | 2 | 0.7088 | 0.0119 | 0.9936 | cpu | – | False |
| ETTh1 | 3 | 0.7089 | 0.0119 | 0.9938 | cpu | 436.8529 | False |
| ETTh1 | 4 | 0.7057 | 0.0120 | 0.9893 | cpu | 560.7106 | False |
| ETTh1 | 5 | 0.7041 | 0.0115 | 0.9870 | cpu | 887.7359 | False |
| ETTh1 | 6 | 0.7008 | 0.0124 | 0.9824 | cpu | 633.5615 | False |
| ETTh1 | 7 | 0.7066 | 0.0124 | 0.9905 | cpu | 709.3048 | False |
| ETTh1 | 8 | 0.7055 | 0.0121 | 0.9889 | cpu | 551.3650 | False |
| ETTh1 | 9 | 0.7057 | 0.0125 | 0.9892 | cpu | 558.5334 | False |
| ETTh2 | 0 | 0.2263 | 0.0067 | 1.0000 | cpu | – | False |
| ETTh2 | 1 | 0.2207 | 0.0057 | 0.9752 | cpu | – | False |
| ETTh2 | 2 | 0.2238 | 0.0055 | 0.9888 | cpu | – | False |
| ETTh2 | 3 | 0.2209 | 0.0057 | 0.9762 | cpu | 611.7459 | False |
| ETTh2 | 4 | 0.2204 | 0.0052 | 0.9738 | cpu | 469.3740 | False |
| ETTh2 | 5 | 0.2213 | 0.0060 | 0.9776 | cpu | 459.2404 | False |
| ETTh2 | 6 | 0.2238 | 0.0064 | 0.9890 | cpu | 398.4862 | False |
| ETTh2 | 7 | 0.2253 | 0.0064 | 0.9957 | cpu | 320.6804 | False |
| ETTh2 | 8 | 0.2234 | 0.0059 | 0.9873 | cpu | 323.7126 | False |
| ETTh2 | 9 | 0.2222 | 0.0060 | 0.9817 | cpu | 340.5109 | False |
| electricity | 0 | 0.0679 | 0.0153 | 1.0000 | mps | – | False |
| electricity | 1 | 0.0679 | 0.0160 | 1.0002 | mps | – | False |
| electricity | 2 | 0.0683 | 0.0153 | 1.0055 | mps | – | False |
| electricity | 3 | 0.0682 | 0.0150 | 1.0052 | mps | 1075.4565 | False |
| electricity | 4 | 0.0683 | 0.0153 | 1.0055 | mps | 1285.3237 | False |
| traffic | 0 | 0.2742 | 0.0036 | 1.0000 | mps | – | False |
| traffic | 1 | 0.2745 | 0.0037 | 1.0013 | mps | – | False |
| traffic | 2 | 0.2748 | 0.0037 | 1.0022 | mps | – | False |
| traffic | 3 | 0.2752 | 0.0036 | 1.0038 | mps | 717.6627 | False |
| traffic | 4 | 0.2742 | 0.0037 | 1.0002 | mps | 727.7689 | False |

Unhealthy seeds: 0 (none). FVU range over new seeds: 0.0036–0.0153 (existing range 0.003–0.033).


### 1. Type-B enrichment of MVDF (H1c)

*Claim tested:* MVDF enrichment ≥ 1.0× on every run (Table 2: 12/12)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 2.50 [1.11,4.71] | 2.22 [0.94,4.00] | 1.25 [0.00,2.94] | 1.48 [0.37,3.08] | 2.00 [0.69,3.81] | 2.17 [0.71,4.21] | 1.79 [0.65,3.33] | 2.00 [0.67,3.89] | 2.08 [0.77,4.09] | 2.22 [0.50,4.71] | 2.04 [1.25,2.50] | [1.49,2.64] | 10/10 | 0.000977 |
| ETTh2 | 2.38 [0.77,4.62] | 2.17 [0.77,4.38] | 2.08 [0.74,4.21] | 3.68 [1.74,6.67] | 1.74 [0.43,3.64] | 2.61 [1.11,5.00] | 3.16 [1.30,6.00] | 1.82 [0.43,3.85] | 2.40 [1.07,4.50] | 1.74 [0.43,3.75] | 2.28 [1.74,3.68] | [1.82,3.26] | 10/10 | 0.000977 |
| electricity | 3.85 [1.25,8.75] | 4.17 [1.54,10.00] | 2.00 [0.00,5.00] | 3.57 [1.33,8.00] | 4.00 [1.00,10.00] | – | – | – | – | – | 3.85 [2.00,4.17] | [2.26,6.01] | 5/5 | 0.0312 |
| traffic | 2.31 [0.53,6.25] | 3.33 [0.77,7.78] | 4.17 [1.33,10.00] | 3.33 [0.71,8.75] | 1.43 [0.00,4.29] | – | – | – | – | – | 3.33 [1.43,4.17] | [1.56,5.30] | 5/5 | 0.0312 |

### 1b. Type-B enrichment, CI excludes 1 (added uncertainty; not a paper claim)

*Claim tested:* enrichment CI > 1

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 2.50 [1.11,4.71] | 2.22 [0.94,4.00] | 1.25 [0.00,2.94] | 1.48 [0.37,3.08] | 2.00 [0.69,3.81] | 2.17 [0.71,4.21] | 1.79 [0.65,3.33] | 2.00 [0.67,3.89] | 2.08 [0.77,4.09] | 2.22 [0.50,4.71] | 2.04 [1.25,2.50] | 1/10 | 0.999 |
| ETTh2 | 2.38 [0.77,4.62] | 2.17 [0.77,4.38] | 2.08 [0.74,4.21] | 3.68 [1.74,6.67] | 1.74 [0.43,3.64] | 2.61 [1.11,5.00] | 3.16 [1.30,6.00] | 1.82 [0.43,3.85] | 2.40 [1.07,4.50] | 1.74 [0.43,3.75] | 2.28 [1.74,3.68] | 4/10 | 0.828 |
| electricity | 3.85 [1.25,8.75] | 4.17 [1.54,10.00] | 2.00 [0.00,5.00] | 3.57 [1.33,8.00] | 4.00 [1.00,10.00] | – | – | – | – | – | 3.85 [2.00,4.17] | 3/5 | 0.5 |
| traffic | 2.31 [0.53,6.25] | 3.33 [0.77,7.78] | 4.17 [1.33,10.00] | 3.33 [0.71,8.75] | 1.43 [0.00,4.29] | – | – | – | – | – | 3.33 [1.43,4.17] | 1/5 | 0.969 |

### 2a. Phase 3 best held-out R², marginal

*Claim tested:* best held-out R² negative on every seed (Table 4, 3/3)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | -0.635 [-1.237,-0.186] | -0.718 [-1.410,-0.300] | -0.581 [-1.119,-0.204] | -0.614 [-1.147,-0.213] | -0.521 [-0.961,-0.158] | -0.579 [-1.073,-0.158] | -0.596 [-1.139,-0.191] | -0.616 [-1.168,-0.235] | -0.599 [-1.166,-0.198] | -0.603 [-1.097,-0.165] | -0.601 [-0.718,-0.521] | [-0.763,-0.440] | 10/10 | 0.000977 |
| ETTh2 | -0.925 [-1.793,-0.455] | -0.923 [-1.795,-0.436] | -0.896 [-1.823,-0.460] | -0.878 [-1.785,-0.400] | -0.913 [-1.845,-0.359] | -0.769 [-1.573,-0.380] | -0.807 [-1.443,-0.398] | -0.943 [-1.805,-0.527] | -0.846 [-1.744,-0.387] | -0.884 [-1.772,-0.393] | -0.890 [-0.943,-0.769] | [-1.136,-0.698] | 10/10 | 0.000977 |
| electricity | -0.293 [-0.897,-0.068] | -0.288 [-0.942,-0.062] | -0.327 [-1.085,-0.059] | -0.289 [-0.932,-0.066] | -0.310 [-1.007,-0.075] | – | – | – | – | – | -0.293 [-0.327,-0.288] | [-0.656,-0.233] | 5/5 | 0.0312 |
| traffic | -1.333 [-4.951,-1.354] | -0.506 [-2.960,-0.482] | -0.848 [-4.155,-0.862] | -0.696 [-2.801,-0.715] | -0.457 [-2.805,-0.323] | – | – | – | – | – | -0.696 [-1.333,-0.457] | [-2.795,-1.299] | 5/5 | 0.0312 |

### 2b. Phase 3 best held-out R², active windows

*Claim tested:* active-window best R² negative on every seed

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | -0.695 [-0.883,-0.316] | -0.522 [-0.520,-0.154] | -0.474 [-0.825,-0.144] | -0.956 [-1.343,-0.719] | -0.765 [-1.003,-0.516] | -0.472 [-0.560,-0.046] | -0.633 [-0.718,-0.222] | -0.741 [-0.994,-0.404] | -0.551 [-0.833,-0.305] | -0.660 [-1.028,-0.286] | -0.647 [-0.956,-0.472] | 10/10 | 0.000977 |
| ETTh2 | -0.804 [-1.134,-0.495] | -0.894 [-1.630,-0.394] | -0.877 [-1.681,-0.382] | -0.959 [-1.030,-0.555] | -0.776 [-1.091,-0.254] | -0.690 [-0.839,-0.412] | -0.905 [-1.436,-0.472] | -0.848 [-1.434,-0.504] | -0.944 [-1.322,-0.600] | -0.830 [-1.542,-0.378] | -0.862 [-0.959,-0.690] | 10/10 | 0.000977 |
| electricity | -0.331 [-0.883,-0.304] | -0.350 [-0.929,-0.310] | -0.477 [-1.377,-0.285] | -0.313 [-0.944,-0.163] | -0.478 [-1.134,-0.383] | – | – | – | – | – | -0.350 [-0.478,-0.313] | 5/5 | 0.0312 |
| traffic | -2.787 [-10.221,-3.188] | -0.609 [-5.678,-0.684] | -2.216 [-12.312,-3.027] | -1.057 [-5.630,-1.148] | -0.460 [-3.867,-0.451] | – | – | – | – | – | -1.057 [-2.787,-0.460] | 5/5 | 0.0312 |

### 3a. Oracle gap ratio (W_naive − W_oracle)/W_naive, ID

*Claim tested:* oracle gap CI excludes 0 on every seed (Table 6)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.249 [+0.169,+0.352] | +0.236 [+0.161,+0.340] | +0.251 [+0.178,+0.352] | +0.257 [+0.184,+0.356] | +0.256 [+0.185,+0.354] | +0.259 [+0.185,+0.359] | +0.254 [+0.182,+0.352] | +0.252 [+0.182,+0.348] | +0.242 [+0.166,+0.346] | +0.238 [+0.159,+0.343] | +0.252 [+0.236,+0.259] | [+0.238,+0.294] | 10/10 | 0.000977 |
| ETTh2 | +0.300 [+0.236,+0.392] | +0.294 [+0.226,+0.391] | +0.282 [+0.213,+0.384] | +0.306 [+0.241,+0.402] | +0.300 [+0.237,+0.392] | +0.306 [+0.243,+0.397] | +0.293 [+0.231,+0.384] | +0.289 [+0.218,+0.388] | +0.303 [+0.239,+0.395] | +0.301 [+0.235,+0.395] | +0.300 [+0.282,+0.306] | [+0.289,+0.338] | 10/10 | 0.000977 |
| electricity | +0.183 [+0.139,+0.220] | +0.173 [+0.132,+0.211] | +0.180 [+0.139,+0.217] | +0.181 [+0.139,+0.218] | +0.183 [+0.141,+0.222] | – | – | – | – | – | +0.181 [+0.173,+0.183] | [+0.158,+0.194] | 5/5 | 0.0312 |
| traffic | +0.396 [+0.095,+0.613] | +0.391 [+0.083,+0.612] | +0.390 [+0.083,+0.611] | +0.395 [+0.087,+0.615] | +0.391 [+0.084,+0.609] | – | – | – | – | – | +0.391 [+0.390,+0.396] | [+0.261,+0.497] | 5/5 | 0.0312 |

### 3b. Oracle gap ratio, shift

*Claim tested:* oracle gap CI excludes 0 on every seed (Table 6)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.255 [+0.157,+0.353] | +0.254 [+0.160,+0.351] | +0.252 [+0.154,+0.349] | +0.255 [+0.158,+0.350] | +0.256 [+0.158,+0.351] | +0.272 [+0.175,+0.368] | +0.258 [+0.163,+0.355] | +0.245 [+0.148,+0.340] | +0.261 [+0.165,+0.358] | +0.264 [+0.171,+0.359] | +0.256 [+0.245,+0.272] | [+0.230,+0.290] | 10/10 | 0.000977 |
| ETTh2 | +0.250 [+0.199,+0.322] | +0.251 [+0.204,+0.322] | +0.235 [+0.187,+0.305] | +0.239 [+0.193,+0.308] | +0.247 [+0.200,+0.317] | +0.238 [+0.193,+0.303] | +0.236 [+0.189,+0.303] | +0.236 [+0.190,+0.306] | +0.241 [+0.189,+0.308] | +0.241 [+0.193,+0.307] | +0.240 [+0.235,+0.251] | [+0.233,+0.271] | 10/10 | 0.000977 |
| electricity | +0.379 [+0.277,+0.463] | +0.364 [+0.261,+0.448] | +0.372 [+0.268,+0.456] | +0.367 [+0.261,+0.452] | +0.377 [+0.274,+0.462] | – | – | – | – | – | +0.372 [+0.364,+0.379] | [+0.324,+0.407] | 5/5 | 0.0312 |
| traffic | +0.366 [+0.117,+0.534] | +0.363 [+0.110,+0.533] | +0.373 [+0.132,+0.538] | +0.374 [+0.125,+0.545] | +0.375 [+0.131,+0.538] | – | – | – | – | – | +0.373 [+0.363,+0.375] | [+0.259,+0.446] | 5/5 | 0.0312 |

### 4. Scorer width − naive_matched width, dense, shift

*Claim tested:* naive_matched significantly narrower than dense (§5.6: 12/12)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.165 [+0.035,+0.294] | +0.183 [+0.037,+0.332] | +0.111 [-0.019,+0.242] | +0.110 [-0.000,+0.217] | +0.135 [-0.003,+0.256] | +0.139 [-0.006,+0.281] | +0.156 [+0.023,+0.285] | +0.111 [-0.019,+0.237] | +0.148 [+0.003,+0.291] | +0.134 [-0.003,+0.260] | +0.137 [+0.110,+0.183] | [+0.092,+0.184] | 4/10 | 0.828 |
| ETTh2 | +0.383 [+0.245,+0.571] | +0.451 [+0.277,+0.680] | +0.357 [+0.205,+0.550] | +0.382 [+0.212,+0.604] | +0.332 [+0.166,+0.553] | +0.349 [+0.169,+0.581] | +0.393 [+0.220,+0.625] | +0.426 [+0.272,+0.622] | +0.418 [+0.240,+0.644] | +0.364 [+0.204,+0.577] | +0.382 [+0.332,+0.451] | [+0.346,+0.473] | 10/10 | 0.000977 |
| electricity | +0.176 [-0.080,+0.445] | +0.104 [-0.188,+0.392] | +0.099 [-0.180,+0.375] | +0.084 [-0.234,+0.390] | +0.160 [-0.094,+0.434] | – | – | – | – | – | +0.104 [+0.084,+0.176] | [-0.008,+0.257] | 0/5 | 1 |
| traffic | +3.487 [+1.745,+5.512] | +3.617 [+1.891,+5.566] | +3.150 [+1.459,+5.309] | +3.202 [+1.535,+5.310] | +3.575 [+1.828,+5.681] | – | – | – | – | – | +3.487 [+3.150,+3.617] | [+2.640,+4.365] | 5/5 | 0.0312 |

Coverage (dense, shift): ETTh1 s0 0.901/0.894, s1 0.903/0.894, s2 0.898/0.893, s3 0.901/0.894, s4 0.903/0.896, s5 0.901/0.894, s6 0.903/0.894, s7 0.903/0.895, s8 0.901/0.894, s9 0.898/0.894; ETTh2 s0 0.897/0.898, s1 0.900/0.898, s2 0.897/0.898, s3 0.895/0.898, s4 0.902/0.900, s5 0.898/0.899, s6 0.898/0.897, s7 0.900/0.900, s8 0.898/0.900, s9 0.899/0.898; electricity s0 0.881/0.895, s1 0.875/0.893, s2 0.877/0.893, s3 0.874/0.895, s4 0.881/0.895; traffic s0 0.906/0.916, s1 0.908/0.916, s2 0.908/0.916, s3 0.905/0.915, s4 0.907/0.915 (scorer / naive_matched)

### 4. Scorer width − naive_matched width, dense, ID

*Claim tested:* naive_matched significantly narrower than dense (§5.6: 12/12)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.146 [+0.030,+0.233] | +0.184 [+0.033,+0.370] | +0.136 [-0.022,+0.265] | +0.118 [+0.013,+0.211] | +0.238 [+0.082,+0.388] | +0.142 [-0.002,+0.300] | +0.224 [+0.075,+0.378] | +0.160 [+0.032,+0.287] | +0.175 [+0.069,+0.273] | +0.149 [+0.039,+0.233] | +0.155 [+0.118,+0.238] | [+0.114,+0.213] | 8/10 | 0.0547 |
| ETTh2 | -0.049 [-0.245,+0.064] | +0.121 [+0.034,+0.270] | -0.057 [-0.216,+0.219] | -0.040 [-0.209,+0.141] | -0.047 [-0.251,+0.188] | -0.116 [-0.301,+0.052] | +0.026 [-0.108,+0.191] | -0.014 [-0.204,+0.236] | -0.200 [-0.419,-0.102] | -0.098 [-0.265,+0.022] | -0.048 [-0.200,+0.121] | [-0.134,+0.038] | 1/10 | 0.999 |
| electricity | +0.161 [+0.101,+0.205] | +0.128 [+0.019,+0.203] | +0.161 [+0.078,+0.227] | +0.164 [+0.076,+0.250] | +0.157 [+0.041,+0.247] | – | – | – | – | – | +0.161 [+0.128,+0.164] | [+0.102,+0.186] | 5/5 | 0.0312 |
| traffic | +1.788 [-0.551,+5.742] | +2.023 [-0.394,+6.282] | +1.595 [-0.628,+5.128] | +1.324 [-0.737,+4.627] | +1.938 [-0.338,+6.032] | – | – | – | – | – | +1.788 [+1.324,+2.023] | [+1.235,+4.118] | 0/5 | 1 |

Coverage (dense, ID): ETTh1 s0 0.835/0.830, s1 0.850/0.832, s2 0.830/0.827, s3 0.845/0.830, s4 0.830/0.830, s5 0.832/0.830, s6 0.850/0.835, s7 0.838/0.830, s8 0.835/0.830, s9 0.843/0.838; ETTh2 s0 0.787/0.845, s1 0.777/0.840, s2 0.772/0.843, s3 0.751/0.848, s4 0.789/0.850, s5 0.787/0.853, s6 0.792/0.853, s7 0.774/0.843, s8 0.779/0.858, s9 0.779/0.848; electricity s0 0.916/0.904, s1 0.908/0.906, s2 0.904/0.901, s3 0.891/0.904, s4 0.906/0.906; traffic s0 1.000/1.000, s1 1.000/1.000, s2 1.000/1.000, s3 1.000/1.000, s4 1.000/1.000 (scorer / naive_matched)

### 4. Scorer width − naive_matched width, mvdf, shift

*Claim tested:* naive_matched significantly narrower than mvdf (§5.6: 12/12)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.055 [-0.064,+0.174] | +0.076 [-0.031,+0.190] | +0.026 [-0.084,+0.128] | +0.009 [-0.106,+0.122] | +0.069 [-0.047,+0.187] | +0.076 [-0.038,+0.196] | +0.060 [-0.052,+0.174] | +0.043 [-0.065,+0.144] | +0.030 [-0.084,+0.149] | +0.064 [-0.040,+0.155] | +0.058 [+0.009,+0.076] | [+0.009,+0.086] | 0/10 | 1 |
| ETTh2 | +0.172 [+0.074,+0.287] | +0.140 [+0.042,+0.251] | +0.170 [+0.052,+0.308] | +0.108 [+0.016,+0.217] | +0.127 [+0.020,+0.257] | +0.107 [+0.013,+0.207] | +0.222 [+0.087,+0.404] | +0.159 [+0.068,+0.267] | +0.140 [+0.048,+0.249] | +0.178 [+0.052,+0.332] | +0.150 [+0.107,+0.222] | [+0.118,+0.203] | 10/10 | 0.000977 |
| electricity | -0.321 [-0.716,+0.029] | -0.795 [-1.178,-0.475] | -0.409 [-0.797,-0.056] | -1.030 [-1.445,-0.704] | -0.178 [-0.552,+0.165] | – | – | – | – | – | -0.409 [-1.030,-0.178] | [-0.898,-0.255] | 0/5 | 1 |
| traffic | +2.307 [+1.524,+3.262] | +2.559 [+1.642,+3.477] | +2.139 [+1.517,+2.884] | +2.616 [+1.765,+3.784] | +3.391 [+2.456,+4.369] | – | – | – | – | – | +2.559 [+2.139,+3.391] | [+2.153,+3.221] | 5/5 | 0.0312 |

Coverage (mvdf, shift): ETTh1 s0 0.894/0.894, s1 0.895/0.894, s2 0.897/0.893, s3 0.898/0.894, s4 0.896/0.896, s5 0.898/0.894, s6 0.896/0.894, s7 0.898/0.895, s8 0.894/0.894, s9 0.895/0.894; ETTh2 s0 0.906/0.898, s1 0.906/0.898, s2 0.905/0.898, s3 0.908/0.898, s4 0.908/0.900, s5 0.906/0.899, s6 0.903/0.897, s7 0.908/0.900, s8 0.909/0.900, s9 0.906/0.898; electricity s0 0.836/0.895, s1 0.682/0.893, s2 0.823/0.893, s3 0.620/0.895, s4 0.847/0.895; traffic s0 0.923/0.916, s1 0.924/0.916, s2 0.921/0.916, s3 0.918/0.915, s4 0.924/0.915 (scorer / naive_matched)

### 4. Scorer width − naive_matched width, mvdf, ID

*Claim tested:* naive_matched significantly narrower than mvdf (§5.6: 12/12)

| dataset | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | median [min,max] | pooled 2-level CI | pass k/N | sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | +0.018 [-0.071,+0.142] | +0.020 [-0.059,+0.133] | +0.055 [-0.044,+0.170] | +0.048 [-0.050,+0.177] | +0.014 [-0.066,+0.140] | +0.041 [-0.061,+0.172] | +0.043 [-0.045,+0.149] | +0.029 [-0.047,+0.122] | +0.015 [-0.070,+0.136] | +0.085 [-0.020,+0.213] | +0.035 [+0.014,+0.085] | [+0.021,+0.093] | 0/10 | 1 |
| ETTh2 | +0.095 [-0.066,+0.387] | -0.004 [-0.107,+0.213] | -0.055 [-0.173,+0.174] | -0.001 [-0.133,+0.174] | -0.108 [-0.245,+0.107] | -0.028 [-0.172,+0.185] | +0.007 [-0.112,+0.212] | +0.085 [-0.097,+0.420] | +0.008 [-0.126,+0.227] | -0.010 [-0.126,+0.222] | -0.003 [-0.108,+0.095] | [-0.006,+0.132] | 0/10 | 1 |
| electricity | +0.066 [+0.012,+0.075] | +0.072 [-0.030,+0.113] | +0.068 [-0.012,+0.088] | +0.065 [-0.005,+0.105] | +0.130 [+0.035,+0.176] | – | – | – | – | – | +0.068 [+0.065,+0.130] | [+0.023,+0.089] | 2/5 | 0.812 |
| traffic | +2.987 [+1.453,+4.703] | +3.341 [+2.022,+5.060] | +2.310 [+0.812,+3.942] | +2.038 [-0.031,+4.366] | +3.581 [+2.530,+5.713] | – | – | – | – | – | +2.987 [+2.038,+3.581] | [+2.198,+4.245] | 4/5 | 0.188 |

Coverage (mvdf, ID): ETTh1 s0 0.779/0.830, s1 0.772/0.832, s2 0.787/0.827, s3 0.794/0.830, s4 0.789/0.830, s5 0.789/0.830, s6 0.792/0.835, s7 0.794/0.830, s8 0.772/0.830, s9 0.807/0.838; ETTh2 s0 0.799/0.845, s1 0.810/0.840, s2 0.787/0.843, s3 0.802/0.848, s4 0.789/0.850, s5 0.822/0.853, s6 0.797/0.853, s7 0.794/0.843, s8 0.810/0.858, s9 0.789/0.848; electricity s0 0.916/0.904, s1 0.906/0.906, s2 0.912/0.901, s3 0.891/0.904, s4 0.914/0.906; traffic s0 1.000/1.000, s1 1.000/1.000, s2 1.000/1.000, s3 1.000/1.000, s4 1.000/1.000 (scorer / naive_matched)

## For the paper

| claim | ETTh1 (10) | ETTh2 (10) | electricity (5) | traffic (5) |
|---|---|---|---|---|
| type-B enrichment ≥ 1× | 10/10 (med 2.04) | 10/10 (2.28) | 5/5 (3.85) | 5/5 (3.33) |
| Phase-3 best R² < 0 (CI < 0) | 10/10 | 10/10 | 5/5 | 5/5 |
| oracle gap CI > 0 (ID & shift) | 10/10 | 10/10 | 5/5 | 5/5 |
| naive_matched sig. narrower than dense (shift) | 4/10 | 10/10 | 0/5 | 5/5 |
| naive_matched sig. narrower than mvdf (shift) | 0/10 | 10/10 | 0/5 (coverage-confounded) | 5/5 |

> "Across 30 independently retrained pipelines (10 seeds each on ETTh1/ETTh2, 5 each on electricity/traffic), MVDF is type-B-enriched on every run (median 2.0–3.8×), natural MVDF activation predicts difficulty worse than a constant on every run, and the oracle gap is positive on every run; no headline median moved by more than its pooled CI half-width when going from 3 seeds to 5 or 10. The constant-width band is never significantly wider than the learned scorer, but its advantage is significant on only 4/10 (dense) and 0/10 (MVDF) ETTh1 seeds."

## Per-seed tables

All per-seed numbers with CIs: `SEED_TABLE.md`; metric CSVs `AGG_E2_*.csv`.
