# Verification record

These are the checks a continuous-integration run would perform. They are recorded here from a
**fresh clone of the published repository in a new virtual environment**, so they can be audited
without GitHub Actions. The workflow in `.github/workflows/ci.yml` runs the same checks on demand
(Actions tab → "ci" → "Run workflow"); it is manual-only.

## How to reproduce this record

```bash
git clone https://github.com/DonaldAadithiyan/causally-real-predictively-elusive && cd causally-real-predictively-elusive
python -m venv .venv && source .venv/bin/activate && make install
python scripts/download_data.py
make smoke
python -m pytest -q
make tables && make figures && git status --porcelain paper/   # empty = byte-identical to committed
```

## Result (release `v1.0-camera-ready`)

```
commit: 2e9ce01 (v1.0-camera-ready)
date: 2026-11-16 21:36 +0530
python: 3.12.11 torch 2.6.0 numpy 2.4.4 sklearn 1.5.2 pandas 3.0.2
platform: Darwin arm64
--- python scripts/download_data.py
[download] ETTh1 <- https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
[ok      ] ETTh1.csv sha256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066
[download] ETTh2 <- https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
[ok      ] ETTh2.csv sha256 a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b
[missing ] electricity.csv: place the preprocessed bundle file at data/raw/electricity.csv (expected sha256 353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570; see docs/PROVENANCE.md)
[missing ] traffic.csv: place the preprocessed bundle file at data/raw/traffic.csv (expected sha256 1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281; see docs/PROVENANCE.md)
--- make smoke
[smoke] forecaster                         ok  (    2s)
[smoke] SAE                                ok  (    2s)
[smoke] dose-response + MVDF selection     ok  (    3s)
[smoke] per-window arrays                  ok  (    3s)
[smoke] Phase 3 battery (best R2 -1.50)    ok  (    4s)
[smoke] Phase 4 oracle (gap 26%)           ok  (    4s)
[smoke] learned scorer vs naive_matched    ok  (    4s)
[smoke] E3 cell (F0, F2) + leakage check   ok  (   22s)
[smoke] PASS in 22s
--- python -m pytest -q
121 passed, 2 skipped, 3 xfailed in 24.02s
--- make tables && make figures; git status --porcelain paper/
(empty = regenerated outputs byte-identical to committed)
```

## How to read it

- **121 passed** — every number printed in the paper that is recomputed from `results/` matches
  (`paper/paper_numbers.yaml`), the configuration constants match the paper, the leakage rule holds
  on synthetic data (and a deliberately leaky feature is caught), and the end-to-end smoke run passes.
- **3 xfailed** — the three known mismatches between the camera-ready text and the results
  (`docs/KNOWN_ISSUES.md` §A). They are *strict* expected failures: if either the text or the result
  changes, the test fails until `paper/paper_numbers.yaml` is updated.
- **2 skipped** — the real-data leakage tests, which need the large artifacts
  (`python scripts/fetch_artifacts.py`). With the artifacts in place they pass (123 passed, 3 xfailed).
- **electricity/traffic "missing"** — expected in a fresh clone: these files are placed by hand
  (`docs/PROVENANCE.md`). With them in place, `scripts/download_data.py --require-all` reports all four
  hashes as ok.
- **Release assets** — `scripts/fetch_artifacts.py` downloaded `artifacts_round2_p126.tar` from the
  GitHub release and verified all 44 files against `artifacts/MANIFEST.sha256` (ok 44, missing 0,
  mismatched 0).
