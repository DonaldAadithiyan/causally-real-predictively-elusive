"""Fetch the datasets into data/raw/ and verify SHA-256 against the copies that produced every
committed result. Fails loudly (exit 1) on any mismatch. No data is redistributed by this
repository.

ETTh1 / ETTh2 are downloaded from the public ETDataset repository (byte-identical, verified).
electricity.csv / traffic.csv are the preprocessed files of the Kaggle data bundle the paper's
models were trained on (370 clients x 35 000 hourly rows; 862 sensors x 17 544 hourly rows).
No public URL serves them byte-identically, so they must be placed in data/raw/ by hand; this
script then checks their hashes. See docs/PROVENANCE.md.

    python scripts/download_data.py            # ETT only + check whatever is present
    python scripts/download_data.py --require-all
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
EXPECTED = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
    "electricity": "353cbeb1d1ec92f0f08851a1c95ef0be4658d120176a0dde478a659bf48b1570",
    "traffic": "1e7b7d9007543c73479ed44b84535c9374d7449e23fc5ea902a9db69930d9281",
}
MANUAL = {"electricity", "traffic"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-all", action="store_true",
                    help="fail if electricity/traffic are not present (needed for E1/E2/E3 on those datasets)")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    ok = True
    for name, exp in EXPECTED.items():
        p = os.path.join(RAW, f"{name}.csv")
        if not os.path.exists(p) and name not in MANUAL:
            print(f"[download] {name} <- {ETT_BASE}{name}.csv")
            urllib.request.urlretrieve(f"{ETT_BASE}{name}.csv", p)
        if not os.path.exists(p):
            msg = (f"[missing ] {name}.csv: place the preprocessed bundle file at data/raw/{name}.csv "
                   f"(expected sha256 {exp}; see docs/PROVENANCE.md)")
            print(msg)
            ok = ok and not a.require_all
            continue
        got = sha256(p)
        if got == exp:
            print(f"[ok      ] {name}.csv sha256 {got}")
        else:
            print(f"[MISMATCH] {name}.csv sha256 {got} != expected {exp}")
            ok = False
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
