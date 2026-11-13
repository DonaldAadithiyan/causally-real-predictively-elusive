"""Print what CSV/text files are visible under KAGGLE_DATA_DIR and which logical
datasets the adapters can build from them. Run this before the main experiment to
confirm the attachment resolved."""
from __future__ import annotations

import os

from . import datasets as DS


def probe():
    root = DS.KAGGLE_DATA_DIR
    print(f"KAGGLE_DATA_DIR = {root}")
    if os.path.isdir(root):
        for dp, _, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith((".csv", ".txt")):
                    p = os.path.join(dp, f)
                    print(f"  {p}   ({os.path.getsize(p) // 1024} KB)")
    else:
        print("  (not a directory - nothing attached)")

    print("\nResolvable datasets:")
    for name in list(DS.LOADERS) + [k for k in DS.DATASET_URLS if k not in DS.LOADERS]:
        try:
            df = DS.LOADERS[name]() if name in DS.LOADERS else None
            if df is not None:
                print(f"  {name:12s} OK   {len(df)} rows x {df.shape[1]-1} channels")
            else:
                print(f"  {name:12s} OK   (HTTP)")
        except Exception as e:
            print(f"  {name:12s} --   {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    probe()
