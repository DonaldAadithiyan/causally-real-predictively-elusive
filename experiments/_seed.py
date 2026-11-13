"""Deterministic per-(seed, dataset) seed derivation.

Both phase scripts previously seeded the global RNG ONCE at the top and then looped
over datasets, so `ETTh1` as the 1st dataset drew different random numbers than
`ETTh1` run alone (earlier datasets advanced the RNG). That made a local single-
dataset re-run non-comparable to a full-suite run at the "same" seed — the exact
thing that produced a spurious I(correlated) discrepancy on ETTh1.

`derive_seed` gives a stable 32-bit seed from (base_seed, dataset_name) using a hash
that does NOT depend on PYTHONHASHSEED.
"""
from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, dataset: str) -> int:
    h = hashlib.sha256(f"{base_seed}::{dataset}".encode()).digest()
    return int.from_bytes(h[:4], "big")
