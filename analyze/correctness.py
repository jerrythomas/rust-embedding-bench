#!/usr/bin/env python3
"""Compare a runner's saved vectors against the reference (cosine similarity).

Each runner accepts --save-vectors PATH; that file is a flat f32 little-endian
binary, vectors-in-order. Compare with:

    python analyze/correctness.py --vectors out/fastembed_short.bin --bucket short

Flags any vector whose cosine vs. the Python reference is below 0.999.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


DIM = 384  # all-MiniLM-L6-v2


def load_bin(path, dim=DIM):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % dim != 0:
        raise ValueError(f"{path}: {raw.size} floats not divisible by {dim}")
    return raw.reshape(-1, dim)


def cosine(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="reference/reference_vectors.npz")
    p.add_argument("--vectors", required=True)
    p.add_argument("--bucket", required=True, choices=["short", "medium", "long"])
    p.add_argument("--threshold", type=float, default=0.999)
    args = p.parse_args()

    ref_npz = np.load(args.reference)
    if args.bucket not in ref_npz.files:
        print(f"bucket {args.bucket} not in {args.reference}", file=sys.stderr)
        sys.exit(1)
    ref = ref_npz[args.bucket]
    got = load_bin(args.vectors)

    n = min(len(ref), len(got))
    if n == 0:
        print("no vectors to compare", file=sys.stderr)
        sys.exit(1)
    sims = cosine(ref[:n], got[:n])

    print(f"compared {n} vectors (bucket={args.bucket})")
    print(f"  min cosine: {sims.min():.6f}")
    print(f"  mean cosine: {sims.mean():.6f}")
    below = int((sims < args.threshold).sum())
    print(f"  below {args.threshold}: {below}")
    sys.exit(0 if below == 0 else 2)


if __name__ == "__main__":
    main()
