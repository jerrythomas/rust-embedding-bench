#!/usr/bin/env python3
"""Generate reference embeddings from Python sentence-transformers (fp32).

Output is consumed by analyze/correctness.py to verify that each Rust runner
produces vectors equivalent to the reference implementation.

Usage:
    pip install -r reference/requirements.txt
    python reference/generate_reference.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus/sentences.json")
    p.add_argument("--out", default="reference/reference_vectors.npz")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    model = SentenceTransformer(args.model)

    out = {}
    for bucket, sentences in corpus["buckets"].items():
        vecs = model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out[bucket] = vecs.astype(np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
