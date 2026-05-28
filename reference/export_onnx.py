#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX + bundled tokenizer for ort_runner.

Idempotent: skips if model.onnx already exists at the target path.
"""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/all-MiniLM-L6-v2")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    out = Path(args.out)
    if (out / "model.onnx").exists():
        print(f"ONNX already at {out}/model.onnx — skipping")
        return

    out.mkdir(parents=True, exist_ok=True)
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    print(f"exporting {args.model} -> {out}")
    m = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    t = AutoTokenizer.from_pretrained(args.model)
    m.save_pretrained(str(out))
    t.save_pretrained(str(out))
    print("done")


if __name__ == "__main__":
    main()
