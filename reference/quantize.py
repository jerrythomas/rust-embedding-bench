#!/usr/bin/env python3
"""Dynamic int8 quantization of the fp32 ONNX model for an apples-to-apples
comparison with fastembed (which ships int8 internally).

Idempotent: skips if the int8 model already exists.
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-dir", default="models/all-MiniLM-L6-v2")
    p.add_argument("--out-dir", default="models/all-MiniLM-L6-v2-int8")
    args = p.parse_args()

    src = Path(args.fp32_dir)
    out = Path(args.out_dir)

    if (out / "model.onnx").exists():
        print(f"int8 model already at {out}/model.onnx — skipping")
        return
    if not (src / "model.onnx").exists():
        raise SystemExit(f"fp32 model not found at {src}/model.onnx; run export_onnx.py first")

    out.mkdir(parents=True, exist_ok=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"quantizing {src}/model.onnx -> {out}/model.onnx (dynamic int8)")
    quantize_dynamic(
        str(src / "model.onnx"),
        str(out / "model.onnx"),
        weight_type=QuantType.QInt8,
    )

    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
              "special_tokens_map.json", "config.json"):
        s = src / f
        if s.exists():
            shutil.copy(s, out / f)

    print("done")


if __name__ == "__main__":
    main()
