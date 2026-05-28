#!/usr/bin/env python3
"""Download Qdrant/all-MiniLM-L6-v2-onnx (the same model fastembed uses).

It's fp32 with graph-level optimizations applied — letting us measure ort
against the *exact* file fastembed ships, isolating wrapper overhead from
model differences. Idempotent.
"""
import shutil
from pathlib import Path


OUT = Path("models/all-MiniLM-L6-v2-qdrant")
CACHE_GUESS = Path(".fastembed_cache/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots")


def from_fastembed_cache() -> bool:
    if not CACHE_GUESS.is_dir():
        return False
    snapshots = list(CACHE_GUESS.iterdir())
    if not snapshots:
        return False
    src = snapshots[0]
    if not (src / "model.onnx").exists():
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "tokenizer.json", "config.json",
              "tokenizer_config.json", "special_tokens_map.json"):
        sp = src / f
        if sp.exists():
            shutil.copy(sp, OUT / f)
    return True


def from_hub() -> bool:
    from huggingface_hub import snapshot_download
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="Qdrant/all-MiniLM-L6-v2-onnx", local_dir=str(OUT))
    return True


def main():
    if (OUT / "model.onnx").exists():
        print(f"qdrant model already at {OUT}/model.onnx — skipping")
        return
    if from_fastembed_cache():
        print(f"copied from .fastembed_cache -> {OUT}")
        return
    from_hub()
    print(f"downloaded to {OUT}")


if __name__ == "__main__":
    main()
