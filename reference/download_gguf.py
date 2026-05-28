#!/usr/bin/env python3
"""Fetch the F16 GGUF for all-MiniLM-L6-v2 so llama_runner can load it.

Strategy:
  1. If models/all-MiniLM-L6-v2-gguf/model.gguf already exists -> done.
  2. If a local Ollama daemon has 'all-minilm' cached -> copy that blob
     (matches what `ollama` itself serves, so the comparison is apples-to-apples).
  3. Otherwise -> download from Hugging Face Hub.

Idempotent.
"""
import json
import shutil
import sys
from pathlib import Path


OUT = Path("models/all-MiniLM-L6-v2-gguf")
TARGET = OUT / "model.gguf"

OLLAMA_MANIFESTS = Path.home() / ".ollama/models/manifests"
OLLAMA_BLOBS = Path.home() / ".ollama/models/blobs"

HUB_REPO = "leliuga/all-MiniLM-L6-v2-GGUF"
HUB_FILE = "all-MiniLM-L6-v2.F16.gguf"


def from_ollama_cache() -> bool:
    if not OLLAMA_MANIFESTS.is_dir():
        return False
    for manifest in OLLAMA_MANIFESTS.rglob("*"):
        if not manifest.is_file():
            continue
        if "all-minilm" not in str(manifest):
            continue
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue
        for layer in data.get("layers", []):
            mt = layer.get("mediaType", "")
            if "gguf" not in mt and "model" not in mt:
                continue
            digest = layer.get("digest", "").split(":", 1)[-1]
            blob = OLLAMA_BLOBS / f"sha256-{digest}"
            if not blob.exists():
                continue
            try:
                with open(blob, "rb") as f:
                    if f.read(4) != b"GGUF":
                        continue
            except OSError:
                continue
            OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy(blob, TARGET)
            return True
    return False


def from_hub() -> bool:
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=HUB_REPO, filename=HUB_FILE, local_dir=str(OUT))
    p = Path(path)
    if p.name != "model.gguf":
        shutil.move(str(p), TARGET)
    return True


def main():
    if TARGET.exists():
        print(f"GGUF already at {TARGET} — skipping")
        return
    if from_ollama_cache():
        print(f"copied GGUF from Ollama cache -> {TARGET}")
        return
    try:
        from_hub()
        print(f"downloaded GGUF from {HUB_REPO} -> {TARGET}")
    except Exception as e:
        print(f"failed to obtain GGUF: {e}", file=sys.stderr)
        print("llama backend will be unavailable. Install Ollama and `ollama pull all-minilm`, "
              "or fix Hub access and re-run.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
