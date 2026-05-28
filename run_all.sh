#!/usr/bin/env bash
# Full benchmark cycle for minilm-bench. Idempotent: re-run safely.
#
# Stages (each skipped if its artifact is already in place):
#   1. uv venv .venv
#   2. install Python deps (sentence-transformers, optimum, numpy)
#   3. ollama pull all-minilm (if daemon reachable)
#   4. cargo build --release --workspace
#   5. optimum export to models/all-MiniLM-L6-v2/model.onnx
#   6. generate reference/reference_vectors.npz
#   7. sweep backend x length x batch x threads
#   8. correctness pass: one --save-vectors run per backend, cosine vs reference
#   9. aggregate results with analyze/compare.py
#
# Env overrides (space-separated):
#   SKIP="ort ollama"   skip whole backends from build/sweep/correctness
#   LENGTHS="short"     length buckets to sweep (default: short medium long)
#   BATCHES="1 8"       batch sizes (default: 1 8 32)
#   THREADS="1"         thread configs (default: 1)
#   WARMUP=50           warmup embeddings per run
#   MEASURE=500         measured embeddings per run

set -euo pipefail
cd "$(dirname "$0")"

UV="${UV:-uv}"
PY=".venv/bin/python"
SKIP="${SKIP:-}"
LENGTHS="${LENGTHS:-short medium long}"
BATCHES="${BATCHES:-1 8 32}"
THREADS="${THREADS:-1}"
WARMUP="${WARMUP:-50}"
MEASURE="${MEASURE:-500}"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

mkdir -p results vectors

skip() { [[ " $SKIP " == *" $1 "* ]]; }

# ---------- 1. uv venv ----------
if [[ ! -x "$PY" ]]; then
    echo ">> creating .venv"
    "$UV" venv --python 3.12 .venv
fi

# ---------- 2. Python deps ----------
if [[ ! -f .venv/.deps_ok ]]; then
    echo ">> installing Python deps (this can take a couple of minutes the first time)"
    "$UV" pip install -p "$PY" \
        -r reference/requirements.txt \
        -r analyze/requirements.txt \
        "optimum[onnxruntime]" onnx
    touch .venv/.deps_ok
fi

# ---------- 3. ollama model ----------
if ! skip ollama; then
    if curl -sf -m 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        if curl -sf -m 5 "$OLLAMA_URL/api/tags" | grep -q '"all-minilm'; then
            echo ">> ollama all-minilm present"
        else
            echo ">> pulling ollama all-minilm"
            curl -s -X POST "$OLLAMA_URL/api/pull" -d '{"model":"all-minilm"}' >/dev/null \
                || echo "   pull failed, ollama backend will be skipped"
        fi
    else
        echo ">> ollama daemon not reachable at $OLLAMA_URL; skipping pull"
    fi
fi

# ---------- 4. cargo build ----------
echo ">> cargo build --release"
cargo build --release --workspace

# ---------- 5. ONNX export + int8 quantization ----------
if ! skip ort && [[ ! -f models/all-MiniLM-L6-v2/model.onnx ]]; then
    echo ">> exporting ONNX model (one-time)"
    "$PY" reference/export_onnx.py
fi
if ! skip ort && [[ ! -f models/all-MiniLM-L6-v2-int8/model.onnx ]]; then
    echo ">> quantizing ONNX model to int8 (one-time)"
    "$PY" reference/quantize.py
fi
if ! skip ort && [[ ! -f models/all-MiniLM-L6-v2-qdrant/model.onnx ]]; then
    echo ">> fetching Qdrant pre-optimized fp32 ONNX (one-time)"
    "$PY" reference/download_qdrant.py
fi

# ---------- 6. reference vectors ----------
if [[ ! -f reference/reference_vectors.npz ]]; then
    echo ">> generating Python reference vectors"
    "$PY" reference/generate_reference.py
fi

# ---------- 7. sweep ----------
BACKENDS=(fastembed candle ollama ort llama)
HAS_INT8=0
HAS_QDRANT=0
[[ -f models/all-MiniLM-L6-v2-int8/model.onnx ]] && HAS_INT8=1
[[ -f models/all-MiniLM-L6-v2-qdrant/model.onnx ]] && HAS_QDRANT=1
for backend in "${BACKENDS[@]}"; do
    if skip "$backend"; then
        echo "skip $backend (SKIP env)"
        continue
    fi
    bin="target/release/${backend}_runner"
    [[ -x "$bin" ]] || { echo "no binary for $backend; skipping"; continue; }
    for length in $LENGTHS; do
        for batch in $BATCHES; do
            for threads in $THREADS; do
                out="results/${backend}_${length}_b${batch}_t${threads}.json"
                echo ">> sweep $backend length=$length batch=$batch threads=$threads"
                if ! RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" "$bin" \
                    --length "$length" \
                    --batch "$batch" \
                    --threads "$threads" \
                    --warmup "$WARMUP" \
                    --measure "$MEASURE" \
                    --out "$out" 2> >(grep -vE "init: embeddings|compute buffer|matches expectation|ggml_metal|~llama_context" >&2); then
                    echo "   FAILED, continuing"
                fi
                if [[ "$backend" == "ort" && "$HAS_INT8" == "1" ]]; then
                    out_i8="results/ort-int8_${length}_b${batch}_t${threads}.json"
                    echo ">> sweep ort-int8 length=$length batch=$batch threads=$threads"
                    RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" "$bin" \
                        --length "$length" \
                        --batch "$batch" \
                        --threads "$threads" \
                        --warmup "$WARMUP" \
                        --measure "$MEASURE" \
                        --out "$out_i8" \
                        --model models/all-MiniLM-L6-v2-int8/model.onnx \
                        --tokenizer models/all-MiniLM-L6-v2-int8/tokenizer.json \
                        --precision int8 \
                        --backend-label ort-int8 \
                    || echo "   FAILED, continuing"
                fi
                if [[ "$backend" == "ort" && "$HAS_QDRANT" == "1" ]]; then
                    out_q="results/ort-qdrant_${length}_b${batch}_t${threads}.json"
                    echo ">> sweep ort-qdrant length=$length batch=$batch threads=$threads"
                    RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" "$bin" \
                        --length "$length" \
                        --batch "$batch" \
                        --threads "$threads" \
                        --warmup "$WARMUP" \
                        --measure "$MEASURE" \
                        --out "$out_q" \
                        --model models/all-MiniLM-L6-v2-qdrant/model.onnx \
                        --tokenizer models/all-MiniLM-L6-v2-qdrant/tokenizer.json \
                        --precision fp32 \
                        --backend-label ort-qdrant \
                    || echo "   FAILED, continuing"
                fi
            done
        done
    done
done

# ---------- 8. correctness ----------
echo
echo ">> correctness pass (cosine vs Python reference)"
for backend in "${BACKENDS[@]}"; do
    if skip "$backend"; then continue; fi
    bin="target/release/${backend}_runner"
    [[ -x "$bin" ]] || continue
    vec="vectors/${backend}_short.bin"
    "$bin" --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
        --out "results/_correctness_${backend}.json" --save-vectors "$vec" >/dev/null
    printf "  %-10s " "$backend"
    "$PY" analyze/correctness.py --vectors "$vec" --bucket short | tail -3 | tr '\n' ' '
    echo
done
if [[ "$HAS_INT8" == "1" ]] && ! skip ort; then
    vec="vectors/ort-int8_short.bin"
    target/release/ort_runner --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
        --out "results/_correctness_ort-int8.json" --save-vectors "$vec" \
        --model models/all-MiniLM-L6-v2-int8/model.onnx \
        --tokenizer models/all-MiniLM-L6-v2-int8/tokenizer.json \
        --precision int8 --backend-label ort-int8 >/dev/null
    printf "  %-10s " "ort-int8"
    "$PY" analyze/correctness.py --vectors "$vec" --bucket short | tail -3 | tr '\n' ' '
    echo
fi
if [[ "$HAS_QDRANT" == "1" ]] && ! skip ort; then
    vec="vectors/ort-qdrant_short.bin"
    target/release/ort_runner --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
        --out "results/_correctness_ort-qdrant.json" --save-vectors "$vec" \
        --model models/all-MiniLM-L6-v2-qdrant/model.onnx \
        --tokenizer models/all-MiniLM-L6-v2-qdrant/tokenizer.json \
        --precision fp32 --backend-label ort-qdrant >/dev/null
    printf "  %-10s " "ort-qdrant"
    "$PY" analyze/correctness.py --vectors "$vec" --bucket short | tail -3 | tr '\n' ' '
    echo
fi

# ---------- 9. aggregate ----------
echo
echo ">> aggregate"
"$PY" analyze/compare.py results/
