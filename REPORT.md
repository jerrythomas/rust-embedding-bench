# MiniLM-L6-v2 embedding throughput on Apple Silicon: a Rust backend comparison

## Background

We wanted to know how the popular Rust ML libraries actually compare when running the same embedding model on the same hardware. Specifically: how much latency and throughput is available for `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 12 heads, 384 hidden, 1536 FFN, 22M parameters) on a developer-class machine, and which of the dimensions you might tune — batch size, threads, precision, GPU offload, request concurrency — actually move the numbers.

## Setup

**Hardware and OS**

| Component | Value |
|---|---|
| Machine | MacBook Pro (Mac16,5) |
| Chip | Apple M4 Max |
| Cores | 16 (12 performance + 4 efficiency) |
| Memory | 48 GB unified |
| OS | macOS 26.3.1 |

**Toolchain**

| Component | Version |
|---|---|
| Rust | 1.95.0 (Homebrew) |
| Python | 3.12.11 (uv-managed venv) |
| sentence-transformers | 5.5.1 |
| onnxruntime (Python) | 1.26.0 |
| optimum | 2.1.0 |

**Rust crates**

| Crate | Version |
|---|---|
| ort | 2.0.0-rc.12 |
| fastembed | 5.14.0 |
| candle-core / candle-nn / candle-transformers | 0.10.2 |
| llama-cpp-2 | 0.1.146 |
| tokenizers | 0.21.4 |
| hf-hub | 0.4.3 |

**Data**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Corpus: 29 sentences across three length buckets (short: 5–20 tokens; medium: 50–100; long: 200–256)
- Reference: Python `sentence-transformers` fp32, used to validate every backend's vectors via cosine similarity

## Backends measured

| Backend | Engine | Model file |
|---|---|---|
| fastembed | ONNX Runtime (via `ort` crate) | Qdrant pre-optimized fp32 ONNX |
| ort fp32 | ONNX Runtime directly | Our own optimum-exported fp32 ONNX |
| ort-qdrant | ONNX Runtime | Same Qdrant fp32 ONNX as fastembed |
| ort-int8 | ONNX Runtime | Dynamic int8 quantization (`quantize_dynamic` / QInt8) |
| candle | candle (pure Rust) | fp32 safetensors from HF |
| ollama | llama.cpp via HTTP daemon | F16 GGUF |
| llama-cpp-2 | llama.cpp in-process | Same F16 GGUF, CPU and Metal variants |

Each runner is a separate Rust binary with the same CLI surface (`--length`, `--batch`, `--threads`, `--warmup`, `--measure`, `--save-vectors`) and writes a JSON record with p50/p95/p99/mean latency, throughput, cold-start time, and peak RSS. A shell driver sweeps configurations and a Python script aggregates results.

## Sweep dimensions

- Sequence length: short / medium / long
- Batch size: 1, 8, 32
- Thread count: 1, 4, 8
- For Ollama: concurrent in-flight HTTP requests at 1, 4, 8, 16
- For llama-cpp-2: CPU only vs `metal` feature enabled

## Correctness

Each backend produced 15 short-text embeddings that were compared against the Python reference (cosine similarity, per sentence). Results:

| backend | min cosine | mean cosine |
|---|---|---|
| fastembed, candle, ort, ort-qdrant | 1.000000 | 1.000000 |
| llama-cpp-2 (CPU and Metal) | 0.999999 | 0.999999 |
| ollama | 0.999999 | 0.999999 |
| ort-int8 (dynamic) | 0.961293 | 0.976919 |

F16 GGUF was indistinguishable from fp32 for retrieval purposes. Naive dynamic int8 quantization measurably degraded the vectors on all 15 short sentences.

## Headline results

**Per-query latency, p50 ms (batch=1, threads=1):**

| backend | short | medium | long |
|---|---|---|---|
| ort-int8 | 0.74 | 3.17 | 7.90 |
| ort-qdrant | 1.02 | 2.82 | 6.67 |
| ort | 1.10 | 2.89 | 6.89 |
| llama-cpp-2 (Metal) | 1.26 | 1.39 | 2.04 |
| fastembed | 1.71 | 3.22 | 5.27 |
| candle | 8.15 | 17.15 | 43.56 |
| ollama | 11.05 | 10.74 | 11.47 |

**Throughput, embeddings/sec (batch=32, threads=8 for CPU backends):**

| backend | short | medium | long |
|---|---|---|---|
| llama-cpp-2 (Metal) | 9676 | 2137 | 948 |
| ort fp32 | 3052 | 848 | 261 |
| ort-qdrant | 3089 | 839 | 258 |
| fastembed | 2849 | 856 | 315 |
| ort-int8 | 2732 | 703 | 250 |
| candle | 603 | 119 | 31 |
| ollama (tuned: b=8, concurrency=16) | 553 | 445 | 260 |

## Observations

**Threading parity matters for any cross-backend comparison.** At `--threads 1`, fastembed appeared 1.5–2x faster than vanilla ort. The cause was that fastembed's underlying ORT session auto-detects available cores, so it was using ~8 threads while our ort runner was honoring the 1-thread setting. At equal thread counts (t=8), ort and fastembed throughput differ by under 5%. Most "library X is much faster than library Y" claims about ONNX-based stacks resolve to this if you check.

**Dynamic int8 quantization did not pay off on this hardware.** Naive `quantize_dynamic` produced a model that was the same speed or slower than fp32 ONNX at most batch sizes, and dropped cosine similarity to the reference from 1.000 to 0.961. The likely causes are quant/dequant overhead between layers and the lack of an AVX-512-VNNI-equivalent int8 fast path on ARM NEON. Memory dropped about 50%, which is genuinely useful in constrained deployments but doesn't help raw throughput here. The same experiment on x86 with VNNI would likely produce different numbers.

**Metal GPU acceleration helped a specific workload shape.** llama-cpp-2 with the `metal` feature was ~3x faster than CPU at short text and high batch sizes (short b=32 throughput went from 3300 → 9676 eps, +193%) but provided negligible gains at batch=1, or at long sequences. GPU dispatch overhead absorbs the compute win on small individual workloads.

**Ollama's HTTP layer is most of the cost for single queries.** Loading the same F16 GGUF directly via llama-cpp-2 in-process gave 1.28 ms p50 on short text vs Ollama's 11.05 ms — roughly 8x faster, same model, same engine. Pipelining 8–16 concurrent requests to Ollama recovers about 3–4x of that on the throughput side (385 eps at concurrency=8 vs 93 eps at concurrency=1), but per-call latency rises with concurrency. Practical knob: concurrency is a throughput-only optimization, not a latency one.

**The inference engine choice mattered more than precision.** llama-cpp-2 with F16 outpaced every fp32 ORT configuration at every batch and length combination. The biggest factor was that llama.cpp enables flash attention by default for the BERT path, which scales better than the standard attention kernel ORT uses for sequence-classification graphs. The "which library calls the model" decision was a larger lever than "what precision the model is in" for this workload.

## Practical conclusions for this model

- **Single-query latency (interactive search, etc.):** ort fp32 or llama-cpp-2. Both around 1–2 ms p50 across all lengths. ort is the simpler integration if you already have ONNX models.
- **High-throughput indexing (embedding every string in a database):** llama-cpp-2 with the `metal` feature. ~9676 eps on short text means about 7 minutes per million rows, single-threaded on the runner side.
- **Using ollama for embeddings:** batch=8 plus concurrency=16. Recovers about 6x over single-request usage. Still 5–17x slower than in-process options because the daemon's protocol layer remains in the hot path. Reasonable if you already run ollama for other workloads; not a great choice if embeddings are the primary use case.
- **Memory-constrained deployments:** ort-int8 cuts RAM by ~50% (190 → 90 MB on this hardware). The accuracy trade-off (cosine 0.96) may or may not be acceptable depending on the downstream retrieval task.

## Limitations

- Single Apple Silicon machine. x86 with AVX-512-VNNI would likely show different int8 numbers and possibly close the fp32/int8 gap.
- Small corpus (29 sentences). Throughput measurements are well-amortized; cold-start figures and tail percentiles have wider error bars.
- Only one model architecture measured. Embedding models in the 100M–1B parameter range would have a different ratio of overhead to compute, and the library choice would matter less.
- The candle and ort runners use scalar mean-pool/normalize in Rust after the model returns hidden states. Fastembed and llama-cpp-2 do pooling inside their respective engines. A small amount of the gap between them is likely this.

## Reproducing the experiment

The harness lives in this directory. After cloning the repository:

```
cd rust-embedding-bench
make bench
```

The pipeline auto-installs Python dependencies into a `uv`-managed `.venv`, builds all Rust runners in release mode, downloads model files (ONNX, GGUF, safetensors), runs the full sweep, validates correctness against the Python reference, and prints an aggregated table. Full cycle is about 10 minutes on warm caches; first run takes longer because of dep downloads and the ONNX export.

Sweep config knobs (env or `make VAR=...`):

```
SKIP="ollama"               # space-separated backends to skip
LENGTHS="short medium long" # which buckets to sweep
BATCHES="1 8 32"            # batch sizes
THREADS="1 4 8"             # thread configs
WARMUP=50 MEASURE=500       # per-run sample counts
```

Other Makefile targets: `make build`, `make sweep`, `make aggregate`, `make correctness`, `make clean`, `make nuke`. `make help` lists them.

Each result is written as a JSON record in `results/`. `make aggregate` (or `python analyze/compare.py results/`) re-renders the comparison table at any time.
