# rust-embedding-bench

A reproducible benchmark harness for sentence-embedding throughput across the popular Rust ML libraries, using `sentence-transformers/all-MiniLM-L6-v2` as the common model.

Compares **ort** (ONNX Runtime), **fastembed**, **candle**, **ollama** (HTTP), and **llama-cpp-2** (in-process llama.cpp). Sweeps sequence length, batch size, thread count, HTTP concurrency, and GPU offload. Every backend's output is validated against a Python `sentence-transformers` reference via cosine similarity before its latency numbers are trusted.

Full methodology and findings: [**REPORT.md**](REPORT.md).

## TL;DR

On an Apple M4 Max, fp32-or-equivalent precision, single-machine benchmark:

| Backend | b=1 short (ms p50) | b=32 short (eps) | b=32 long (eps) |
|---|---|---|---|
| **llama-cpp-2 (Metal)** | 1.26 | **9676** | **948** |
| ort fp32 (t=8) | 1.10 | 3052 | 261 |
| fastembed | 1.71 | 2849 | 315 |
| candle (t=8) | 8.15 | 603 | 31 |
| ollama (HTTP, default) | 11.05 | 433 | 223 |
| ollama (b=8, concurrency=16) | 19.7 | 553 | 260 |

All backends validated at cosine ≥ 0.999 vs Python reference, except naive dynamic int8 quantization (cosine 0.961).

Headline observations:

- **Single-query latency:** ort fp32 or llama-cpp-2, both around 1–2 ms p50 across all sequence lengths.
- **High-throughput indexing:** llama-cpp-2 with the `metal` feature flag. ~9.7k embeddings per second on short text.
- **Threading parity matters.** A lot of "library X is much faster than library Y" comparisons resolve to "X auto-detected cores; Y honored `--threads 1`."
- **Ollama's HTTP layer is most of the cost** for single-query workloads. The same F16 GGUF loaded in-process via llama-cpp-2 is roughly 8x faster.
- **Dynamic int8 quantization was a wash on Apple Silicon.** Same speed or slower than fp32 ONNX, cosine drops to 0.96. May be different on x86 with VNNI.

## Quick start

Prerequisites: Rust toolchain, [`uv`](https://docs.astral.sh/uv/) (or any way to make a Python 3.12 venv), CMake (build dep for llama-cpp-2). Optional: a local Ollama daemon if you want to benchmark it.

```bash
git clone https://github.com/jerrythomas/rust-embedding-bench.git
cd rust-embedding-bench
make bench
```

`make help` lists the other targets (`build`, `sweep`, `aggregate`, `correctness`, `clean`, `nuke`). The pipeline is idempotent. On a clean checkout it will:

1. Create `.venv` and install Python deps (sentence-transformers, optimum, numpy)
2. Pull `all-minilm` into Ollama (if the daemon is reachable; otherwise the ollama backend is skipped)
3. Build all Rust runners in release mode
4. Export the ONNX model and dynamic-int8-quantize it
5. Generate Python reference embeddings
6. Sweep every backend × length × batch × threads combination
7. Run a correctness check (cosine vs reference) per backend
8. Print an aggregated comparison table

Cold first run: ~10 minutes (model downloads + cargo build). Warm reruns: ~2–3 minutes.

## What's measured

**Backends**

| Backend | Engine | Model file |
|---|---|---|
| fastembed | ONNX Runtime via `ort` | Qdrant pre-optimized fp32 ONNX (downloaded) |
| ort | ONNX Runtime | Optimum-exported fp32 ONNX |
| ort-qdrant | ONNX Runtime | Same model fastembed uses, called from our wrapper |
| ort-int8 | ONNX Runtime | Dynamic int8 quantization of the fp32 ONNX |
| candle | candle (pure Rust) | fp32 safetensors from HF |
| ollama | llama.cpp via HTTP daemon | F16 GGUF |
| llama-cpp-2 | llama.cpp in-process | Same F16 GGUF; CPU build by default, Metal optional |

**Sweep dimensions**

- Sequence length: short (5–20 tokens) / medium (50–100) / long (200–256)
- Batch size: 1, 8, 32
- Thread count: 1, 4, 8 (controlled per-runner; passed via env to libraries that honor `RAYON_NUM_THREADS` / `OMP_NUM_THREADS`)
- Ollama HTTP concurrency: 1, 4, 8, 16 (parallel in-flight requests)
- llama-cpp-2 GPU offload: enable via `features = ["metal"]` in `runners/llama_runner/Cargo.toml`

## Configuration

The sweep is configured via environment variables or `make VAR=...`. Examples:

```bash
# Only fastembed and ort, short text, batch=32, all thread configs
make bench SKIP="candle ollama llama" LENGTHS=short BATCHES=32 THREADS="1 4 8"

# Fast smoke test
make bench WARMUP=20 MEASURE=100

# Remote Ollama
OLLAMA_HOST=http://other-host:11434 make bench
```

| Variable | Default | Effect |
|---|---|---|
| `SKIP` | empty | Space-separated backends to skip (e.g. `"ollama ort"`) |
| `LENGTHS` | `short medium long` | Length buckets to sweep |
| `BATCHES` | `1 8 32` | Batch sizes to sweep |
| `THREADS` | `1` | Thread counts to sweep |
| `WARMUP` | 50 | Warmup embeddings per run (discarded) |
| `MEASURE` | 500 | Measured embeddings per run |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon URL |

Individual runners can also be invoked directly:

```bash
./target/release/ort_runner --length medium --batch 32 --threads 8 \
    --warmup 50 --measure 500 --out my_run.json
```

## Project layout

```
.
├── Makefile                    # entry point (`make bench`, `make help`, ...)
├── REPORT.md                   # methodology + full results
├── LICENSE                     # MIT
├── Cargo.toml                  # Rust workspace
├── corpus/sentences.json       # 29 deterministic test sentences
├── reference/
│   ├── generate_reference.py   # Python sentence-transformers reference
│   ├── export_onnx.py          # ONNX export via optimum
│   ├── quantize.py             # dynamic int8 quantization
│   ├── download_qdrant.py      # Qdrant pre-optimized ONNX
│   └── download_gguf.py        # F16 GGUF (Ollama cache or HF Hub)
├── runners/
│   ├── shared/                 # common CLI args, result schema, IO helpers
│   ├── fastembed_runner/
│   ├── ort_runner/             # supports --model, --tokenizer, --backend-label
│   ├── candle_runner/
│   ├── ollama_runner/          # supports --concurrency
│   └── llama_runner/           # in-process llama-cpp-2
└── analyze/
    ├── compare.py              # aggregate JSON results into a table
    └── correctness.py          # cosine similarity vs Python reference
```

## Adding your own backend

Each runner is a small (~150 line) Rust binary that:

1. Parses `shared::CommonArgs` (corpus, length, batch, threads, warmup, measure, out, save_vectors)
2. Loads the model once at startup
3. Runs a warmup loop
4. Runs a measurement loop, tracking per-item latency
5. Writes a `BenchResult` JSON via `shared::write_result`

To add a new library `foo`:

```bash
mkdir -p runners/foo_runner/src
cp runners/fastembed_runner/Cargo.toml runners/foo_runner/Cargo.toml
cp runners/fastembed_runner/src/main.rs runners/foo_runner/src/main.rs
# Edit the dependencies and the embed loop. Add foo_runner to the workspace
# members in the root Cargo.toml. Add "foo" to BACKENDS in run_all.sh.
./run_all.sh
```

Run with `--save-vectors vectors/foo_short.bin` once and let the harness compare them to the reference vectors. The correctness check catches most "the embedding pipeline is wired wrong" bugs before any latency claim becomes load-bearing.

## Hardware caveats

The numbers in this README and in REPORT.md were collected on a MacBook Pro with an Apple M4 Max (12 performance + 4 efficiency cores), 48 GB unified memory, macOS 26.3.1. Results on different hardware will differ. In particular:

- x86 with AVX-512 VNNI is expected to do considerably better with int8 (the AVX-512 dot-product path is a real fast track that ARM NEON lacks an equivalent for).
- NVIDIA GPU support is available in `ort` and `llama-cpp-2` via feature flags, but is not exercised here.
- ARM NEON fp32 throughput is very competitive on Apple Silicon, narrowing the typical fp32-vs-int8 gap seen on x86.

If you re-run this on different hardware, the harness will produce JSON output you can drop into [REPORT.md](REPORT.md)-style tables.

## License

[MIT](LICENSE).
