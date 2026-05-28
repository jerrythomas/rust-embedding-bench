# rust-embedding-bench — Makefile
#
# Common targets:
#   make bench         # full cycle (setup + build + sweep + correctness + aggregate)
#   make build         # cargo build --release --workspace
#   make sweep         # run the sweep only (assumes setup + build are done)
#   make aggregate     # render the comparison table from results/
#   make correctness   # cosine vs Python reference, one pass per backend
#   make clean         # remove results, vectors, target/
#   make nuke          # also wipe .venv/, models/, .fastembed_cache/
#   make help          # this list
#
# Sweep config is overridable via env or `make VAR=...`:
#   make bench LENGTHS=short BATCHES=32 THREADS="1 8" WARMUP=20 MEASURE=200
#   make bench SKIP="ollama ort"

SHELL := /usr/bin/env bash

PY      := .venv/bin/python
UV      := uv

# Sweep config (env-overridable)
SKIP        ?=
LENGTHS     ?= short medium long
BATCHES     ?= 1 8 32
THREADS     ?= 1
WARMUP      ?= 50
MEASURE     ?= 500
OLLAMA_HOST ?= http://localhost:11434

# Noise filter for llama.cpp's chatty init output
LLAMA_NOISE := init: embeddings|compute buffer|matches expectation|ggml_metal|~llama_context|llama_context:|sched_reserve|graph_reserve|set_abort_callback|n_outputs|llama_kv_cache|llama_memory|load_tensors|create_tensor|done_getting_tensors

BACKENDS := fastembed candle ollama ort llama

.PHONY: help bench setup build sweep correctness aggregate clean nuke venv models reference ollama-pull

.DEFAULT_GOAL := help

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bench: setup build sweep correctness aggregate  ## Full benchmark cycle

setup: venv ollama-pull models reference  ## venv + Python deps + ollama pull + model files + reference vectors

build:  ## cargo build --release --workspace
	cargo build --release --workspace

venv: .venv/.ok
.venv/.ok:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install -p $(PY) -r reference/requirements.txt -r analyze/requirements.txt "optimum[onnxruntime]" onnx huggingface_hub
	@touch $@

ollama-pull:  ## Pull all-minilm into the Ollama daemon if reachable
	@if curl -sf -m 2 $(OLLAMA_HOST)/api/tags >/dev/null 2>&1; then \
		if curl -sf -m 5 $(OLLAMA_HOST)/api/tags | grep -q '"all-minilm'; then \
			echo ">> ollama all-minilm present"; \
		else \
			echo ">> pulling ollama all-minilm"; \
			curl -s -X POST $(OLLAMA_HOST)/api/pull -d '{"model":"all-minilm"}' >/dev/null \
				|| echo "   pull failed, ollama backend will be skipped"; \
		fi; \
	else \
		echo ">> ollama daemon not reachable at $(OLLAMA_HOST); skipping pull"; \
	fi

models: venv  ## Export ONNX, quantize, fetch Qdrant + GGUF
	@$(PY) reference/export_onnx.py
	@$(PY) reference/quantize.py
	@$(PY) reference/download_qdrant.py
	@$(PY) reference/download_gguf.py

reference: venv  ## Generate the Python reference vectors
	@$(PY) reference/generate_reference.py

sweep:  ## Run the configured sweep (assumes setup + build are done)
	@mkdir -p results vectors
	@HAS_INT8=0; HAS_QDRANT=0; \
	[[ -f models/all-MiniLM-L6-v2-int8/model.onnx ]] && HAS_INT8=1; \
	[[ -f models/all-MiniLM-L6-v2-qdrant/model.onnx ]] && HAS_QDRANT=1; \
	for backend in $(BACKENDS); do \
		case " $(SKIP) " in *" $$backend "*) echo "skip $$backend (SKIP env)"; continue;; esac; \
		bin="target/release/$${backend}_runner"; \
		if [[ ! -x $$bin ]]; then echo "no binary for $$backend; run 'make build'"; continue; fi; \
		for length in $(LENGTHS); do \
			for batch in $(BATCHES); do \
				for threads in $(THREADS); do \
					out="results/$${backend}_$${length}_b$${batch}_t$${threads}.json"; \
					echo ">> sweep $$backend length=$$length batch=$$batch threads=$$threads"; \
					RAYON_NUM_THREADS=$$threads OMP_NUM_THREADS=$$threads $$bin \
						--length $$length --batch $$batch --threads $$threads \
						--warmup $(WARMUP) --measure $(MEASURE) --out $$out \
						2> >(grep -vE "$(LLAMA_NOISE)" >&2) \
						|| echo "   FAILED, continuing"; \
					if [[ $$backend == ort && $$HAS_INT8 == 1 ]]; then \
						out_i8="results/ort-int8_$${length}_b$${batch}_t$${threads}.json"; \
						echo ">> sweep ort-int8 length=$$length batch=$$batch threads=$$threads"; \
						RAYON_NUM_THREADS=$$threads OMP_NUM_THREADS=$$threads $$bin \
							--length $$length --batch $$batch --threads $$threads \
							--warmup $(WARMUP) --measure $(MEASURE) --out $$out_i8 \
							--model models/all-MiniLM-L6-v2-int8/model.onnx \
							--tokenizer models/all-MiniLM-L6-v2-int8/tokenizer.json \
							--precision int8 --backend-label ort-int8 \
							|| echo "   FAILED, continuing"; \
					fi; \
					if [[ $$backend == ort && $$HAS_QDRANT == 1 ]]; then \
						out_q="results/ort-qdrant_$${length}_b$${batch}_t$${threads}.json"; \
						echo ">> sweep ort-qdrant length=$$length batch=$$batch threads=$$threads"; \
						RAYON_NUM_THREADS=$$threads OMP_NUM_THREADS=$$threads $$bin \
							--length $$length --batch $$batch --threads $$threads \
							--warmup $(WARMUP) --measure $(MEASURE) --out $$out_q \
							--model models/all-MiniLM-L6-v2-qdrant/model.onnx \
							--tokenizer models/all-MiniLM-L6-v2-qdrant/tokenizer.json \
							--precision fp32 --backend-label ort-qdrant \
							|| echo "   FAILED, continuing"; \
					fi; \
				done; \
			done; \
		done; \
	done

correctness:  ## Cosine vs Python reference, one pass per backend
	@mkdir -p vectors
	@echo ">> correctness pass (cosine vs Python reference)"
	@for backend in $(BACKENDS); do \
		case " $(SKIP) " in *" $$backend "*) continue;; esac; \
		bin="target/release/$${backend}_runner"; \
		[[ -x $$bin ]] || continue; \
		vec="vectors/$${backend}_short.bin"; \
		$$bin --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
			--out "results/_correctness_$${backend}.json" --save-vectors "$$vec" \
			2> >(grep -vE "$(LLAMA_NOISE)" >&2) >/dev/null; \
		printf "  %-10s " "$$backend"; \
		$(PY) analyze/correctness.py --vectors "$$vec" --bucket short 2>/dev/null | tail -3 | tr '\n' ' '; \
		echo; \
	done
	@if [[ -f models/all-MiniLM-L6-v2-int8/model.onnx ]]; then \
		vec="vectors/ort-int8_short.bin"; \
		target/release/ort_runner --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
			--out "results/_correctness_ort-int8.json" --save-vectors "$$vec" \
			--model models/all-MiniLM-L6-v2-int8/model.onnx \
			--tokenizer models/all-MiniLM-L6-v2-int8/tokenizer.json \
			--precision int8 --backend-label ort-int8 >/dev/null 2>&1; \
		printf "  %-10s " "ort-int8"; \
		$(PY) analyze/correctness.py --vectors "$$vec" --bucket short 2>/dev/null | tail -3 | tr '\n' ' '; \
		echo; \
	fi
	@if [[ -f models/all-MiniLM-L6-v2-qdrant/model.onnx ]]; then \
		vec="vectors/ort-qdrant_short.bin"; \
		target/release/ort_runner --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
			--out "results/_correctness_ort-qdrant.json" --save-vectors "$$vec" \
			--model models/all-MiniLM-L6-v2-qdrant/model.onnx \
			--tokenizer models/all-MiniLM-L6-v2-qdrant/tokenizer.json \
			--precision fp32 --backend-label ort-qdrant >/dev/null 2>&1; \
		printf "  %-10s " "ort-qdrant"; \
		$(PY) analyze/correctness.py --vectors "$$vec" --bucket short 2>/dev/null | tail -3 | tr '\n' ' '; \
		echo; \
	fi

aggregate:  ## Render comparison table from existing results/
	@$(PY) analyze/compare.py results/

clean:  ## Remove results/, vectors/, target/, baselines/
	rm -rf results vectors target baselines

nuke: clean  ## Also wipe .venv/, models/, .fastembed_cache/ (full reset)
	rm -rf .venv models .fastembed_cache
