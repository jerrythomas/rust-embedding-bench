# rust-embedding-bench — Makefile
#
# Common targets:
#   make bench         # full cycle (venv, deps, build, sweep, correctness, aggregate)
#   make build         # cargo build --release --workspace
#   make sweep         # run the sweep only (assumes deps + binaries are ready)
#   make aggregate     # render the comparison table from results/
#   make correctness   # cosine vs Python reference, one pass per backend
#   make clean         # remove results, vectors, target/
#   make help          # this list
#
# Sweep config is overridable via env or `make VAR=...`:
#   make bench LENGTHS=short BATCHES=32 THREADS="1 8" WARMUP=20 MEASURE=200
#   make bench SKIP="ollama ort"

SHELL := /usr/bin/env bash

PY      := .venv/bin/python
UV      := uv

# Sweep config (env-overridable)
SKIP     ?=
LENGTHS  ?= short medium long
BATCHES  ?= 1 8 32
THREADS  ?= 1
WARMUP   ?= 50
MEASURE  ?= 500

# Export to children (run_all.sh reads these)
export SKIP LENGTHS BATCHES THREADS WARMUP MEASURE

.PHONY: help bench build sweep correctness aggregate clean nuke venv deps models reference

.DEFAULT_GOAL := help

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bench:  ## Full benchmark cycle: setup + build + sweep + correctness + aggregate
	./run_all.sh

build:  ## cargo build --release --workspace
	cargo build --release --workspace

sweep: build  ## Run the configured sweep (skips setup; for re-runs against the same env)
	./run_all.sh

aggregate:  ## Render comparison table from existing results/
	@$(PY) analyze/compare.py results/

correctness:  ## Recompute cosine vs Python reference for every backend
	@for backend in fastembed candle ollama ort llama; do \
		bin="target/release/$${backend}_runner"; \
		[[ -x $$bin ]] || continue; \
		printf "  %-10s " "$$backend"; \
		$$bin --length short --batch 1 --threads 1 --warmup 5 --measure 5 \
			--out "results/_correctness_$${backend}.json" \
			--save-vectors "vectors/$${backend}_short.bin" >/dev/null 2>&1; \
		$(PY) analyze/correctness.py --vectors "vectors/$${backend}_short.bin" --bucket short | tail -3 | tr '\n' ' '; \
		echo; \
	done

venv: .venv/.ok  ## Create the uv-managed Python venv if it doesn't exist
.venv/.ok:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install -p $(PY) -r reference/requirements.txt -r analyze/requirements.txt "optimum[onnxruntime]" onnx
	@touch $@

reference: venv  ## Generate the Python reference vectors
	$(PY) reference/generate_reference.py

models: venv  ## Download/export/quantize all model files
	$(PY) reference/export_onnx.py
	$(PY) reference/quantize.py
	$(PY) reference/download_qdrant.py

clean:  ## Remove results/, vectors/, target/, baselines/
	rm -rf results vectors target baselines

nuke: clean  ## Also wipe .venv/, models/, .fastembed_cache/ (starts from scratch)
	rm -rf .venv models .fastembed_cache
