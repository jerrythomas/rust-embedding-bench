#!/usr/bin/env python3
"""Aggregate JSON benchmark results from results/ into a comparison table.

Usage:
    python analyze/compare.py results/
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("usage: compare.py <results_dir>", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        c = data["config"]
        m = data["metrics"]
        rows.append({
            "backend": data["backend"],
            "length": c["length"],
            "batch": c["batch_size"],
            "threads": c["threads"],
            "precision": c["precision"],
            "p50_ms": m["latency_ms"]["p50"],
            "p95_ms": m["latency_ms"]["p95"],
            "p99_ms": m["latency_ms"]["p99"],
            "mean_ms": m["latency_ms"]["mean"],
            "eps": m["throughput_eps"],
            "cold_ms": m["cold_start_ms"],
            "rss_mb": m["rss_peak_mb"],
        })

    if not rows:
        print("no results found")
        return

    headers = ["backend", "length", "batch", "threads", "precision",
               "p50_ms", "p95_ms", "p99_ms", "mean_ms", "eps", "cold_ms", "rss_mb"]

    def cell(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    widths = {h: max(len(h), max(len(cell(r[h])) for r in rows)) for h in headers}

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    rows.sort(key=lambda r: (r["length"], r["batch"], r["threads"], r["backend"]))
    for r in rows:
        print(" | ".join(cell(r[h]).ljust(widths[h]) for h in headers))

    print()
    print("Note: ollama rss_mb measures the runner process only; the model lives")
    print("in the ollama daemon. To capture daemon RSS, probe `pgrep ollama` separately.")


if __name__ == "__main__":
    main()
