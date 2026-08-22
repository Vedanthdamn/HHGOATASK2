"""Latency benchmark: runs a batch of real MSMARCO-XI queries through the
pipeline and reports P50 / P70 / P100 latency.

Reports two numbers, because they answer different questions:

  - RETRIEVAL leg (embed query + Chroma ANN search + BM25 + RRF fusion):
    this is the "chunking + vector DB retrieval" leg named in the 200ms
    target (chunking itself is precomputed at index time, not on the query
    path). Measured with zero network calls, so it reflects pipeline
    engineering rather than a third-party API's latency.
  - FULL end-to-end (STT skipped here; includes retrieval + guardrails +
    Claude generation over the network): reported honestly even though an
    LLM API round-trip alone typically exceeds 200ms, so the two numbers
    should not be conflated.

Usage:
    python scripts/benchmark.py --n 50
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.dataset import load_msmarco_queries
from src.embeddings import Embedder
from src.harness import RagHarness
from src.retrieval import hybrid_retrieve
from src.vectorstore import VectorStore


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(round((p / 100) * (len(s) - 1))), len(s) - 1)
    return s[idx]


def summarize(name: str, values_ms: list[float]) -> dict:
    return {
        "stage": name,
        "n": len(values_ms),
        "p50_ms": round(percentile(values_ms, 50), 2),
        "p70_ms": round(percentile(values_ms, 70), 2),
        "p100_ms": round(percentile(values_ms, 100), 2),
        "mean_ms": round(statistics.mean(values_ms), 2) if values_ms else 0.0,
        "min_ms": round(min(values_ms), 2) if values_ms else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of queries to benchmark")
    parser.add_argument("--lang", default=config.DATASET_LANG)
    parser.add_argument("--skip-generation", action="store_true", help="only benchmark retrieval, not Claude generation")
    args = parser.parse_args()

    print(f"Loading {args.n} sample queries from MSMARCO-XI[{args.lang}] ...")
    queries = load_msmarco_queries(lang=args.lang, n=args.n)
    print(f"Collected {len(queries)} distinct queries.")

    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    print(f"Vector store has {store.count()} indexed chunks.")

    if queries:
        hybrid_retrieve(store, queries[0])  # warm up model/index caches once, like a long-lived server would

    retrieval_times = []
    for q in queries:
        t0 = time.perf_counter()
        hybrid_retrieve(store, q)
        retrieval_times.append((time.perf_counter() - t0) * 1000)

    results = {"retrieval_only": summarize("retrieval_only (embed+ANN+BM25+fusion)", retrieval_times)}

    if not args.skip_generation and config.ANTHROPIC_API_KEY:
        harness = RagHarness(store=store, embedder=embedder)
        full_times, extractive_times, synthesis_times, llm_times = [], [], [], []
        statuses, modes = [], []
        for q in queries:
            t0 = time.perf_counter()
            res = harness.run(query=q)
            elapsed = (time.perf_counter() - t0) * 1000
            full_times.append(elapsed)
            statuses.append(res.status)
            modes.append(res.generation_mode or "n/a")
            if res.generation_mode == "extractive":
                extractive_times.append(elapsed)
            elif res.generation_mode == "extractive_synthesis":
                synthesis_times.append(elapsed)
            elif res.generation_mode == "llm":
                llm_times.append(elapsed)

        results["end_to_end_mixed"] = summarize(
            "end_to_end (retrieval+guardrails+generation, extractive fast path where confident, Claude fallback otherwise)",
            full_times,
        )
        if extractive_times:
            results["end_to_end_extractive_only"] = summarize(
                "end_to_end, extractive fast-path queries only (no LLM network call)", extractive_times,
            )
        if synthesis_times:
            results["end_to_end_extractive_synthesis_only"] = summarize(
                "end_to_end, extractive-synthesis queries only (no LLM network call)", synthesis_times,
            )
        if llm_times:
            results["end_to_end_llm_only"] = summarize(
                "end_to_end, LLM-fallback queries only (Claude network call)", llm_times,
            )
        results["status_breakdown"] = {s: statuses.count(s) for s in set(statuses)}
        results["generation_mode_breakdown"] = {m: modes.count(m) for m in set(modes)}
    else:
        print("Skipping generation benchmark (no ANTHROPIC_API_KEY or --skip-generation set).")

    results["target_ms"] = 200
    results["retrieval_meets_target"] = results["retrieval_only"]["p100_ms"] < 200
    if "end_to_end_extractive_only" in results:
        results["extractive_fast_path_meets_target"] = results["end_to_end_extractive_only"]["p100_ms"] < 200
    if "end_to_end_mixed" in results:
        # The headline number: every non-LLM query, worst case included.
        results["end_to_end_meets_target"] = results["end_to_end_mixed"]["p100_ms"] < 200

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "latency_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Latency Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
