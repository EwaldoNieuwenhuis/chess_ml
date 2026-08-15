#!/usr/bin/env python3
"""
Benchmark Stockfish Engine Evaluation Latency.

Measures throughput and query latency across consecutive board evaluations
to verify persistent session performance.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.engine import StockfishManager


def run_benchmark(n_queries: int = 100, depth: int = 10) -> None:
    print("\n" + "=" * 60)
    print("♟️  STOCKFISH ENGINE EVALUATION BENCHMARK")
    print("=" * 60)

    manager = StockfishManager()
    mode = "Fallback Minimax" if manager.is_fallback_mode else f"Native Stockfish ({manager.binary_path})"
    print(f"Engine Mode    : {mode}")
    print(f"Test Iterations: {n_queries}")
    print(f"Search Depth   : {depth}")
    print("-" * 60)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Warmup query
    manager.evaluate(fen, depth=depth)

    latencies_ms: list[float] = []
    for i in range(n_queries):
        t0 = time.perf_counter()
        manager.evaluate(fen, depth=depth)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    manager.close()

    avg_lat = sum(latencies_ms) / len(latencies_ms)
    min_lat = min(latencies_ms)
    max_lat = max(latencies_ms)
    p95_lat = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(f"Average Latency : {avg_lat:.2f} ms")
    print(f"Median (p50)    : {sorted(latencies_ms)[len(latencies_ms)//2]:.2f} ms")
    print(f"95th Percentile : {p95_lat:.2f} ms")
    print(f"Min / Max       : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Throughput      : {1000.0 / avg_lat:.1f} queries/sec")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Stockfish engine latency.")
    parser.add_argument("-n", "--queries", type=int, default=50, help="Number of benchmark queries")
    parser.add_argument("-d", "--depth", type=int, default=5, help="Search depth per query")
    args = parser.parse_args()

    run_benchmark(n_queries=args.queries, depth=args.depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
