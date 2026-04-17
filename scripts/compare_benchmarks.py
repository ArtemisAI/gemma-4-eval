#!/usr/bin/env python3
"""Compare GPU benchmark results across multiple runs.

Reads bench_*.json files from results/ and produces a comparison table.

Usage:
    python scripts/compare_benchmarks.py [--results-dir results/]
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    """Load all benchmark JSON files."""
    p = Path(results_dir)
    files = sorted(p.glob("bench_*.json"))
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            data["_file"] = f.name
            results.append(data)
    return results


def print_comparison(results: list[dict]):
    """Print side-by-side comparison table."""
    if not results:
        print("No benchmark results found.")
        return

    gpus = [r["gpu_name"] for r in results]
    header = f"{'Test':<32} " + " ".join(f"{g:>16}" for g in gpus)
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("GPU BENCHMARK COMPARISON — Gemma4 27B Q6_K on llama-server")
    print("=" * len(header))

    # Baseline comparison
    print(f"\n{'BASELINE (tokens/sec, mean ± stdev)':^{len(header)}}")
    print(separator)
    print(header)
    print(separator)

    # Get all test names from first result
    test_names = list(results[0].get("baseline", {}).keys())

    for test in test_names:
        row = f"{test:<32} "
        for r in results:
            data = r.get("baseline", {}).get(test, {})
            tps = data.get("tokens_per_second", {})
            if tps:
                mean = tps.get("mean", 0)
                stdev = tps.get("stdev", 0)
                row += f"{mean:>10.1f} ±{stdev:<4.1f} "
            else:
                row += f"{'ERROR':>16} "
        print(row)

    print(separator)

    # Average across all tests
    row = f"{'AVERAGE'::<32} "
    for r in results:
        speeds = []
        for test, data in r.get("baseline", {}).items():
            tps = data.get("tokens_per_second", {})
            if tps and tps.get("mean"):
                speeds.append(tps["mean"])
        if speeds:
            avg = sum(speeds) / len(speeds)
            row += f"{avg:>10.1f}{'':>6} "
        else:
            row += f"{'N/A':>16} "
    print(row)

    # Stress comparison
    any_stress = any(r.get("stress") for r in results)
    if any_stress:
        print(f"\n{'STRESS TEST (aggregate tokens/sec)':^{len(header)}}")
        print(separator)
        print(header)
        print(separator)

        # Get concurrency levels
        stress_keys = set()
        for r in results:
            if r.get("stress"):
                stress_keys.update(r["stress"].keys())

        for key in sorted(stress_keys):
            row = f"{key:<32} "
            for r in results:
                data = (r.get("stress") or {}).get(key, {})
                agg = data.get("aggregate_tokens_per_second")
                if agg is not None:
                    row += f"{agg:>10.1f}{'':>6} "
                else:
                    row += f"{'N/A':>16} "
            print(row)
        print(separator)

    # Performance per dollar
    # Vast.ai pricing (approximate)
    pricing = {
        "RTX 3090": 0.16,
        "RTX 4090": 0.38,
        "RTX 5090": 0.38,
    }

    print(f"\n{'PERFORMANCE PER DOLLAR':^{len(header)}}")
    print(separator)

    for r in results:
        gpu = r["gpu_name"]
        speeds = []
        for test, data in r.get("baseline", {}).items():
            tps = data.get("tokens_per_second", {})
            if tps and tps.get("mean"):
                speeds.append(tps["mean"])
        if speeds:
            avg_tps = sum(speeds) / len(speeds)
            price = pricing.get(gpu, 0)
            if price > 0:
                tps_per_dollar = avg_tps / price
                print(f"  {gpu:<20} {avg_tps:>8.1f} t/s  @  ${price:.2f}/hr  =  {tps_per_dollar:>8.0f} tokens/sec/$")
            else:
                print(f"  {gpu:<20} {avg_tps:>8.1f} t/s  (local, no cost)")

    print(separator)
    print(f"\nTimestamps: {', '.join(r.get('timestamp','?')[:19] for r in results)}")
    print(f"Files: {', '.join(r.get('_file','?') for r in results)}")


def main():
    parser = argparse.ArgumentParser(description="Compare GPU benchmarks")
    parser.add_argument("--results-dir", default="results/", help="Directory with bench_*.json files")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print_comparison(results)


if __name__ == "__main__":
    main()
