#!/usr/bin/env python3
"""GPU Benchmark Suite for Gemma4 31B on llama-server.

Runs standardized baseline and stress tests across different GPU types.
Produces clean, publishable results with identical methodology per GPU.

Usage:
    python scripts/gpu_benchmark.py --endpoint http://HOST:8080 --gpu-name "RTX 4090"
    python scripts/gpu_benchmark.py --endpoint http://HOST:8080 --gpu-name "RTX 5090" --stress
"""

import argparse
import json
import time
import statistics
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Test prompts — identical across all GPUs for clean comparison
# ---------------------------------------------------------------------------

BASELINE_PROMPTS = {
    "short_generation": {
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one sentence."}],
        "max_tokens": 64,
        "description": "Minimal generation — measures decode startup latency",
    },
    "medium_generation": {
        "messages": [{"role": "user", "content": "Explain how a hash table works. Be concise but thorough."}],
        "max_tokens": 512,
        "description": "Medium output — standard decode speed",
    },
    "long_generation": {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed Python implementation of a binary search tree with insert, delete, "
                    "search, and in-order traversal. Include type hints and docstrings."
                ),
            }
        ],
        "max_tokens": 2048,
        "description": "Long output — sustained decode throughput",
    },
    "long_input_short_output": {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the following in exactly one sentence:\n\n"
                    + "The quick brown fox jumps over the lazy dog. " * 200
                    + "\n\nOne sentence summary:"
                ),
            }
        ],
        "max_tokens": 128,
        "description": "Large prompt — measures prefill (prompt processing) speed",
    },
    "thinking_task": {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Prove that the square root of 2 is irrational using proof by contradiction. "
                    "Show every step clearly."
                ),
            }
        ],
        "max_tokens": 4096,
        "description": "Reasoning task — thinking + content generation",
        "extra_body": {"thinking_budget_tokens": 2048},
    },
    "code_generation": {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write an async Python HTTP client with exponential backoff retry, "
                    "connection pooling, and proper error handling. Use aiohttp."
                ),
            }
        ],
        "max_tokens": 2048,
        "description": "Code generation — real-world coding task",
    },
}

STRESS_PROMPTS = {
    "concurrent_simple": {
        "messages": [{"role": "user", "content": "Write a haiku about programming."}],
        "max_tokens": 128,
        "description": "Simple prompt for concurrent load testing",
    },
    "concurrent_medium": {
        "messages": [
            {
                "role": "user",
                "content": "Explain the difference between TCP and UDP. Include use cases for each.",
            }
        ],
        "max_tokens": 512,
        "description": "Medium prompt for concurrent load testing",
    },
}


def run_single_request(client: OpenAI, model: str, prompt: dict) -> dict:
    """Run a single request and collect metrics."""
    messages = prompt["messages"]
    max_tokens = prompt.get("max_tokens", 2048)
    extra_body = prompt.get("extra_body")

    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            extra_body=extra_body,
            timeout=300,
        )
        elapsed = time.monotonic() - start

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", "") or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        return {
            "status": "ok",
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
            "prompt_tokens_per_second": round(prompt_tokens / elapsed, 2) if elapsed > 0 else 0,
            "content_length": len(content),
            "thinking_length": len(reasoning),
            "content_populated": bool(content.strip()),
            "time_to_first_token_estimate_s": None,  # would need streaming
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "status": "error",
            "elapsed_s": round(elapsed, 3),
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "content_length": 0,
            "thinking_length": 0,
            "content_populated": False,
        }


def run_baseline(client: OpenAI, model: str, runs_per_prompt: int = 3) -> dict:
    """Run baseline tests — each prompt N times, collect stats."""
    print(f"\n{'='*60}")
    print(f"BASELINE BENCHMARK ({runs_per_prompt} runs per prompt)")
    print(f"{'='*60}")

    results = {}
    for name, prompt in BASELINE_PROMPTS.items():
        print(f"\n  [{name}] {prompt['description']}")
        runs = []
        for i in range(runs_per_prompt):
            r = run_single_request(client, model, prompt)
            runs.append(r)
            status = "OK" if r["status"] == "ok" else f"ERR: {r.get('error', '?')}"
            print(f"    run {i+1}: {r['tokens_per_second']} t/s, {r['completion_tokens']} tokens, {r['elapsed_s']}s — {status}")

        ok_runs = [r for r in runs if r["status"] == "ok"]
        if ok_runs:
            speeds = [r["tokens_per_second"] for r in ok_runs]
            results[name] = {
                "description": prompt["description"],
                "runs": len(ok_runs),
                "errors": len(runs) - len(ok_runs),
                "tokens_per_second": {
                    "mean": round(statistics.mean(speeds), 2),
                    "median": round(statistics.median(speeds), 2),
                    "stdev": round(statistics.stdev(speeds), 2) if len(speeds) > 1 else 0,
                    "min": round(min(speeds), 2),
                    "max": round(max(speeds), 2),
                },
                "avg_completion_tokens": round(statistics.mean([r["completion_tokens"] for r in ok_runs])),
                "avg_elapsed_s": round(statistics.mean([r["elapsed_s"] for r in ok_runs]), 2),
                "content_populated_rate": sum(1 for r in ok_runs if r["content_populated"]) / len(ok_runs),
                "raw_runs": ok_runs,
            }
        else:
            results[name] = {"description": prompt["description"], "runs": 0, "errors": len(runs), "error": "all runs failed"}

    return results


def run_stress(client: OpenAI, model: str, concurrency_levels: list[int] = None) -> dict:
    """Run stress tests — concurrent requests at increasing load."""
    concurrency_levels = concurrency_levels or [1, 2, 4, 8]
    print(f"\n{'='*60}")
    print(f"STRESS TEST (concurrency: {concurrency_levels})")
    print(f"{'='*60}")

    results = {}
    prompt = STRESS_PROMPTS["concurrent_medium"]

    for level in concurrency_levels:
        print(f"\n  Concurrency={level}: sending {level} simultaneous requests...")
        start = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(run_single_request, client, model, prompt) for _ in range(level)]
            runs = [f.result() for f in concurrent.futures.as_completed(futures)]

        wall_time = time.monotonic() - start
        ok_runs = [r for r in runs if r["status"] == "ok"]

        if ok_runs:
            individual_speeds = [r["tokens_per_second"] for r in ok_runs]
            total_tokens = sum(r["completion_tokens"] for r in ok_runs)
            aggregate_tps = total_tokens / wall_time if wall_time > 0 else 0

            results[f"concurrent_{level}"] = {
                "concurrency": level,
                "wall_time_s": round(wall_time, 2),
                "requests_ok": len(ok_runs),
                "requests_error": len(runs) - len(ok_runs),
                "individual_tokens_per_second": {
                    "mean": round(statistics.mean(individual_speeds), 2),
                    "min": round(min(individual_speeds), 2),
                    "max": round(max(individual_speeds), 2),
                },
                "aggregate_tokens_per_second": round(aggregate_tps, 2),
                "total_tokens_generated": total_tokens,
            }
            print(f"    OK: {len(ok_runs)}/{level} requests, aggregate {aggregate_tps:.1f} t/s, wall time {wall_time:.1f}s")
        else:
            results[f"concurrent_{level}"] = {
                "concurrency": level,
                "wall_time_s": round(wall_time, 2),
                "requests_ok": 0,
                "requests_error": level,
                "error": "all requests failed",
            }
            print(f"    FAILED: all {level} requests errored")

    return results


def get_server_metrics(base_url: str) -> dict:
    """Fetch llama-server /metrics endpoint for hardware info."""
    import urllib.request
    try:
        metrics_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/metrics"
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode()
        return {"raw_metrics": text[:2000], "available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}


def run_benchmark(endpoint: str, model: str, gpu_name: str, stress: bool = False,
                  runs: int = 3, output_dir: str = None):
    """Run full benchmark suite."""
    base = endpoint.rstrip("/").rsplit("/v1", 1)[0] + "/v1"
    client = OpenAI(base_url=base, api_key="not-needed", timeout=300)

    print(f"\n{'#'*60}")
    print(f"  GPU BENCHMARK: {gpu_name}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Model: {model}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'#'*60}")

    # Warm up — discard first request
    print("\n  Warming up (1 request, discarded)...")
    run_single_request(client, model, BASELINE_PROMPTS["short_generation"])

    # Run baseline
    baseline = run_baseline(client, model, runs_per_prompt=runs)

    # Run stress if requested
    stress_results = None
    if stress:
        stress_results = run_stress(client, model)

    # Collect server metrics
    metrics = get_server_metrics(endpoint)

    # Build report
    report = {
        "gpu_name": gpu_name,
        "model": model,
        "endpoint": endpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs_per_prompt": runs,
        "baseline": baseline,
        "stress": stress_results,
        "server_metrics": metrics,
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {gpu_name}")
    print(f"{'='*60}")
    for name, data in baseline.items():
        if "tokens_per_second" in data:
            tps = data["tokens_per_second"]
            print(f"  {name:30s}  {tps['mean']:6.1f} t/s (±{tps['stdev']:.1f})  "
                  f"avg {data['avg_completion_tokens']} tokens in {data['avg_elapsed_s']}s")

    # Save
    output_dir = Path(output_dir or Path(__file__).parent.parent / "results")
    output_dir.mkdir(exist_ok=True)
    gpu_slug = gpu_name.lower().replace(" ", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bench_{gpu_slug}_{ts}.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Results saved to {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark for Gemma4 on llama-server")
    parser.add_argument("--endpoint", required=True, help="llama-server endpoint (e.g. http://host:8080)")
    parser.add_argument("--model", default="gemma4", help="Model name")
    parser.add_argument("--gpu-name", required=True, help="GPU identifier (e.g. 'RTX 4090')")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--runs", type=int, default=3, help="Runs per baseline prompt")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()

    run_benchmark(args.endpoint, args.model, args.gpu_name,
                  stress=args.stress, runs=args.runs, output_dir=args.output)


if __name__ == "__main__":
    main()
