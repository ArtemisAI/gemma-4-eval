#!/usr/bin/env python3
"""Gemma4 evaluation runner.

Runs eval tasks against a llama-server endpoint and collects telemetry.
Results are saved to results/ and optionally pushed to Prometheus.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from evals.tasks import reasoning, coding, system_design, tool_use, multilingual

console = Console()

TASK_MODULES = {
    "reasoning": reasoning,
    "coding": coding,
    "system_design": system_design,
    "tool_use": tool_use,
    "multilingual": multilingual,
}


def load_config(path: str = None) -> dict:
    config_path = path or Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_eval(client: OpenAI, model: str, task: dict, config: dict) -> dict:
    """Run a single eval task and collect metrics."""
    defaults = config.get("defaults", {})
    messages = task["messages"]
    extra_body = {}
    if defaults.get("thinking_budget_tokens"):
        extra_body["thinking_budget_tokens"] = defaults["thinking_budget_tokens"]

    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=defaults.get("max_tokens", 4096),
            temperature=defaults.get("temperature", 0.7),
            extra_body=extra_body if extra_body else None,
            timeout=defaults.get("timeout", 600),
        )
        elapsed = time.monotonic() - start

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""

        # Usage stats
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        return {
            "task_name": task["name"],
            "category": task["category"],
            "status": "pass" if content.strip() else "fail",
            "content_populated": bool(content.strip()),
            "content_length": len(content),
            "thinking_length": len(reasoning_content),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(tokens_per_sec, 1),
            "elapsed_seconds": round(elapsed, 2),
            "error": None,
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "task_name": task["name"],
            "category": task["category"],
            "status": "error",
            "content_populated": False,
            "content_length": 0,
            "thinking_length": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_second": 0,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


def run_evals(endpoint: str, model: str, task_filter: str = None, config: dict = None):
    """Run all eval tasks and return results."""
    config = config or load_config()
    base = endpoint.rstrip("/").rsplit("/v1", 1)[0] + "/v1"
    timeout_s = config.get("defaults", {}).get("timeout", 600)
    client = OpenAI(base_url=base, api_key="not-needed", timeout=timeout_s)

    results = []
    tasks_to_run = {}

    for name, module in TASK_MODULES.items():
        if task_filter and name != task_filter:
            continue
        tasks_to_run[name] = module.get_tasks()

    total = sum(len(t) for t in tasks_to_run.values())
    console.print(f"\n[bold]Running {total} eval tasks against {endpoint}[/bold]\n")

    for category, tasks in tasks_to_run.items():
        console.print(f"[cyan]--- {category} ({len(tasks)} tasks) ---[/cyan]")
        for task in tasks:
            task["category"] = category
            console.print(f"  Running: {task['name']}...", end=" ")
            result = run_single_eval(client, model, task, config)
            status_icon = "[green]PASS[/green]" if result["status"] == "pass" else "[red]FAIL[/red]"
            console.print(f"{status_icon} ({result['elapsed_seconds']}s, {result['tokens_per_second']} t/s)")
            results.append(result)

    return results


def print_summary(results: list):
    """Print a summary table of results."""
    table = Table(title="Eval Results Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Content", justify="right")
    table.add_column("Thinking", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        status = "[green]PASS[/green]" if r["status"] == "pass" else "[red]FAIL[/red]"
        table.add_row(
            r["task_name"],
            r["category"],
            status,
            f"{r['content_length']} chars",
            f"{r['thinking_length']} chars",
            f"{r['tokens_per_second']} t/s",
            f"{r['elapsed_seconds']}s",
        )

    console.print(table)

    # Summary stats
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    content_populated = sum(1 for r in results if r["content_populated"])
    avg_speed = sum(r["tokens_per_second"] for r in results) / total if total else 0

    console.print(f"\n[bold]Pass rate:[/bold] {passed}/{total} ({100*passed//total}%)")
    console.print(f"[bold]Content populated:[/bold] {content_populated}/{total}")
    console.print(f"[bold]Avg speed:[/bold] {avg_speed:.1f} t/s")


def save_results(results: list, output_dir: str = None):
    """Save results to JSON file."""
    output_dir = Path(output_dir or Path(__file__).parent.parent / "results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(results),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/green]")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Gemma4 Eval Runner")
    parser.add_argument("--endpoint", required=True, help="llama-server endpoint URL")
    parser.add_argument("--model", default="gemma4", help="Model name")
    parser.add_argument("--task", choices=list(TASK_MODULES.keys()), help="Run specific task category")
    parser.add_argument("--config", help="Path to config YAML")
    parser.add_argument("--output", help="Output directory for results")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    results = run_evals(args.endpoint, args.model, args.task, config)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
