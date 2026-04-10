# gemma-4-eval

Evaluation framework, benchmarks, and telemetry for **Gemma4** inference on [llama.cpp](https://github.com/ggml-org/llama.cpp).

Built to track model quality, performance, and known issues — and to contribute fixes back to the community.

## What's Inside

```
evals/           # Evaluation tasks and runner
telemetry/       # Prometheus, Loki, Grafana configs
scripts/         # Utility scripts (pull models, benchmark, etc.)
results/         # Benchmark results and reports
.github/         # CI workflows for automated evals
```

## Quick Start

### Run evals against a llama-server instance

```bash
# Install deps
pip install -r requirements.txt

# Run all eval tasks
python -m evals.runner --endpoint http://localhost:8080/v1/chat/completions --model gemma4

# Run a specific category
python -m evals.runner --endpoint http://localhost:8080/v1/chat/completions --model gemma4 --task reasoning
```

### Set up telemetry

See [telemetry/README.md](telemetry/README.md) for Prometheus + Loki + Grafana setup.

## Eval Categories

| Category | Description | Key Metrics |
|---|---|---|
| **reasoning** | Multi-step logical reasoning, proofs, puzzles | Correctness, thinking tokens, content tokens |
| **coding** | Code generation with constraints (async, error handling) | Completeness, compilability, content field populated |
| **system_design** | Architecture questions at scale | Depth, capacity numbers, content field populated |
| **tool_use** | Function calling, JSON parameter handling | Parse success, delimiter correctness |

## Known Issues

We track all open Gemma4 issues in llama.cpp. See our [issue tracker](https://github.com/ArtemisAI/gemma-4-eval/issues) and the upstream issues we monitor:

- **Content field empty** on Ollama (thinking eats max_tokens) -- solved by llama-server + `--reasoning-format deepseek`
- **Tool call infinite loops** (ggml-org/llama.cpp#21375) -- PEG parser issue, active discussion
- **Cache reuse broken** (ggml-org/llama.cpp#21468) -- partial fix via PR #21513
- **--parallel broken** (ggml-org/llama.cpp#21329) -- use `--parallel 1` as workaround

## Deployment Reference

Our production setup on zion-training (RTX 3090 + RTX 3060, 36GB VRAM):

```bash
llama-server \
  --model gemma4-31b-Q6_K.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --split-mode layer \
  --parallel 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -fa on --jinja \
  --reasoning-format deepseek \
  --metrics \
  --port 8080 --host 0.0.0.0
```

Key flags:
- `--jinja` + `--reasoning-format deepseek`: Proper thinking/content separation
- `--cache-type-k q4_0 --cache-type-v q4_0`: KV cache compression for VRAM savings
- `--metrics`: Prometheus-compatible `/metrics` endpoint
- `thinking_budget_tokens` available per-request to cap thinking

## Contributing

We welcome contributions! Areas where help is needed:
- Additional eval tasks and datasets
- Grafana dashboard improvements
- Upstream llama.cpp bug fixes (see tracked issues)
- Performance optimization research

## License

MIT
