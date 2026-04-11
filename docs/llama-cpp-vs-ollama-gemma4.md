# Gemma4 on llama.cpp vs Ollama: Issue Comparison

> Last updated: 2026-04-10
> Context: ArtemisAI migrated Gemma4 31B production inference from Ollama to llama.cpp (llama-server) on zion-training

## Summary

Moving from Ollama to llama.cpp **fixes all client-facing blockers** (empty content, broken thinking control, structured output). Remaining issues are shared engine-level bugs that affect both.

## Issue Comparison Table

### Client-Facing Issues (Production Blockers)

| Issue | Ollama Status | llama.cpp Status | Resolution |
|-------|--------------|------------------|------------|
| Empty `content` field — all output goes to `reasoning` | BROKEN — `/v1/chat/completions` doesn't forward `think` param | **FIXED** — `--reasoning-format none` puts all output in `content` | Server-level flag |
| No thinking control on OpenAI-compat endpoint | BROKEN — `think` param not wired in `/v1/` (#15293) | **FIXED** — `--reasoning-format` controls at server level | Server-level flag |
| `think=false` breaks structured output / `format` | BROKEN — Ollama interaction bug (#15260) | **WORKS** — grammar constraints independent of reasoning mode | Not an issue |
| Tool call JSON parse failures | BROKEN — Ollama's Go `Gemma4.go` parser adds bugs (#15445, #15315) on top of engine issues | **PARTIAL** — llama.cpp PEG parser still has issues (#21384, #21375) but one fewer layer of breakage | Better, not perfect |
| Vision/multimodal not available | BROKEN — not wired in Ollama v0.20.0 (#15299) | **WORKS** — mmproj supported natively (crash bugs on large images exist) | Functional |
| Model loading/registry failures | Ollama-specific (#15236, #15447) | **N/A** — direct GGUF file, no registry | Not an issue |

### Shared Engine Issues (Affect Both)

| Issue | Category | llama.cpp Issue | Notes |
|-------|----------|----------------|-------|
| Unicode/accent dropping (French, Polish, German) | Tokenizer | #21578, Ollama #15234, #15270 | Shared tokenizer code — affects both identically |
| Vulkan/AMD gibberish output | Backend | #21516, #21336, #21400 | Same Vulkan compute code — Intel Arc, AMD iGPU |
| OOM / VRAM estimation errors | Memory | #21323, #21414, #21690 | Shared CUDA memory management |
| Infinite output / EOS not triggering | Generation | #21365 | Stop condition bug in engine |
| KV cache quantization gibberish (`-nkvo`) | Performance | #21726, #21686 | KV offload interaction |
| NCCL crash on Blackwell (RTX 5090) | CUDA | #21719 | System NCCL doesn't support sm_120a — workaround: rebuild without NCCL |
| Vision crashes on large images | Multimodal | #21427, #21550, #21461 | GGML assertion / bounds checking |
| ARM64 + Blackwell segfault (DGX Spark) | Platform | #21730, Ollama #15318 | Scheduler split inputs limit |
| MoE scheduling performance | Performance | #21424, Ollama #15286 | Attention overhead in MoE variants |
| `--parallel` broken with Gemma4 | Server | #21329 | Multi-slot parallel decoding regression |
| Cache reuse not working despite `-fa` | Performance | #21468 | SWA hybrid attention limitation |

### Ollama-Only Issues (Not Present in llama.cpp)

| Issue | Ollama Issue | Description |
|-------|-------------|-------------|
| `think` not default in `/api/generate` | #15268, #15283 | Ollama routing code |
| Empty response on long system prompts | #15428 | Ollama may truncate before dispatch |
| `tool_calls` index always 0 in streaming | #15457 | Ollama streaming response assembly |
| Windows memory layout regression | #15352 | Ollama CPU buffer allocation for vision |
| APU GPU/CPU split allocation | #15285 | Ollama device selection logic |
| Streaming 500 on OpenAI-compat | #15379 | Ollama Go HTTP handler |

## Thinking Mode Behavior

### Key Finding
Gemma4 31B **always thinks** — even with thinking "disabled", the model generates thinking tags with an empty thought block (per Google's model card). The `--reasoning-format` flag only controls where the output goes, not whether the model thinks.

### Configuration Options

| Flag | Behavior | Use Case |
|------|----------|----------|
| `--reasoning-format deepseek` | Splits output: thinking → `reasoning_content`, answer → `content` | When clients handle `reasoning_content` |
| `--reasoning-format none` | All output (thinking + answer) → `content` field | **Production default** — works with all OpenAI-compat clients |
| No `--jinja` flag | Chat template not applied, raw completions | Not recommended for chat |

### Thinking Tags in Content
With `--reasoning-format none`, content looks like:
```
<|channel>thought
[internal reasoning here]
<channel|>[actual answer here]
```
Clients receive the full stream. A proxy-level tag stripper could clean this up.

### Google's Recommended Sampling
- Temperature: 1.0
- Top-p: 0.95
- Top-k: 64

## Benchmark Performance

### Our Benchmarks (Q6_K / Q4_K_M, llama-server)

| GPU | Quant | Tokens/sec | Relative |
|-----|-------|-----------|----------|
| RTX 3090 (24GB) | Q6_K | ~18.4 t/s | 1.0x (baseline) |
| RTX 4090 (24GB) | Q4_K_M | ~39.9 t/s | 2.17x |
| RTX 5090 (32GB) | Q4_K_M | Blocked by NCCL | See workaround above |

### Google's Coding Benchmarks (31B model)

| Benchmark | Score |
|-----------|-------|
| LiveCodeBench v6 | 80.0% |
| Codeforces ELO | 2150 |
| AIME 2026 (no tools) | 89.2% |

### RPI Paper Findings (arXiv:2604.07035)

- Gemma-4-E4B achieved best weighted accuracy (0.675) at only 14.9GB VRAM
- Gemma-4-26B-A4B close second (0.663) but needs 48GB VRAM
- Gemma models dominated ARC and Math benchmarks vs Phi-4 and Qwen3
- Few-shot CoT was the best prompting strategy for 6/7 models tested
- MoE parameter count alone doesn't predict deployment efficiency

## Production Configuration

### Current Setup (zion-training)
```bash
# /etc/systemd/system/llama-server.service
ExecStart=/opt/llama.cpp/bin/llama-server \
    --model /opt/models/gemma4-31b-Q6_K.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --split-mode layer \
    --parallel 1 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -fa on \
    --jinja \
    --reasoning-format none \
    --port 8080 --host 0.0.0.0 \
    --metrics
```

### NewAPI Routing
- Channel #174: `gemma4:31b` → `http://100.70.28.111:8080`
- Model mapping: `gemma4:31b` → `gemma4-31b-Q6_K.gguf`
- Public endpoint: `https://llm.ai-automate.me/v1/chat/completions`

## References

- [ArtemisAI/gemma-4-eval](https://github.com/ArtemisAI/gemma-4-eval) — Benchmark scripts and results
- [llama.cpp Gemma4 issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+gemma4) — 35+ open issues
- [Ollama Gemma4 issues](https://github.com/ollama/ollama/issues?q=is%3Aissue+is%3Aopen+gemma4) — 39+ open issues
- [Google Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [HuggingFace Gemma 4 Blog](https://huggingface.co/blog/gemma4)
- [arXiv:2604.07035](https://arxiv.org/abs/2604.07035) — Dense vs MoE Reasoning Benchmarks
