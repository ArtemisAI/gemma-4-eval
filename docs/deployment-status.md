# Gemma4 Deployment Status Report
**Last updated:** 2026-04-11
**Companion docs:** [Architectural Impedances](./architectural-impedances.md) | [Fleet Benchmark](/home/artemisai/sys_admin/proxmox/gpu_fleet_benchmark_2026-04-10.md)

---

## 1. Inference Backends (Running)

| Backend | Machine | Runtime | Model/Quant | Context | Speed | Endpoint | Status |
|---|---|---|---|---|---|---|---|
| llama-server | zion-training | llama.cpp b1-0893f50, CUDA 12.6 | gemma4-31b Q6_K | **131072** | 15.7-18.5 t/s | `http://100.70.28.111:8080/v1` | **Production** |
| llama-server | io-training | llama.cpp b1-0893f50, CUDA 12.6 | gemma4-31b Q4_K_M | **65536** | ~15 t/s | `http://100.68.161.101:8080/v1` | **Production** |
| llama-server | artemisai-local | llama.cpp b1-0893f50 (io binary), CUDA 13.0 | gemma4-31b Q4_K_M | **65536** | ~16.4 t/s | `http://100.86.157.4:8080/v1` | **Production** |
| Ollama | vast.ai | Ollama (Docker) | gemma4:31b Q4_K_M | 32768 | 43.3 t/s | `http://108.255.76.60:55482` (SSH tunnel) | Active |

### GPU Pooling & Parallelism

All llama-server nodes use `--split-mode layer` to pool multiple GPUs:

- **zion-training**: RTX 3090 (24GB) + RTX 3060 (12GB) = **36GB pooled VRAM**
  - Layer splitting: ~65% of layers on GPU0, ~35% on GPU1
  - VRAM: GPU0 19.3GB / GPU1 10.9GB = 30.2GB used, 5.7GB headroom
- **io-training**: 2× RTX 3060 (12GB each) = **24GB pooled VRAM**
  - Layer splitting: ~50/50 across both GPUs
  - VRAM: GPU0 10.5GB / GPU1 10.7GB = 21.3GB used, 2.6GB headroom
- **artemisai-local**: 2× RTX 3060 (12GB each) = **24GB pooled VRAM**
  - Layer splitting: ~50/50 across both GPUs
  - VRAM: GPU0 11.2GB / GPU1 11.5GB = 22.6GB used, 1.2GB headroom
  - Ollama removed, replaced with llama-server (2026-04-11)

**Note on tensor parallelism modes:**
- `--split-mode layer` (current): Distributes transformer layers across GPUs. Each GPU processes its assigned layers sequentially. No inter-GPU communication during forward pass. Works with any GPU combination.
- `--split-mode row` (true tensor parallelism): Splits weight matrices across GPUs, processes in parallel. Requires NCCL. Not currently used — our build lacks NCCL, and benchmarks showed layer split was faster without it.

### strip-gemma proxy

**Status: STOPPED (2026-04-11)** — Redundant with `--reasoning-format auto` on current llama-server builds (b1-0893f50+, includes PR #21418 Gemma4 PEG parser). Binary remains at `/usr/local/bin/strip-gemma` on both nodes for community use. Service disabled on boot.

**For community users** (KoboldCpp, vLLM, older llama.cpp): strip-gemma is published at [github.com/ArtemisAI/strip-gemma](https://github.com/ArtemisAI/strip-gemma).

---

## 2. Context Window Configuration

### Architecture enables large context cheaply

Gemma4's **interleaved sliding window attention (iSWA)** + **shared KV caching** + **q4_0 KV quantization** means context scaling costs far less VRAM than traditional transformers:

| Context Size | VRAM Delta (vs 8k) | Speed Impact | Notes |
|---|---|---|---|
| 8K | baseline | 29.8 t/s (4090 ref) | — |
| 32K | +1.4 GB | 29.6 t/s | Previous config |
| 64K | +1.4 GB | 29.6 t/s | io-training max safe |
| **128K** | **+3.0 GB** | **29.6 t/s** | **zion-training current** |
| 200K | +8+ GB | 14.1 t/s | Offloading starts on 24GB |
| 256K (model max) | +12+ GB | 10.0 t/s | Only viable on 48GB+ |

### Current deployment

| Machine | `--ctx-size` | Rationale |
|---|---|---|
| zion-training | **131072** | 36GB VRAM, 5.7GB headroom at this size |
| io-training | **65536** | 24GB VRAM, 2.6GB headroom — 131k would OOM |

### Why this matters for coding agents

A coding agent session can easily fill 50-100k tokens with codebase context, conversation history, tool results, and system prompts. With 131k context on zion-training, a request can include ~120k tokens of context and still generate ~10k tokens of response — sufficient for real-world agent workloads.

---

## 3. Thinking Token Budget

### How it works

Gemma4 always thinks before answering. The thinking tokens and content tokens **share the `max_tokens` budget**. If `max_tokens` is too low, thinking consumes the entire budget and `content` is empty.

### Per-request control

llama-server supports `thinking_budget_tokens` per request to cap reasoning:

| Config | Speed | Thinking | Content | Quality |
|---|---|---|---|---|
| Default (unlimited thinking) | 18.5 t/s | 2,359 chars | 2,102 chars | Full proof |
| `thinking_budget_tokens=500` | 18.5 t/s | 1,680 chars | 2,138 chars | Same quality, 17% faster |

### Minimum `max_tokens` guidelines

| Use case | Minimum `max_tokens` | Recommended |
|---|---|---|
| One-liner answer | 512 | 1024 |
| Code generation | 2048 | 4096-8192 |
| Complex reasoning | 2048 | 4096+ |
| Agent coding session | 4096 | 8192-16384 |

---

## 4. NewAPI Proxy Wiring

### Registered channels
- **zion-training** llama-server (`:8080` direct — strip-gemma bypassed)
- **io-training** llama-server (`:8080` direct)
- **vast.ai** Ollama endpoint

### Machines to demote to embedding-only (issue #66)
- niobe (GPU occupied by production services)
- zero-one-inference (AMD /dev/kfd missing)

### Empty content bug (issue #59)
**RESOLVED for llama-server nodes.** `--reasoning-format auto` properly separates `content` and `reasoning_content`.
**Still affects vast.ai** (Ollama backend). Needs proxy-level fix or switch to llama-server.

---

## 5. pi-agent Configuration

**Config:** `~/.pi/agent/models.json`
- Points to `http://100.70.28.111:8080/v1` (direct to llama-server, no strip-gemma proxy)
- `contextWindow: 131072` (matches server config)

---

## 6. Observability Stack

| Component | Status | Endpoint |
|---|---|---|
| Prometheus | Running | Scraping llama-server metrics |
| Loki | v3.7.1 on artemisai-local | journald via Grafana Alloy on zion-training |
| Grafana | 9 panels | http://localhost:3000/d/llama-server-gemma4/ |
| Prometheus alerts | Configured | LlamaServerDown, SlowGen, HighQueue, NoActivity |

---

## 7. Evaluation Results

**Full eval suite: 14/14 PASS** (2026-04-10, Q6_K on zion-training)

Categories tested: reasoning (4), coding (3), system design (3), tool use (4)
All tasks: `content` populated, `reasoning_content` separated correctly.

Speed: 18.1-18.6 t/s across all tasks.

### Benchmark comparisons

| GPU | Quant | Decode t/s | Prefill t/s |
|---|---|---|---|
| RTX 3090+3060 (zion) | Q6_K | 18.5 t/s | 400 t/s |
| RTX 4090 (vast.ai) | Q6_K | 39.9 t/s | 943 t/s |
| 2× RTX 3060 (io) | Q4_K_M | ~15 t/s | ~280 t/s |

---

## 8. Open Issues

| # | Issue | Priority | Status |
|---|---|---|---|
| #54 | mc-inference GPU passthrough crash | P1 | DOWN — needs physical power cycle |
| #57 | AMD /dev/kfd missing on zero-one | P1 | Blocked on kernel driver |
| #59 | Empty content field | P0 | **RESOLVED** (llama-server), still affects Ollama |
| #60 | niobe GPU occupied | P2 | Embedding-only |
| #65 | GCC 15 / CUDA 12.8 incompatibility | P1 | Docker workaround available |
| #66 | NewAPI channel routing update | P2 | Pending |
| MoE halting | 26b halts at ~49 tokens with long system prompts | P1 | Under investigation |

---

## 9. Recovery Information

If this session is lost, the critical state is:

1. **llama-server services**: Check `systemctl status llama-server` on zion-training and io-training
2. **Service files**: `/etc/systemd/system/llama-server.service` on each node
3. **Models**: `/opt/models/` on each node
4. **Binaries**: `/opt/llama.cpp/bin/llama-server` (zion), `/opt/llama.cpp/build/bin/llama-server` (io)
5. **pi-agent config**: `~/.pi/agent/models.json` on artemisai-local
6. **strip-gemma**: Installed but disabled. Binary at `/usr/local/bin/strip-gemma`, service at `/etc/systemd/system/strip-gemma.service`
7. **Eval results**: `/home/artemisai/projects/gemma-4-eval/results/`
8. **Architecture docs**: `/home/artemisai/projects/gemma-4-eval/docs/`
