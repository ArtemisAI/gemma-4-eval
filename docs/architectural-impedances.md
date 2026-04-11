# Architectural Impedances in Gemma 4 Local Inference Pipelines

**Date:** 2026-04-10  
**Models:** gemma4:31b (dense), gemma4:26b-A4b (MoE), gemma4:e4b, gemma4:e2b  
**Status:** Active investigation — documents systemic failures from model arch through middleware to proxy layers

---

## 1. Model Architecture Summary

### 1.1 Dense vs MoE Variants

| Variant | Total Params | Active/Token | Context Window | Key Feature |
|---|---|---|---|---|
| gemma4:31b | 31B | 31B | 256K | Dense, all params active every token |
| gemma4:26b (MoE) | 25.2B | ~3.8B | 256K | 128 experts + 1 shared, top-2 routing |
| gemma4:e4b | 7.9B | 4.5B | 128K | Dense, per-layer embeddings |
| gemma4:e2b | 5.1B | 2.3B | 128K | Dense, per-layer embeddings, audio support |

**MoE economics:** Only 3.8B params active per token (top-2 of 128 experts + 1 shared). Inference compute is comparable to a 4B dense model, but all 25.2B weights must still reside in VRAM. Q4_K_M file estimated at ~14-15GB, runtime ~18-20GB — fits in 24GB with headroom.

**Fleet implications:**
- MoE variant could replace 31B dense on 24GB machines, freeing VRAM for longer contexts or NUM_PARALLEL=2
- Priority: benchmark gemma4:26b-A4b on zion-training

### 1.2 Interleaved Sliding Window Attention (iSWA)

Gemma 4 alternates attention layers:

- **Sliding-window layers:** Local attention (512-1024 token radius). O(n) per token.
- **Global layers:** Full-context attention. O(n²) per token.
- **Final layer:** Always global — synthesizes localized computations into cohesive output.

**Impact on KV cache:** At 256K context, full global attention would require ~8GB fp16 KV cache per request on 31B. This is why our q4_0 KV compression was critical — it shrank this from ~8GB to ~2GB.

### 1.3 Shared Key-Value Caching

Final decoder layers reuse KV tensors from preceding non-shared layers rather than computing independent projections.

**Benefits:** Reduced VRAM, lower compute latency during prefill.  
**Cost:** Breaks assumptions in inference engines about sequential tensor independence. Requires rewrites of prompt caching and context shifting algorithms.

**Impact on llama.cpp:** Upstream issue ggml-org/llama.cpp#21468 — cache reuse broken with shared KV. Partial fix in PR #21513.

### 1.4 Per-Layer Embeddings (E2b/E4b only)

Smaller variants inject localized embedding lookup tables into every decoder layer. These provide a secondary conditioning signal via cheap lookups (no dense matmul), boosting representational capacity from 4.5B effective params to higher-quality output.

**Fleet relevance:** E2b variant would be the right target for zero-one-inference's 8GB AMD RX 6600 XT once ROCm KFD is fixed.

### 1.5 Dual Rotary Position Embeddings

- **Standard RoPE:** Applied to sliding-window layers (local syntactic relationships)
- **Proportional RoPE:** Applied to global layers (prevents quality degradation at extreme context distances)

No inference-engine impact; handled natively by llama.cpp and vLLM.

---

## 2. Dual-Channel Reasoning Architecture

### 2.1 Thought Channel Mechanics

Gemma 4 generates two distinct output channels per request:

1. **Thought channel:** `<|channel>thought\n` ... `<channel|>` — internal reasoning tokens  
2. **Content channel:** Standard text output — the user-facing answer

Activation: The `﹄` or `<|think|>` token must be injected at the start of the system prompt. Without it, the model still emits the structural markers but with an empty thought block.

### 2.2 Disabled Thinking Produces Empty Markers (NOT Absent Markers)

When `ﴽ` is omitted, the model outputs:

```
<|channel>thought\n<channel|>
```

...then proceeds to the final answer. Any middleware parsing the stream MUST handle empty thought blocks — failure to do so causes pipeline collapse.

### 2.3 Per-Request Thinking Budget

llama-server supports `thinking_budget_tokens` as a per-request parameter (not available in Ollama's OpenAI translation). This caps the reasoning phase token count, reducing generation time for simple queries.

**Benchmarks (zion-training, Q6_K):**

| Config | Speed | Thinking | Content | Quality |
|---|---|---|---|---|
| Default (full thinking) | 18.5 t/s | 2,359 chars | 2,102 chars ✅ | Correct + full proof |
| `thinking_budget_tokens=500` | 18.5 t/s | 1,680 chars | 2,138 chars ✅ | Same quality, 17% faster |

---

## 3. Middleware Translation Defects

### 3.1 Ollama `/v1/chat/completions` — Empty Content Field (CRITICAL)

**Status:** SOLVED by switching to llama-server

The Ollama OpenAI-compatible endpoint (`/v1/chat/completions`) routes the entire token stream — both thought channel and content channel — into the non-standard `reasoning` field, leaving `content` permanently empty.

| Field | Should contain | Actually contains |
|---|---|---|
| `choices[0].message.content` | Final answer | `""` (empty string) |
| `choices[0].message.reasoning` | N/A (non-standard) | Full model output (thinking + answer) |

**Severity:** CATASTROPHIC for any client using standard OpenAI SDK. The model generates a perfect response, but it's invisible to the client.

**Streaming exacerbation:** Under SSE, every delta chunk populates `reasoning` and starves `content`. The UI shows nothing until the entire stream completes, then renders blank.

### 3.2 Ollama `think: false` Parameter Ignored

The `/v1/chat/completions` endpoint silently drops the `think` parameter. The model always receives the `ﴽ` trigger regardless, generating massive reasoning traces the user never requested. No way to disable thinking via the OpenAI-compatible endpoint.

### 3.3 llama-server Resolution (Production)

Switching to llama-server with `--reasoning-format deepseek --jinja` produces correct output:

- `reasoning_content` field: Contains the thought channel output
- `content` field: Contains the final answer (always populated)
- `thinking_budget_tokens`: Per-request control (Ollama lacks this)

**Production config (zion-training):**

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

### 3.4 Tool Calling Defects

Tracked in eval task `evals/tasks/tool_use.py`:

| Issue | Severity | Description | Status |
|---|---|---|---|
| llama.cpp#21375 | High | Infinite repetition loop with PEG-gemma4 parser when thinking + tool calling interact | Active discussion |
| llama.cpp#21384 | Medium | Array params serialized as JSON string containing `{` or `}` characters | Active |
| llama.cpp#21316 | Medium | Unexpected tokens after tool calls | Active |
| llama.cpp#21468 | Medium | Cache reuse broken due to shared KV architecture | Partial fix PR #21513 |
| llama.cpp#21329 | Low | `--parallel` broken | Workaround: `--parallel 1` |

### 3.5 MoE Variant Specific Issue: Premature Halting

When gemma4:26b is fed a system prompt exceeding 500 characters, the model frequently halts after ~49 tokens, returning an empty response (both channels). This appears to be a model-level defect, not middleware.

**Impact:** Could block MoE deployment for RAG or long system-prompt use cases. Needs eval before fleet adoption.

---

## 4. Hardware Dynamics and Memory Bandwidth

### 4.1 The CPU Offload Performance Cliff

When model weights + KV cache exceed VRAM, inference engines offload the least-critical layers to system RAM. The PCIe bus bandwidth (~32-64 GB/s) is a tiny fraction of GPU memory bandwidth (~1,008 GB/s on RTX 4090).

**For Gemma 4, this is catastrophic because:**
- The model generates thousands of reasoning tokens before any user-visible output
- Every token in the forward pass must cross the PCIe bottleneck
- A reasoning trace of 2,000 tokens at 3.9 t/s takes ~512 seconds — many proxy clients timeout at 300s

**Our fleet benchmarks (CPU offload penalty):**

| Machine | Config | CPU/GPU Split | Speed | vs Full-GPU |
|---|---|---|---|---|
| artemisai-local | fp16 KV, NUM_PARALLEL=2 | 38% CPU / 62% GPU | 3.9 t/s | baseline |
| artemisai-local | q4_0 KV + flash attn | 17% CPU / 83% GPU | 8.4 t/s | +115% |
| vast.ai (RTX 4090) | fp16 KV | 13% CPU / 87% GPU | 9.5 t/s | baseline |
| vast.ai (RTX 4090) | q4_0 KV + flash attn | 100% GPU | 43.2 t/s | +355% |
| io-training | q4_0 KV + flash attn | 32% CPU / 68% GPU | 3.6 t/s | — |

Even 13% CPU offload on an RTX 4090 costs 78% of throughput.

### 4.2 Context Depth vs Throughput (RTX 4090 Reference)

| Context Depth | Generation Velocity | VRAM Used | Operational State |
|---|---|---|---|
| 8,000 tokens | 29.8 t/s | 20.9 GB | Fully native VRAM |
| 64,000 tokens | 29.6 t/s | 22.3 GB | Fully native VRAM |
| 128,000 tokens | 29.6 t/s | 23.9 GB | Fully native VRAM (q4_0 KV) |
| 200,000 tokens | 14.1 t/s | >24.0 GB | Severe system RAM offloading |
| 256,000 tokens | 10.0 t/s | >24.0 GB | Catastrophic offloading + thermal throttling |

**Fleet implication:** Our 16,384-32,768 context windows are well within the safe zone. No need to extend beyond that for current workloads.

### 4.3 KV Cache Quantization: The Highest-ROI Optimization

| Optimization | Effect | Quality Impact |
|---|---|---|
| q4_0 KV cache | ~4× compression (fp16 ~8GB → q4_0 ~2GB at 32K ctx) | Minor attention quality reduction |
| q8_0 KV cache | ~2× compression | Negligible quality reduction |
| Flash Attention | Reduces memory fragmentation | None |
| NUM_PARALLEL=1 | Frees VRAM headroom | Reduces concurrent request capacity |

**Production recommendation:** q4_0 KV + flash attention on all machines. Accept the minor quality hit for the 2-4× speed boost.

---

## 5. Fleet-Specific Hardware Impedances

### 5.1 NVIDIA Tensor Parallelism Blocked (Issue #65)

**Problem:** GCC 15 (Fedora 43) is incompatible with CUDA 12.8. vLLM requires CUDA 12.9+ for Gemma4.

**Resolution paths:**
1. **vLLM via Docker** — `vllm/vllm-openai:gemma4` (CUDA 12.9) bypasses host GCC/CUDA conflict
2. **Wait for CUDA 12.9+ packages** for Fedora 43
3. **Cross-compile llama.cpp with CUDA 12.9** in a container

**Recommendation:** Docker-based vLLM deployment. Also enables `--async-scheduling` and native MoE support.

### 5.2 AMD ROCm: /dev/kfd Missing (Issue #57)

**Problem:** zero-one-inference (AMD RX 6600 XT) has no `/dev/kfd` — ROCm KFD kernel support absent.

**Impact:** GPU completely bypassed. All models run 100% CPU (1.51 t/s for 31B, 9.2 t/s for e4b — but e4b runtime exceeds 8GB VRAM).

**Resolution steps:**
1. Install ROCm kernel driver package for Fedora 43
2. Verify `/dev/kfd` and `/dev/dri/renderD128` appear
3. Test with E2b model (~2-3GB runtime) for true GPU inference
4. vLLM ROCm Docker image: `vllm/vllm-openai-rocm:gemma4` (requires ROCm 7.2.1)

### 5.3 mc-inference: Hardware Down (Issue #54)

AMD RX 7900 XTX (24GB). GPU passthrough crash in Proxmox. Needs physical power cycle.

**If restored:** 24GB VRAM would be equivalent to a single RTX 3090 — another full-GPU inference node.

### 5.4 niobe: GPU Occupied (Issue #60)

RTX 3060 12GB fully occupied by production Python services since March 12. `CUDA_VISIBLE_DEVICES=-1` forces CPU-only.

**Resolution:** Either migrate services off niobe or accept it as embedding-only.

### 5.5 NewAPI Routing Update (Issue #66)

io-training, niobe, and zero-one-inference should be removed from inference channels and added to embedding-only channels in NewAPI. artemisai-local removed from inference (root disk 99% full).

---

## 6. vLLM Deployment Considerations

### 6.1 Why vLLM Over Ollama/llama-server

| Feature | Ollama | llama-server | vLLM |
|---|---|---|---|
| OpenAI compat | Broken (empty content) | ✅ (with --reasoning-format) | ✅ Native |
| Thinking control | `think` ignored on /v1 | Per-request ✅ | Per-request ✅ |
| MoE support | Basic | Basic | Optimized |
| Tensor parallelism | No cross-GPU | Layer split only | ✅ Native |
| Continuous batching | Limited | Limited | ✅ Optimized |
| CUDA requirement | 12.x | 12.x | 12.9+ for Gemma4 |

### 6.2 Docker Deployment Path

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:gemma4 \
  --model google/gemma-4-31b-it \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype q4_0 \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --async-scheduling
```

**Blocked by:** Issue #65 (CUDA 12.8 on Fedora 43). Docker resolves this by shipping its own CUDA 12.9 runtime.

### 6.3 MoE (gemma4:26b-A4b) Deployment

```bash
# vLLM has native MoE optimization
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:gemma4 \
  --model google/gemma-4-26b-it \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --enable-prefix-caching
```

Estimated: ~18-20GB VRAM at Q4 — fits in 24GB with headroom for longer contexts.

---

## 7. Open Issues Tracker

| # | Issue | Priority | Machine | Status |
|---|---|---|---|---|
| #54 | mc-inference GPU passthrough crash — needs physical power cycle | P1 | mc-inference | DOWN |
| #57 | AMD /dev/kfd missing — no ROCm KFD kernel support | P1 | zero-one | Blocked on kernel |
| #60 | niobe GPU occupied by production services since Mar 12 | P2 | niobe | Embedding-only |
| #65 | GCC 15 incompatible with CUDA 12.8 — blocks tensor parallel | P1 | Fedora hosts | Docker workaround |
| #66 | NewAPI routing: remove non-inference machines, add to embedding channels | P2 | NewAPI | Pending |
| llama.cpp#21375 | Tool call infinite loops (PEG parser) | P1 | All | Upstream active |
| llama.cpp#21384 | Array params with `{` `}` serialization | P2 | All | Upstream active |
| llama.cpp#21316 | Unexpected tokens after tool calls | P2 | All | Upstream active |
| llama.cpp#21468 | Cache reuse broken (shared KV) | P2 | All | Partial fix #21513 |
| llama.cpp#21329 | `--parallel` broken | P3 | All | Workaround: `--parallel 1` |
| MoE halting | 26b model halts at ~49 tokens with system prompts >500 chars | P1 | TBD | Under investigation |

---

## 8. Action Items

1. **Benchmark gemma4:26b-A4b on zion-training** — validate quality vs 31B and check for >500 char prompt halting
2. **Deploy vLLM via Docker on zion-training** — resolve CUDA 12.8 blocker (issue #65)
3. **Physical power cycle mc-inference** — restore 24GB AMD GPU (issue #54)
4. **Install ROCm kernel driver on zero-one** — enable `/dev/kfd` (issue #57)
5. **Update NewAPI routing** — demote non-inference machines to embedding channels (issue #66)
6. **Free disk space on artemisai-local** — investigate root disk 99% full, restore to inference rotation
7. **Write eval for MoE halting bug** — system prompt >500 chars, verify output completeness
8. **Evaluate vLLM `--async-scheduling`** for throughput gains on multi-request workloads