#!/bin/bash
# Setup llama-server on a Vast.ai instance with Gemma4 31B Q6_K
# Identical config across all instances for clean benchmarking
#
# Usage: ssh vastai-instance 'bash -s' < scripts/vastai_setup_llama_server.sh

set -euo pipefail

echo "=== llama-server setup for GPU benchmarking ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown')"

# Install deps if not present
apt-get update -qq
apt-get install -y -qq cmake build-essential git wget curl libcurl4-openssl-dev >/dev/null 2>&1

# Build llama.cpp from source (latest release)
echo "=== Building llama.cpp ==="
cd /root
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cmake --build build --config Release -j$(nproc) --target llama-server 2>&1 | tail -5
cp build/bin/llama-server /usr/local/bin/
echo "llama-server built: $(llama-server --version 2>&1 | head -1 || echo 'ok')"

# Download model — Gemma4 31B Q6_K (same as zion-training production)
echo "=== Downloading model ==="
MODEL_DIR=/root/models
mkdir -p "$MODEL_DIR"
MODEL_FILE="$MODEL_DIR/gemma-4-27b-it-Q6_K.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    # Use huggingface-cli or wget
    wget -q --show-progress -O "$MODEL_FILE" \
        "https://huggingface.co/bartowski/google_gemma-4-27b-it-GGUF/resolve/main/google_gemma-4-27b-it-Q6_K.gguf" \
        || echo "Direct download failed, trying alternate..."
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model download failed. Try manual download."
    exit 1
fi

echo "Model size: $(du -h "$MODEL_FILE" | cut -f1)"

# Start llama-server with IDENTICAL config to zion-training
# Key: same flags for fair comparison, only GPU-specific params differ
echo "=== Starting llama-server ==="
llama-server \
    --model "$MODEL_FILE" \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --parallel 1 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -fa on --jinja \
    --reasoning-format deepseek \
    --metrics \
    --port 8080 --host 0.0.0.0 \
    > /root/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "llama-server started (PID $LLAMA_PID)"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        echo "=== Setup complete ==="
        echo "Endpoint: http://$(hostname -I | awk '{print $1}'):8080"
        nvidia-smi
        exit 0
    fi
    sleep 2
done

echo "ERROR: Server did not start in 120s"
cat /root/llama-server.log | tail -20
exit 1
