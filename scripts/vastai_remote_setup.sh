#!/bin/bash
# Run on Vast.ai instance: build llama.cpp + download model + start server
# Usage: ssh root@instance 'bash -s' < scripts/vastai_remote_setup.sh
set -euo pipefail

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Installing build deps ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake build-essential git wget curl libcurl4-openssl-dev >/dev/null 2>&1
echo "Done"

echo "=== Building llama.cpp ==="
cd /root
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build build --config Release -j$(nproc) --target llama-server 2>&1 | tail -3
cp build/bin/llama-server /usr/local/bin/
echo "llama-server built OK"

echo "=== Downloading Gemma4 27B Q6_K (~24GB) ==="
mkdir -p /root/models
MODEL=/root/models/gemma4-27b-it-Q6_K.gguf
if [ ! -f "$MODEL" ]; then
    wget --progress=dot:giga -O "$MODEL" \
        "https://huggingface.co/bartowski/google_gemma-4-27b-it-GGUF/resolve/main/google_gemma-4-27b-it-Q6_K.gguf"
fi
ls -lh "$MODEL"

echo "=== Starting llama-server ==="
nohup llama-server \
    --model "$MODEL" \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --parallel 1 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -fa on --jinja \
    --reasoning-format deepseek \
    --metrics \
    --port 8080 --host 0.0.0.0 \
    > /root/llama-server.log 2>&1 &

echo "Waiting for llama-server to be healthy..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "SERVER READY after ${i}s"
        curl -s http://localhost:8080/health
        echo ""
        echo "=== SETUP COMPLETE ==="
        exit 0
    fi
    sleep 2
done
echo "ERROR: server did not start"
tail -30 /root/llama-server.log
exit 1
