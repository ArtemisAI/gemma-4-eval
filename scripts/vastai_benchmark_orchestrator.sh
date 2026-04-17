#!/bin/bash
# Orchestrator: wait for Vast.ai instances, setup llama-server, run benchmarks, teardown
# Usage: bash scripts/vastai_benchmark_orchestrator.sh

set -euo pipefail

INSTANCE_4090=34554185
INSTANCE_5090=34554188
SSH_KEY=~/.ssh/vastai_overflow
MODEL_URL="https://huggingface.co/bartowski/google_gemma-4-27b-it-GGUF/resolve/main/google_gemma-4-27b-it-Q6_K.gguf"
BENCHMARK_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/scripts/gpu_benchmark.py"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"

echo "=== Vast.ai GPU Benchmark Orchestrator ==="
echo "4090 instance: $INSTANCE_4090"
echo "5090 instance: $INSTANCE_5090"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Wait for instances to be running
wait_for_instance() {
    local id=$1
    local name=$2
    echo "Waiting for $name (ID: $id) to be running..."
    for i in $(seq 1 60); do
        status=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
for inst in json.load(sys.stdin):
    if inst['id'] == $id:
        print(inst.get('actual_status', 'unknown'))
" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            echo "  $name is running!"
            return 0
        fi
        echo "  $name: $status (attempt $i/60)"
        sleep 10
    done
    echo "ERROR: $name did not start in 10 minutes"
    return 1
}

get_ssh_info() {
    local id=$1
    vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
for inst in json.load(sys.stdin):
    if inst['id'] == $id:
        print(f\"{inst.get('ssh_host','?')}:{inst.get('ssh_port','?')}\")
" 2>/dev/null
}

ssh_cmd() {
    local host_port=$1
    local host=$(echo "$host_port" | cut -d: -f1)
    local port=$(echo "$host_port" | cut -d: -f2)
    shift
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 \
        -i "$SSH_KEY" -p "$port" root@"$host" "$@"
}

# Step 1: Wait for both instances
wait_for_instance $INSTANCE_4090 "RTX 4090" &
wait_for_instance $INSTANCE_5090 "RTX 5090" &
wait

SSH_4090=$(get_ssh_info $INSTANCE_4090)
SSH_5090=$(get_ssh_info $INSTANCE_5090)
echo ""
echo "SSH 4090: $SSH_4090"
echo "SSH 5090: $SSH_5090"

# Step 2: Setup llama-server on both (in parallel)
echo ""
echo "=== Setting up llama-server on both instances ==="

setup_instance() {
    local ssh_info=$1
    local name=$2
    echo "[$name] Starting setup..."
    ssh_cmd "$ssh_info" 'bash -s' < "$(dirname "$0")/vastai_setup_llama_server.sh" 2>&1 | \
        sed "s/^/[$name] /"
    echo "[$name] Setup complete"
}

setup_instance "$SSH_4090" "4090" &
setup_instance "$SSH_5090" "5090" &
wait

# Step 3: Get endpoints
PORT_4090=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
for inst in json.load(sys.stdin):
    if inst['id'] == $INSTANCE_4090:
        ports = inst.get('ports', {})
        p = ports.get('8080/tcp', [{}])
        if isinstance(p, list):
            print(f\"{p[0].get('HostIp','?')}:{p[0].get('HostPort','?')}\")
        else:
            print(f\"{inst.get('public_ipaddr','?')}:8080\")
" 2>/dev/null)

PORT_5090=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
for inst in json.load(sys.stdin):
    if inst['id'] == $INSTANCE_5090:
        ports = inst.get('ports', {})
        p = ports.get('8080/tcp', [{}])
        if isinstance(p, list):
            print(f\"{p[0].get('HostIp','?')}:{p[0].get('HostPort','?')}\")
        else:
            print(f\"{inst.get('public_ipaddr','?')}:8080\")
" 2>/dev/null)

echo ""
echo "4090 endpoint: http://$PORT_4090"
echo "5090 endpoint: http://$PORT_5090"

# Step 4: Run benchmarks
echo ""
echo "=== Running benchmarks ==="

cd "$(dirname "$0")/.."

# Baseline + stress on 4090
python3 scripts/gpu_benchmark.py \
    --endpoint "http://$PORT_4090" \
    --model gemma4 \
    --gpu-name "RTX 4090" \
    --stress --runs 3 &

# Baseline + stress on 5090
python3 scripts/gpu_benchmark.py \
    --endpoint "http://$PORT_5090" \
    --model gemma4 \
    --gpu-name "RTX 5090" \
    --stress --runs 3 &

wait

# Step 5: Run 3090 baseline (local zion-training)
echo ""
echo "=== Running 3090 baseline (zion-training) ==="
python3 scripts/gpu_benchmark.py \
    --endpoint "http://100.70.28.111:8080" \
    --model gemma4 \
    --gpu-name "RTX 3090" \
    --stress --runs 3

# Step 6: Teardown
echo ""
echo "=== Tearing down Vast.ai instances ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
vastai destroy instance $INSTANCE_4090 2>&1
vastai destroy instance $INSTANCE_5090 2>&1
echo "Both instances destroyed. Benchmark complete."
