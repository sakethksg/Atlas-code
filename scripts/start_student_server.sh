#!/bin/bash
# ── Launch Student Model vLLM Server ──
# Qwen2.5-Coder-1.5B-Instruct on a single GPU

set -euo pipefail

export CUDA_VISIBLE_DEVICES="2"

MODEL_PATH="${1:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

echo "Starting Student vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Port: 8001"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 8192 \
    --trust-remote-code
