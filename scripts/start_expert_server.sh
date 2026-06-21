#!/bin/bash
# ── Launch Expert Model vLLM Server ──
# Qwen2.5-Coder-32B-Instruct with 2-GPU tensor parallelism

set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1"

MODEL_PATH="${1:-Qwen/Qwen2.5-Coder-32B-Instruct}"

echo "Starting Expert vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  Port: 8000"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 32768 \
    --trust-remote-code
