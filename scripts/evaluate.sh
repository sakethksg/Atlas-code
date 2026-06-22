#!/bin/bash
# Standalone evaluation runner script for HumanEval and MBPP.
#
# Usage:
#   ./scripts/evaluate.sh humaneval
#   ./scripts/evaluate.sh mbpp
#   ./scripts/evaluate.sh humaneval,mbpp

set -e

BENCHMARK=${1:-"humaneval,mbpp"}

echo "=========================================================="
echo "AFDAD Benchmark Evaluation Runner"
echo "Target Benchmarks: $BENCHMARK"
echo "=========================================================="

# Check if uv is installed, otherwise fallback to python
if command -v uv &> /dev/null; then
    uv run python -m afdad.main mode=evaluate evaluation.benchmarks="[$BENCHMARK]"
else
    python -m afdad.main mode=evaluate evaluation.benchmarks="[$BENCHMARK]"
fi

echo "Evaluation completed successfully."
