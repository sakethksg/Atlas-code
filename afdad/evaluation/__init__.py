"""Benchmark evaluation — HumanEval, MBPP.

Provides benchmark runners that evaluate the student model's code
generation capabilities and compute Pass@k metrics using the unbiased
estimator from Chen et al. (2021).
"""

from afdad.evaluation.humaneval import HumanEvalRunner
from afdad.evaluation.mbpp import MBPPRunner
from afdad.evaluation.benchmark_base import BaseBenchmarkRunner
from afdad.evaluation.metrics import estimate_pass_at_k, compute_pass_at_k_batch

__all__ = [
    "HumanEvalRunner",
    "MBPPRunner",
    "BaseBenchmarkRunner",
    "estimate_pass_at_k",
    "compute_pass_at_k_batch",
]

