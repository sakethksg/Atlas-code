"""
Benchmark data loaders — HumanEval, MBPP.

Provides a single source of truth for loading benchmark datasets.
All components (evaluation runners, training pipeline) import from
here instead of duplicating loading logic.
"""

from afdad.datasets.loaders import load_humaneval, load_mbpp, load_humaneval_as_list

__all__ = ["load_humaneval", "load_mbpp", "load_humaneval_as_list"]

