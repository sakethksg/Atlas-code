"""Code execution and test evaluation.

Provides subprocess-based code execution with timeout enforcement,
stdout/stderr capture, and structured result parsing. Bridges generated
code with HumanEval/MBPP-style test formats.
"""

from afdad.execution.executor import CodeExecutor
from afdad.execution.evaluator import Evaluator

__all__ = ["CodeExecutor", "Evaluator"]
