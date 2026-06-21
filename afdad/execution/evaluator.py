"""
Code evaluator — runs test cases against generated code and reports results.

Bridges the code executor with HumanEval/MBPP-style test formats.
"""

from __future__ import annotations

import asyncio
from typing import Any

from omegaconf import DictConfig

from afdad.execution.executor import CodeExecutor
from afdad.utils.logging import get_logger
from afdad.utils.models import ExecutionResult


class Evaluator:
    """Evaluates generated code against test cases.

    Parameters
    ----------
    execution_cfg:
        Execution configuration passed through to :class:`CodeExecutor`.
    """

    def __init__(self, execution_cfg: DictConfig) -> None:
        self.executor = CodeExecutor(execution_cfg)
        self.logger = get_logger()

    async def evaluate(
        self,
        code: str,
        test_code: str,
        entry_point: str = "",
    ) -> ExecutionResult:
        """Evaluate code against a single test suite.

        Parameters
        ----------
        code:
            Generated Python code.
        test_code:
            Test code (e.g., ``check(candidate)`` body from HumanEval).
        entry_point:
            Function name to pass to the test harness.

        Returns
        -------
        ExecutionResult
            Structured execution result.
        """
        self.logger.debug(f"Evaluating code ({len(code)} chars) with tests")
        return await self.executor.execute(
            code=code,
            test_code=test_code,
            entry_point=entry_point,
        )

    async def evaluate_humaneval(
        self,
        code: str,
        problem: dict[str, Any],
    ) -> ExecutionResult:
        """Evaluate code against a HumanEval problem.

        Parameters
        ----------
        code:
            Generated function code.
        problem:
            HumanEval problem dict with ``test``, ``entry_point`` keys.

        Returns
        -------
        ExecutionResult
        """
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")

        # HumanEval tests use `check(candidate)` pattern
        # We need to construct the full test script
        full_test = self._build_humaneval_test(code, test_code, entry_point)
        return await self.executor.execute(code=full_test, test_code="")

    async def evaluate_batch(
        self,
        codes: list[str],
        test_codes: list[str],
        entry_points: list[str] | None = None,
        max_concurrent: int = 8,
    ) -> list[ExecutionResult]:
        """Evaluate a batch of code samples concurrently.

        Parameters
        ----------
        codes:
            List of generated code samples.
        test_codes:
            List of corresponding test code.
        entry_points:
            Optional list of entry point function names.
        max_concurrent:
            Maximum concurrent evaluations.

        Returns
        -------
        list[ExecutionResult]
        """
        if entry_points is None:
            entry_points = [""] * len(codes)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _eval_one(
            code: str, test: str, ep: str
        ) -> ExecutionResult:
            async with semaphore:
                return await self.evaluate(code, test, ep)

        tasks = [
            _eval_one(c, t, e)
            for c, t, e in zip(codes, test_codes, entry_points)
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _build_humaneval_test(
        code: str, test_code: str, entry_point: str
    ) -> str:
        """Build a self-contained HumanEval test script."""
        return f"""\
import sys

# Generated Code
{code}

# HumanEval Test
{test_code}

passed = 0
failed = 0
total = 0

try:
    check({entry_point})
    passed += 1
except AssertionError as e:
    failed += 1
    print(f"ASSERTION FAIL: {{e}}", file=sys.stderr)
except Exception as e:
    failed += 1
    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
total += 1

print(f"RESULTS: passed={{passed}} failed={{failed}} total={{total}}")
"""


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text by the given number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))
