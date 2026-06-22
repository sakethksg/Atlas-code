"""
Benchmark data loaders — centralised loading for HumanEval and MBPP.

Provides a single source of truth for loading benchmark datasets.
Previously, loading logic was duplicated across ``humaneval.py``,
``mbpp.py``, and ``train_student.py``. This module eliminates that
duplication.

Usage::

    from afdad.datasets.loaders import load_humaneval, load_mbpp

    problems = load_humaneval()        # dict[str, dict]
    mbpp_problems = load_mbpp()        # dict[str, dict]
    task_list = load_humaneval_as_list(max_tasks=50)  # list[dict]
"""

from __future__ import annotations

from typing import Any

from afdad.utils.logging import get_logger

logger = get_logger()


def load_humaneval() -> dict[str, dict[str, Any]]:
    """Load HumanEval problems.

    Attempts to load from the ``human-eval`` package first, falling back
    to the HuggingFace ``datasets`` library.

    Returns:
        Mapping from task_id to problem dict. Each problem contains:
        ``task_id``, ``prompt``, ``entry_point``, ``test``,
        ``canonical_solution``.
    """
    try:
        from human_eval.data import read_problems

        logger.debug("Loading HumanEval from human-eval package")
        return read_problems()
    except ImportError:
        logger.debug("Loading HumanEval from HuggingFace datasets")
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test")
        return {
            row["task_id"]: {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "test": row["test"],
                "canonical_solution": row["canonical_solution"],
            }
            for row in ds
        }


def load_mbpp() -> dict[str, dict[str, Any]]:
    """Load MBPP (sanitized) problems from HuggingFace datasets.

    Returns:
        Mapping from task_id (as string) to problem dict. Each problem
        contains: ``task_id``, ``text``, ``code``, ``test_list``,
        ``test_setup_code``.
    """
    from datasets import load_dataset

    logger.debug("Loading MBPP sanitized from HuggingFace datasets")
    ds = load_dataset("mbpp", "sanitized", split="test")
    return {
        str(row["task_id"]): {
            "task_id": str(row["task_id"]),
            "text": row["text"],
            "code": row["code"],
            "test_list": row["test_list"],
            "test_setup_code": row.get("test_setup_code", ""),
        }
        for row in ds
    }


def load_humaneval_as_list(max_tasks: int | None = None) -> list[dict[str, Any]]:
    """Load HumanEval problems as a list suitable for pipeline processing.

    Converts the dict format into a list with ``test_cases`` structured
    for the AFDAD pipeline's ``AFDADState``.

    Args:
        max_tasks: Maximum number of problems to return. ``None`` for all.

    Returns:
        List of problem dicts with ``task_id``, ``prompt``, ``entry_point``,
        and ``test_cases`` keys.
    """
    problems = load_humaneval()
    items = list(problems.items())
    if max_tasks is not None:
        items = items[:max_tasks]

    return [
        {
            "task_id": task_id,
            "prompt": p["prompt"],
            "entry_point": p.get("entry_point", ""),
            "test_cases": [{"test_code": p.get("test", "")}],
        }
        for task_id, p in items
    ]
