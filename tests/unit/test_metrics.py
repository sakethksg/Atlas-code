"""Unit tests for the Pass@k metrics module."""

from __future__ import annotations

import pytest
from afdad.evaluation.metrics import estimate_pass_at_k, compute_pass_at_k_batch


def test_estimate_pass_at_k_exact() -> None:
    # If correct count c == n, pass rate is 1.0
    assert estimate_pass_at_k(num_samples=10, num_correct=10, k=1) == 1.0
    assert estimate_pass_at_k(num_samples=5, num_correct=5, k=3) == 1.0

    # If correct count c == 0, pass rate is 0.0
    assert estimate_pass_at_k(num_samples=10, num_correct=0, k=1) == 0.0
    assert estimate_pass_at_k(num_samples=5, num_correct=0, k=3) == 0.0

    # If samples n - correct c < k, pass rate is 1.0 (since any subset of size k contains at least one correct)
    assert estimate_pass_at_k(num_samples=5, num_correct=4, k=2) == 1.0
    assert estimate_pass_at_k(num_samples=5, num_correct=3, k=3) == 1.0


def test_estimate_pass_at_k_math() -> None:
    # Analytical case: n=2, c=1, k=1 -> pass@1 = 1 - (1/2) = 0.5
    assert estimate_pass_at_k(num_samples=2, num_correct=1, k=1) == pytest.approx(0.5)

    # Analytical case: n=3, c=1, k=2 -> pass@2 = 1 - ((2/3) * (1/2)) = 1 - 1/3 = 2/3
    assert estimate_pass_at_k(num_samples=3, num_correct=1, k=2) == pytest.approx(2/3)


def test_compute_pass_at_k_batch() -> None:
    results = {
        "task_1": 2,  # 2 correct out of 3
        "task_2": 0,  # 0 correct out of 3
        "task_3": 3,  # 3 correct out of 3
    }
    # n=3, k=1
    # task 1: c=2, pass@1 = 1 - (1/3) = 2/3
    # task 2: c=0, pass@1 = 0
    # task 3: c=3, pass@1 = 1.0
    # Avg = (2/3 + 0 + 1) / 3 = (5/3) / 3 = 5/9 = 0.5555...
    assert compute_pass_at_k_batch(results, num_samples=3, k=1) == pytest.approx(5/9)
