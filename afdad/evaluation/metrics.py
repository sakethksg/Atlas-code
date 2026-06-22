"""
Evaluation metrics — shared Pass@k computation.

Implements the unbiased Pass@k estimator from:

    Chen et al., "Evaluating Large Language Models Trained on Code", 2021.
    https://arxiv.org/abs/2107.03374

Research Significance:
    Pass@k is the primary metric reported in the AFDAD paper. This module
    provides a single, tested implementation used by all benchmark runners,
    preventing divergence between HumanEval and MBPP metric computation.
"""

from __future__ import annotations

import math
from typing import Sequence


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """Compute Pass@k using the unbiased estimator.

    Uses the combinatorial formula from Chen et al. (2021):
    ``Pass@k = 1 - C(n-c, k) / C(n, k)``

    where n = num_samples, c = num_correct.

    Args:
        num_samples: Total number of code samples generated per problem.
        num_correct: Number of samples that passed all tests.
        k: The k in Pass@k (e.g., 1 for Pass@1).

    Returns:
        Estimated Pass@k probability in [0, 1].

    Raises:
        ValueError: If k > num_samples or inputs are negative.
    """
    if num_samples < 0 or num_correct < 0 or k < 0:
        raise ValueError(
            f"All inputs must be non-negative: "
            f"num_samples={num_samples}, num_correct={num_correct}, k={k}"
        )
    if k > num_samples:
        raise ValueError(
            f"k ({k}) cannot exceed num_samples ({num_samples})"
        )
    if num_correct > num_samples:
        raise ValueError(
            f"num_correct ({num_correct}) cannot exceed num_samples ({num_samples})"
        )

    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - math.prod(
        (num_samples - num_correct - i) / (num_samples - i)
        for i in range(k)
    )


def compute_pass_at_k_batch(
    results: dict[str, int],
    num_samples: int,
    k: int,
) -> float:
    """Compute average Pass@k across a batch of problems.

    Args:
        results: Mapping from task_id to number of correct samples.
        num_samples: Number of samples generated per problem.
        k: The k in Pass@k.

    Returns:
        Average Pass@k across all problems.
    """
    if not results:
        return 0.0

    scores = [
        estimate_pass_at_k(num_samples, num_correct, k)
        for num_correct in results.values()
    ]
    return sum(scores) / len(scores)
