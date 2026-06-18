"""
MBPP benchmark runner — evaluates the student model on MBPP.

Computes Pass@1 and Pass@k metrics, similar to HumanEval runner.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from omegaconf import DictConfig

from afdad.agents.coder import CoderAgent
from afdad.agents.planner import PlannerAgent
from afdad.execution.evaluator import Evaluator
from afdad.utils.logging import get_logger, log_metrics


def _estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """Compute Pass@k using the unbiased estimator."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod(
        (num_samples - num_correct - i) / (num_samples - i)
        for i in range(k)
    )


class MBPPRunner:
    """Runs MBPP benchmark evaluation.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.planner = PlannerAgent(cfg.model.student)
        self.coder = CoderAgent(cfg.model.student)
        self.evaluator = Evaluator(cfg.sandbox)
        self.num_samples: int = cfg.evaluation.num_samples_per_task
        self.temperature: float = cfg.evaluation.temperature

    async def run(self) -> dict[str, float]:
        """Run the full MBPP benchmark.

        Returns
        -------
        dict[str, float]
            Dictionary with Pass@1, Pass@5, and aggregate metrics.
        """
        problems = self._load_mbpp()
        self.logger.info(
            f"Running MBPP: {len(problems)} problems, "
            f"{self.num_samples} samples each"
        )

        results: dict[str, int] = {}  # task_id → num_correct

        for task_id, problem in problems.items():
            correct = await self._evaluate_problem(problem)
            results[task_id] = correct
            self.logger.info(
                f"  {task_id}: {correct}/{self.num_samples} correct"
            )

        # Compute Pass@k
        pass_at_1 = self._compute_pass_at_k(results, k=1)
        pass_at_5 = self._compute_pass_at_k(results, k=5)

        metrics = {
            "mbpp/pass@1": pass_at_1,
            "mbpp/pass@5": pass_at_5,
            "mbpp/num_problems": len(problems),
        }

        log_metrics(metrics)
        self.logger.info(
            f"[bold]MBPP Results:[/bold] "
            f"Pass@1={pass_at_1:.3f}, Pass@5={pass_at_5:.3f}"
        )

        return metrics

    async def _evaluate_problem(
        self, problem: dict[str, Any]
    ) -> int:
        """Generate multiple samples and count correct ones."""
        task = problem["text"]
        test_list = problem.get("test_list", [])

        # Build test code from assertions
        test_code = "\n".join(test_list) if test_list else ""

        # Generate plan once
        plan_output = await self.planner.generate(task=task)
        plan_str = plan_output.algorithm_strategy

        correct = 0
        for _ in range(self.num_samples):
            try:
                code = await self.coder.generate(task=task, plan=plan_str)

                # Build full test script
                full_script = f"{code}\n\n{test_code}"
                result = await self.evaluator.evaluate(
                    code=full_script, test_code=""
                )
                if result.is_success:
                    correct += 1
            except Exception as exc:
                self.logger.warning(f"Sample failed: {exc}")

        return correct

    def _compute_pass_at_k(
        self, results: dict[str, int], k: int
    ) -> float:
        """Compute average Pass@k across all problems."""
        scores = [
            _estimate_pass_at_k(self.num_samples, num_correct, k)
            for num_correct in results.values()
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _load_mbpp() -> dict[str, dict[str, Any]]:
        """Load MBPP problems from HuggingFace datasets.

        Returns
        -------
        dict[str, dict]
            Mapping from task_id to problem dict.
        """
        from datasets import load_dataset

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
