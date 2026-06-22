"""
Base benchmark runner — abstract class for evaluation runners.

Provides a shared execution loop and metrics computation for benchmark runners,
reducing duplicate code while allowing customized problem loading and evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

from afdad.agents.coder import CoderAgent
from afdad.agents.planner import PlannerAgent
from afdad.evaluation.metrics import compute_pass_at_k_batch
from afdad.execution.evaluator import Evaluator
from afdad.utils.logging import get_logger, log_metrics


class BaseBenchmarkRunner(ABC):
    """Abstract base runner for benchmarks.

    Manages the common workflow of loading tasks, executing them sequentially,
    tracking pass rates across multiple samples, and reporting metrics.
    """

    def __init__(self, cfg: DictConfig, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self.logger = get_logger()
        self.planner = PlannerAgent(cfg.model.student)
        self.coder = CoderAgent(cfg.model.student)
        self.evaluator = Evaluator(cfg.execution)
        self.num_samples: int = cfg.evaluation.num_samples_per_task
        self.temperature: float = cfg.evaluation.temperature

    @abstractmethod
    def load_problems(self) -> dict[str, Any]:
        """Load benchmark problems."""
        pass

    @abstractmethod
    async def _evaluate_problem(self, problem: dict[str, Any]) -> int:
        """Evaluate a single problem, returning the number of correct samples."""
        pass

    async def run(self) -> dict[str, float]:
        """Run the full benchmark evaluation."""
        problems = self.load_problems()
        self.logger.info(
            f"Running {self.name}: {len(problems)} problems, "
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
        pass_at_1 = compute_pass_at_k_batch(results, self.num_samples, k=1)
        pass_at_5 = compute_pass_at_k_batch(results, self.num_samples, k=5)

        metrics = {
            f"{self.name.lower()}/pass@1": pass_at_1,
            f"{self.name.lower()}/pass@5": pass_at_5,
            f"{self.name.lower()}/num_problems": len(problems),
        }

        log_metrics(metrics)
        self.logger.info(
            f"[bold]{self.name} Results:[/bold] "
            f"Pass@1={pass_at_1:.3f}, Pass@5={pass_at_5:.3f}"
        )

        return metrics
