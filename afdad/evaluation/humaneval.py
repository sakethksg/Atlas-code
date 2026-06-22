"""
HumanEval benchmark runner — evaluates the student model on HumanEval.

Computes Pass@1 and Pass@k metrics using the unbiased estimator
from Chen et al. (2021).

Research Significance:
    HumanEval is one of the two primary benchmarks for evaluating
    code generation capability. Improvement on HumanEval Pass@1
    across training rounds demonstrates the effectiveness of AFDAD's
    failure-driven distillation.
"""

from typing import Any

from omegaconf import DictConfig

from afdad.datasets.loaders import load_humaneval
from afdad.evaluation.benchmark_base import BaseBenchmarkRunner



class HumanEvalRunner(BaseBenchmarkRunner):
    """Runs HumanEval benchmark evaluation.

    Generates multiple code samples per problem and computes Pass@k
    metrics. Uses the student model's Planner and Coder agents.

    Args:
        cfg: Full Hydra configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, name="HumanEval")

    def load_problems(self) -> dict[str, Any]:
        return load_humaneval()

    async def _evaluate_problem(
        self, problem: dict[str, Any]
    ) -> int:
        """Generate multiple samples and count correct ones."""
        task = problem["prompt"]

        # Generate plan once
        plan_output = await self.planner.generate(task=task)
        plan_str = plan_output.algorithm_strategy

        correct = 0
        for _ in range(self.num_samples):
            try:
                code = await self.coder.generate(task=task, plan=plan_str)
                result = await self.evaluator.evaluate_humaneval(
                    code=code, problem=problem
                )
                if result.is_success:
                    correct += 1
            except Exception as exc:
                self.logger.warning(f"Sample failed: {exc}")

        return correct

