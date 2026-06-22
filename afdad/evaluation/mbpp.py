"""
MBPP benchmark runner — evaluates the student model on MBPP.

Computes Pass@1 and Pass@k metrics using the unbiased estimator
from Chen et al. (2021).

Research Significance:
    MBPP is the second primary benchmark alongside HumanEval.
    It tests a broader range of programming tasks and serves as
    a complementary evaluation of AFDAD's distillation effectiveness.
"""

from typing import Any

from omegaconf import DictConfig

from afdad.datasets.loaders import load_mbpp
from afdad.evaluation.benchmark_base import BaseBenchmarkRunner


class MBPPRunner(BaseBenchmarkRunner):
    """Runs MBPP benchmark evaluation.

    Generates multiple code samples per problem and computes Pass@k
    metrics. Uses the student model's Planner and Coder agents.

    Args:
        cfg: Full Hydra configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, name="MBPP")

    def load_problems(self) -> dict[str, Any]:
        return load_mbpp()

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

