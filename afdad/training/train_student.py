"""
End-to-end training loop — the main training pipeline.

Runs problems through the LangGraph → collects trajectories →
builds dataset → trains LoRA → evaluates → repeats.
"""

from __future__ import annotations

import asyncio
from typing import Any

from omegaconf import DictConfig

from afdad.distillation.trainer import TrainingOrchestrator
from afdad.evaluation.humaneval import HumanEvalRunner
from afdad.evaluation.mbpp import MBPPRunner
from afdad.failures.clustering import FailureClustering
from afdad.graph.workflow import build_graph
from afdad.trajectories.replay_buffer import ReplayBuffer
from afdad.utils.logging import get_logger, log_metrics
from afdad.utils.models import TrainingExample


class StudentTrainer:
    """End-to-end AFDAD training loop.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger()

        # ── Experiment flags ──
        exp = cfg.get("experiment", {})
        if hasattr(exp, "experiment"):
            exp = exp.experiment
        self._use_adaptive_curriculum: bool = exp.get("use_adaptive_curriculum", True)

        # Components
        self.replay_buffer = ReplayBuffer(cfg.trajectories, seed=cfg.seed)
        self.clustering = FailureClustering(cfg.clustering)
        self.orchestrator = TrainingOrchestrator(
            model_cfg=cfg.model.student,
            training_cfg=cfg.training if hasattr(cfg, "training") else cfg,
            distillation_cfg=cfg.distillation,
            replay_buffer=self.replay_buffer,
        )

        # Pipeline
        self.graph = build_graph(cfg)

        # Evaluation
        self.humaneval_runner = HumanEvalRunner(cfg)
        self.mbpp_runner = MBPPRunner(cfg)

        # Reporting & Tracking
        from afdad.evaluation.reporter import ResultReporter
        from afdad.utils.experiment import ExperimentTracker
        self.reporter = ResultReporter(cfg.output_dir)
        self.tracker = ExperimentTracker(cfg)

    async def run(self) -> dict[str, Any]:
        """Execute the full training loop.

        Returns
        -------
        dict[str, Any]
            Final metrics across all rounds.
        """
        num_rounds = self.cfg.pipeline.num_training_rounds
        max_tasks = self.cfg.pipeline.max_tasks_per_round

        all_metrics: list[dict[str, Any]] = []

        # ── Initial evaluation ──
        self.logger.info("[bold]═══ Initial Evaluation ═══[/bold]")
        baseline_metrics = await self._evaluate(round_id=0)
        all_metrics.append({"round": 0, **baseline_metrics})

        run_name = self.cfg.get("experiment", {}).get("experiment", {}).get("name", "run")
        self.reporter.generate_report(run_name, baseline_metrics, round_id=0)
        self.tracker.log_round(round_id=0, metrics=baseline_metrics)

        # ── Training rounds ──
        for round_id in range(1, num_rounds + 1):
            self.logger.info(
                f"\n[bold]═══ Training Round {round_id}/{num_rounds} ═══[/bold]"
            )

            # Step 1: Run problems through pipeline
            await self._run_pipeline_round(
                round_id=round_id,
                max_tasks=max_tasks,
            )

            # Step 2: Train student
            if len(self.replay_buffer) > 0:
                failure_rates = (
                    self.clustering.get_failure_rates()
                    if self._use_adaptive_curriculum
                    else None
                )
                if failure_rates is not None:
                    self.logger.info(f"Failure rates: {failure_rates}")
                else:
                    self.logger.info("Adaptive curriculum disabled — uniform sampling")

                adapter_path = self.orchestrator.run_training_round(
                    failure_rates=failure_rates,
                    round_id=round_id,
                )
                self.logger.info(f"Adapter saved: {adapter_path}")

            # Step 3: Evaluate
            round_metrics = await self._evaluate(round_id=round_id)
            all_metrics.append({"round": round_id, **round_metrics})

            # Save reporting metadata
            dataset_stats = self.replay_buffer.get_cluster_distribution()
            self.reporter.generate_report(run_name, round_metrics, round_id=round_id)
            self.tracker.log_round(round_id=round_id, metrics=round_metrics, dataset_stats=dataset_stats)

            # Log improvement
            if len(all_metrics) >= 2:
                prev = all_metrics[-2]
                curr = all_metrics[-1]
                for key in ["humaneval/pass@1", "mbpp/pass@1"]:
                    if key in prev and key in curr:
                        delta = curr[key] - prev[key]
                        self.logger.info(
                            f"  {key}: {prev[key]:.3f} → {curr[key]:.3f} "
                            f"({'↑' if delta > 0 else '↓'}{abs(delta):.3f})"
                        )

        # Save final replay buffer
        self.replay_buffer.save()

        # Save experiment summary
        self.tracker.save_summary()

        return {"rounds": all_metrics}

    async def _run_pipeline_round(
        self,
        round_id: int,
        max_tasks: int,
    ) -> None:
        """Run coding problems through the AFDAD graph pipeline."""
        problems = self._load_problems(max_tasks)

        self.logger.info(f"Processing {len(problems)} problems through pipeline")

        for i, problem in enumerate(problems):
            self.logger.info(
                f"[dim]Task {i + 1}/{len(problems)}[/dim]"
            )

            initial_state = {
                "task": problem["prompt"],
                "task_id": problem.get("task_id", f"task_{i}"),
                "test_cases": problem.get("test_cases", []),
                "entry_point": problem.get("entry_point", ""),
                "attempt": 0,
                "max_attempts": self.cfg.pipeline.max_repair_attempts,
                "success": False,
                "expert_called": False,
            }

            try:
                result = await self.graph.ainvoke(initial_state)

                # Collect training example if generated
                if result.get("training_example"):
                    example = TrainingExample(**result["training_example"])
                    self.replay_buffer.add(example)

            except Exception as exc:
                self.logger.error(f"Pipeline error on task {i}: {exc}")

        self.logger.info(
            f"Round {round_id}: Buffer size = {len(self.replay_buffer)}"
        )

    async def _evaluate(self, round_id: int) -> dict[str, float]:
        """Run evaluation benchmarks."""
        metrics: dict[str, float] = {}

        benchmarks = list(self.cfg.evaluation.benchmarks)

        if "humaneval" in benchmarks:
            he_metrics = await self.humaneval_runner.run()
            metrics.update(he_metrics)

        if "mbpp" in benchmarks:
            mbpp_metrics = await self.mbpp_runner.run()
            metrics.update(mbpp_metrics)

        log_metrics(metrics, step=round_id)
        return metrics

    def _load_problems(self, max_tasks: int) -> list[dict[str, Any]]:
        """Load coding problems for the pipeline.

        Uses the shared loader from :mod:`afdad.datasets.loaders`.

        Args:
            max_tasks: Maximum number of problems to return.

        Returns:
            List of problem dicts for pipeline processing.
        """
        from afdad.datasets.loaders import load_humaneval_as_list

        return load_humaneval_as_list(max_tasks=max_tasks)

