"""
AFDAD Main Entrypoint — Hydra-powered CLI for the full framework.

Usage:
    # Run full AFDAD training pipeline
    python -m afdad.main mode=train

    # Run a single task through the pipeline
    python -m afdad.main mode=single_task task="Write a function that returns fibonacci numbers"

    # Evaluate the student model
    python -m afdad.main mode=evaluate

    # Run a specific ablation study
    python -m afdad.main mode=train experiment.experiment.name=run_0_baseline \\
        experiment.experiment.use_agentic_repair=false \\
        experiment.experiment.use_distillation=false \\
        experiment.experiment.use_failure_clustering=false
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """AFDAD main entrypoint."""
    from afdad.utils.logging import setup_logging, init_wandb, get_logger

    # ── Setup ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("[bold]AFDAD — Adaptive Failure-Driven Agentic Distillation[/bold]")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Init W&B
    wandb_run = init_wandb(cfg)

    # ── Mode dispatch ──
    mode = cfg.get("mode", "train")
    logger.info(f"Mode: [bold]{mode}[/bold]")

    if mode == "single_task":
        asyncio.run(_run_single_task(cfg))
    elif mode == "train":
        asyncio.run(_run_training(cfg))
    elif mode == "evaluate":
        asyncio.run(_run_evaluation(cfg))
    elif mode == "visualise":
        _visualise_graph(cfg)
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)

    # Cleanup
    if wandb_run:
        wandb_run.finish()


async def _run_single_task(cfg: DictConfig) -> None:
    """Run a single task through the AFDAD pipeline."""
    from afdad.graph.workflow import build_graph
    from afdad.utils.logging import get_logger

    logger = get_logger()

    task = cfg.get("task", "Write a Python function that returns the sum of two numbers.")
    logger.info(f"Single task: {task[:100]}...")

    graph = build_graph(cfg)

    initial_state = {
        "task": task,
        "task_id": "single_task",
        "test_cases": [],
        "entry_point": "",
        "attempt": 0,
        "max_attempts": cfg.pipeline.max_repair_attempts,
        "success": False,
        "expert_called": False,
    }

    result = await graph.ainvoke(initial_state)

    logger.info(f"\n[bold]Result:[/bold]")
    logger.info(f"  Success: {result.get('success')}")
    logger.info(f"  Expert called: {result.get('expert_called')}")
    logger.info(f"  Failure cluster: {result.get('failure_cluster', 'N/A')}")

    if result.get("student_code"):
        logger.info(f"\n[bold]Student Code:[/bold]\n{result['student_code']}")
    if result.get("repaired_code"):
        logger.info(f"\n[bold]Repaired Code:[/bold]\n{result['repaired_code']}")


async def _run_training(cfg: DictConfig) -> None:
    """Run the full AFDAD training pipeline."""
    from afdad.training.train_student import StudentTrainer
    from afdad.utils.logging import get_logger

    logger = get_logger()
    trainer = StudentTrainer(cfg)
    results = await trainer.run()

    logger.info(f"\n[bold]Training Complete![/bold]")
    for round_data in results.get("rounds", []):
        r = round_data.get("round", "?")
        p1 = round_data.get("humaneval/pass@1", 0)
        logger.info(f"  Round {r}: Pass@1 = {p1:.3f}")


async def _run_evaluation(cfg: DictConfig) -> None:
    """Run evaluation benchmarks."""
    from afdad.evaluation.humaneval import HumanEvalRunner
    from afdad.evaluation.mbpp import MBPPRunner
    from afdad.utils.logging import get_logger

    logger = get_logger()
    benchmarks = list(cfg.evaluation.benchmarks)

    if "humaneval" in benchmarks:
        runner = HumanEvalRunner(cfg)
        await runner.run()

    if "mbpp" in benchmarks:
        runner = MBPPRunner(cfg)
        await runner.run()


def _visualise_graph(cfg: DictConfig) -> None:
    """Render the AFDAD graph."""
    from afdad.graph.workflow import visualise_graph

    output_path = cfg.get("output_path", "afdad_graph.png")
    visualise_graph(cfg, output_path)


if __name__ == "__main__":
    main()
