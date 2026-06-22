"""Utility helpers for the AFDAD framework.

Provides shared infrastructure: Pydantic data models, structured logging
with Rich console and W&B integration, and reproducibility utilities.
"""

from afdad.utils.logging import get_logger, log_event, log_metrics, setup_logging
from afdad.utils.models import (
    FailureCluster,
    TrainingExample,
    TaskResult,
    ExecutionResult,
    FailureRecord,
    RepairTrajectory,
    RepairStep,
)
from afdad.utils.reproducibility import (
    seed_everything,
    get_seeded_rng,
    save_experiment_snapshot,
)
from afdad.utils.experiment import ExperimentTracker

__all__ = [
    "get_logger",
    "log_event",
    "log_metrics",
    "setup_logging",
    "FailureCluster",
    "TrainingExample",
    "TaskResult",
    "ExecutionResult",
    "FailureRecord",
    "RepairTrajectory",
    "RepairStep",
    "seed_everything",
    "get_seeded_rng",
    "save_experiment_snapshot",
    "ExperimentTracker",
]


