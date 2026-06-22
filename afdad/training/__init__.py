"""Training loop orchestration.

Manages the end-to-end AFDAD training loop: run problems through the
LangGraph pipeline → collect repair trajectories → build distillation
dataset → train QLoRA adapter → evaluate → repeat.
"""

from afdad.training.train_student import StudentTrainer

__all__ = ["StudentTrainer"]

