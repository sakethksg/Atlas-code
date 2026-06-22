"""Distillation — dataset building and LoRA training.

Converts repair trajectories into chat-format training datasets and
runs QLoRA fine-tuning on the student SLM. Key insight: we distill
problem + failure + repair reasoning + corrected solution (not just
problem → solution), teaching the student debugging skills.
"""

from afdad.distillation.dataset_builder import DatasetBuilder
from afdad.distillation.lora_train import LoRATrainer
from afdad.distillation.trainer import TrainingOrchestrator

__all__ = ["DatasetBuilder", "LoRATrainer", "TrainingOrchestrator"]
