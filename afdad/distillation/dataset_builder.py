"""
Dataset builder — converts replay buffer entries into HuggingFace Dataset
format for distillation training.

Key insight from the spec: we distill NOT `problem → solution`,
but `problem + failure + repair reasoning + corrected solution`.
This teaches the student debugging skills.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.utils.logging import get_logger
from afdad.utils.models import TrainingExample


# ── Chat Template ─────────────────────────────────────────────

DISTILLATION_SYSTEM = """\
You are a Python programmer who learns from mistakes. \
Given a coding problem, an algorithmic plan, and a previous failed attempt \
with its error analysis, write a corrected solution."""

DISTILLATION_USER_TEMPLATE = """\
## Problem

{problem}

## Plan

{plan}

## Previous Failed Attempt

```python
{failed_code}
```

## What Went Wrong

{repair_reasoning}

## Task

Write the corrected Python function."""

DISTILLATION_ASSISTANT_TEMPLATE = """\
```python
{repaired_code}
```"""


class DatasetBuilder:
    """Builds training datasets from distillation examples.

    Parameters
    ----------
    cfg:
        Distillation configuration with ``dataset_dir`` and ``max_seq_length``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.dataset_dir = Path(cfg.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length: int = cfg.max_seq_length
        self.logger = get_logger()

    def build_chat_dataset(
        self,
        examples: list[TrainingExample],
        output_name: str = "train",
    ) -> Path:
        """Convert training examples into a chat-format JSONL dataset.

        Each example becomes a multi-turn conversation:
        system → user (problem + failure context) → assistant (corrected code).

        Parameters
        ----------
        examples:
            List of training examples from the replay buffer.
        output_name:
            Name for the output file (without extension).

        Returns
        -------
        Path
            Path to the saved JSONL file.
        """
        import json

        output_path = self.dataset_dir / f"{output_name}.jsonl"

        records: list[dict[str, Any]] = []
        for ex in examples:
            messages = self._format_messages(ex)
            records.append({"messages": messages})

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.logger.info(
            f"Built dataset with {len(records)} examples → {output_path}"
        )
        return output_path

    def build_hf_dataset(
        self,
        examples: list[TrainingExample],
    ) -> Any:
        """Build a HuggingFace ``Dataset`` object for training.

        Parameters
        ----------
        examples:
            List of training examples.

        Returns
        -------
        datasets.Dataset
        """
        from datasets import Dataset

        records = []
        for ex in examples:
            messages = self._format_messages(ex)
            records.append({"messages": messages})

        dataset = Dataset.from_list(records)
        self.logger.info(
            f"Built HuggingFace Dataset with {len(dataset)} examples"
        )
        return dataset

    def _format_messages(
        self, example: TrainingExample
    ) -> list[dict[str, str]]:
        """Format a training example into chat messages."""
        user_content = DISTILLATION_USER_TEMPLATE.format(
            problem=example.problem,
            plan=example.plan or "No plan available.",
            failed_code=example.failed_code,
            repair_reasoning=example.repair_reasoning,
        )

        assistant_content = DISTILLATION_ASSISTANT_TEMPLATE.format(
            repaired_code=example.repaired_code,
        )

        return [
            {"role": "system", "content": DISTILLATION_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
