"""
Training orchestrator — manages training rounds, checkpoint selection,
and adapter management for the student model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.distillation.dataset_builder import DatasetBuilder
from afdad.distillation.lora_train import LoRATrainer
from afdad.trajectories.replay_buffer import ReplayBuffer
from afdad.utils.logging import get_logger, log_metrics


class TrainingOrchestrator:
    """Manages the full distillation training lifecycle.

    Parameters
    ----------
    model_cfg:
        Student model configuration.
    training_cfg:
        Training/LoRA configuration.
    distillation_cfg:
        Distillation configuration (dataset_dir, max_seq_length).
    replay_buffer:
        The replay buffer containing training examples.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        training_cfg: DictConfig,
        distillation_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.replay_buffer = replay_buffer
        self.logger = get_logger()

        self.dataset_builder = DatasetBuilder(distillation_cfg)
        self.lora_trainer = LoRATrainer(model_cfg, training_cfg)

        self._training_round: int = 0
        self._adapter_paths: list[Path] = []

    def run_training_round(
        self,
        failure_rates: dict[str, float] | None = None,
        round_id: int | None = None,
    ) -> Path:
        """Execute a single training round.

        1. Sample from replay buffer (with adaptive curriculum).
        2. Build HuggingFace dataset.
        3. Train QLoRA adapter.
        4. Return adapter path.

        Parameters
        ----------
        failure_rates:
            Cluster failure rates for adaptive sampling.
        round_id:
            Optional round identifier (defaults to auto-increment).

        Returns
        -------
        Path
            Path to the trained adapter.
        """
        if round_id is not None:
            self._training_round = round_id
        else:
            self._training_round += 1

        self.logger.info(
            f"[bold]Training Round {self._training_round}[/bold] — "
            f"Buffer size: {len(self.replay_buffer)}"
        )

        # 1. Sample training examples
        examples = self.replay_buffer.sample(
            n=len(self.replay_buffer),
            failure_rates=failure_rates,
        )

        if not examples:
            self.logger.warning("No examples in replay buffer — skipping.")
            return Path()

        # Log cluster distribution
        dist = self.replay_buffer.get_cluster_distribution()
        self.logger.info(f"Cluster distribution: {dist}")
        log_metrics(
            {f"cluster_{k}": v for k, v in dist.items()},
            step=self._training_round,
        )

        # 2. Build dataset
        dataset = self.dataset_builder.build_hf_dataset(examples)

        # 3. Train
        output_dir = (
            Path(self.training_cfg.training.output_dir)
            / f"round_{self._training_round}"
        )
        adapter_path = self.lora_trainer.train(
            dataset=dataset,
            output_dir=str(output_dir),
        )

        self._adapter_paths.append(adapter_path)

        log_metrics(
            {
                "training_round": self._training_round,
                "num_examples": len(examples),
                "adapter_path": str(adapter_path),
            },
            step=self._training_round,
        )

        return adapter_path

    def get_latest_adapter(self) -> Path | None:
        """Return the most recently trained adapter path."""
        return self._adapter_paths[-1] if self._adapter_paths else None

    def merge_latest_adapter(self, output_path: str | Path) -> Path:
        """Merge the latest adapter into the base student model.

        Parameters
        ----------
        output_path:
            Where to save the merged model.

        Returns
        -------
        Path
            Path to the merged model.
        """
        adapter = self.get_latest_adapter()
        if adapter is None:
            raise ValueError("No adapter available to merge.")

        return self.lora_trainer.merge_adapter(
            base_model_name=self.model_cfg.name,
            adapter_path=adapter,
            output_path=output_path,
        )
