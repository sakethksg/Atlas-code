"""
Priority replay buffer — weighted sampling for adaptive curriculum.

Oversamples training examples from higher-failure clusters,
so the student focuses on its weakest areas.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from afdad.utils.logging import get_logger
from afdad.utils.models import TrainingExample


class ReplayBuffer:
    """Weighted replay buffer for adaptive curriculum training.

    Parameters
    ----------
    cfg:
        Trajectories configuration with ``buffer_capacity`` and ``save_dir``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.capacity: int = cfg.buffer_capacity
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        self._buffer: list[TrainingExample] = []
        self._load_buffer()

    # ── Public API ────────────────────────────────────────────

    def add(self, example: TrainingExample) -> None:
        """Add a training example to the buffer.

        If the buffer exceeds capacity, the oldest example is removed.
        """
        self._buffer.append(example)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def add_batch(self, examples: list[TrainingExample]) -> None:
        """Add multiple examples to the buffer."""
        for ex in examples:
            self.add(ex)

    def sample(
        self,
        n: int,
        failure_rates: dict[str, float] | None = None,
    ) -> list[TrainingExample]:
        """Sample examples with optional adaptive curriculum weighting.

        Parameters
        ----------
        n:
            Number of examples to sample.
        failure_rates:
            Mapping from cluster name → failure rate. If provided,
            examples from higher-failure clusters are oversampled.

        Returns
        -------
        list[TrainingExample]
        """
        if not self._buffer:
            return []

        n = min(n, len(self._buffer))

        if failure_rates is None:
            return random.sample(self._buffer, n)

        # Compute weights based on failure rates
        weights = self._compute_weights(failure_rates)
        indices = random.choices(
            range(len(self._buffer)),
            weights=weights,
            k=n,
        )
        return [self._buffer[i] for i in indices]

    def get_all(self) -> list[TrainingExample]:
        """Return all examples in the buffer."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Adaptive Curriculum ───────────────────────────────────

    def _compute_weights(
        self, failure_rates: dict[str, float]
    ) -> list[float]:
        """Compute sampling weights from failure rates.

        Higher failure rate → higher sampling weight.
        """
        weights: list[float] = []
        for ex in self._buffer:
            cluster_name = ex.failure_cluster.value
            rate = failure_rates.get(cluster_name, 0.0)
            # Weight = 1 + rate so that all examples have non-zero weight
            # but higher-failure clusters are oversampled
            weight = 1.0 + rate * 5.0  # 5x amplification factor
            weights.append(weight)

        # Normalise
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Save the buffer to disk."""
        filepath = self.save_dir / "replay_buffer.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in self._buffer:
                f.write(ex.model_dump_json() + "\n")
        self.logger.info(
            f"Saved {len(self._buffer)} examples to {filepath}"
        )

    def _load_buffer(self) -> None:
        """Load the buffer from disk if it exists."""
        filepath = self.save_dir / "replay_buffer.jsonl"
        if not filepath.exists():
            return

        self._buffer = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._buffer.append(TrainingExample(**data))

        self.logger.info(
            f"Loaded {len(self._buffer)} examples from replay buffer"
        )

    def get_cluster_distribution(self) -> dict[str, int]:
        """Return the distribution of examples per cluster."""
        dist: dict[str, int] = {}
        for ex in self._buffer:
            name = ex.failure_cluster.value
            dist[name] = dist.get(name, 0) + 1
        return dist
