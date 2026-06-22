"""Unit tests for the ReplayBuffer class."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig

from afdad.trajectories.replay_buffer import ReplayBuffer
from afdad.utils.models import FailureCluster, TrainingExample


@pytest.fixture
def temp_save_dir(tmp_path: Path) -> Path:
    return tmp_path / "trajectories"


@pytest.fixture
def traj_cfg(temp_save_dir: Path) -> DictConfig:
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "buffer_capacity": 5,
        "save_dir": str(temp_save_dir),
        "curriculum_amplification": 5.0,
    })


def test_replay_buffer_capacity(traj_cfg: DictConfig) -> None:
    buffer = ReplayBuffer(traj_cfg, seed=42)
    assert len(buffer) == 0

    # Add examples exceeding capacity
    for i in range(7):
        ex = TrainingExample(
            problem=f"Problem {i}",
            plan="Plan",
            failed_code="code",
            repair_reasoning="reasoning",
            repaired_code="repaired",
            failure_cluster=FailureCluster.SYNTAX,
        )
        buffer.add(ex)

    assert len(buffer) == 5
    # The first two should have been popped
    examples = buffer.get_all()
    assert examples[0].problem == "Problem 2"
    assert examples[-1].problem == "Problem 6"


def test_replay_buffer_sampling_uniform(traj_cfg: DictConfig) -> None:
    buffer = ReplayBuffer(traj_cfg, seed=42)
    for i in range(5):
        ex = TrainingExample(
            problem=f"Problem {i}",
            plan="Plan",
            failed_code="code",
            repair_reasoning="reasoning",
            repaired_code="repaired",
            failure_cluster=FailureCluster.SYNTAX if i % 2 == 0 else FailureCluster.RUNTIME,
        )
        buffer.add(ex)

    sampled = buffer.sample(3)
    assert len(sampled) == 3
    # Check that sample counts are correct
    dist = buffer.get_cluster_distribution()
    assert dist["Syntax"] == 3
    assert dist["Runtime"] == 2


def test_replay_buffer_weighted_sampling(traj_cfg: DictConfig) -> None:
    buffer = ReplayBuffer(traj_cfg, seed=123)
    # Add 2 runtime failures and 2 syntax failures
    for i in range(2):
        buffer.add(TrainingExample(
            problem=f"Syntax {i}",
            plan="",
            failed_code="",
            repair_reasoning="",
            repaired_code="",
            failure_cluster=FailureCluster.SYNTAX,
        ))
        buffer.add(TrainingExample(
            problem=f"Runtime {i}",
            plan="",
            failed_code="",
            repair_reasoning="",
            repaired_code="",
            failure_cluster=FailureCluster.RUNTIME,
        ))

    # Syntax has failure rate 1.0, Runtime has 0.0
    # Weights for Syntax will be 1 + 1.0 * 5 = 6
    # Weights for Runtime will be 1 + 0.0 * 5 = 1
    failure_rates = {"Syntax": 1.0, "Runtime": 0.0}
    weights = buffer._compute_weights(failure_rates)
    # Expected relative weights: Syntax is 6/14 ~ 0.428 each, Runtime is 1/14 ~ 0.071 each
    assert weights[0] == pytest.approx(6 / 14)  # Syntax 0
    assert weights[1] == pytest.approx(1 / 14)  # Runtime 0


def test_replay_buffer_save_load(traj_cfg: DictConfig, temp_save_dir: Path) -> None:
    buffer = ReplayBuffer(traj_cfg, seed=42)
    ex = TrainingExample(
        problem="Test Save Load",
        plan="",
        failed_code="failed",
        repair_reasoning="reasoning",
        repaired_code="repaired",
        failure_cluster=FailureCluster.LOGIC,
    )
    buffer.add(ex)
    buffer.save()

    # Create new buffer loading from the same directory
    buffer2 = ReplayBuffer(traj_cfg, seed=42)
    assert len(buffer2) == 1
    assert buffer2.get_all()[0].problem == "Test Save Load"
