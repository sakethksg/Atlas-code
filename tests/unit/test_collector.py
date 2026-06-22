"""Unit tests for the TrajectoryCollector."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from afdad.trajectories.collector import TrajectoryCollector
from afdad.utils.models import FailureCluster, RepairTrajectory


@pytest.fixture
def collector_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create({
        "save_dir": str(tmp_path / "trajectories"),
    })


def test_trajectory_collection_workflow(collector_cfg: DictConfig) -> None:
    collector = TrajectoryCollector(collector_cfg)
    
    # Create empty trajectory
    traj = collector.create_trajectory()
    assert isinstance(traj, RepairTrajectory)
    assert len(traj.steps) == 0
    
    # Add steps
    traj = collector.add_step(
        traj,
        agent="Debugger",
        action="generate_repair",
        reasoning="fix syntax",
        output="def test(): return 1",
    )
    assert len(traj.steps) == 1
    assert traj.num_attempts == 1
    
    # Finalise
    traj = collector.finalise(
        traj,
        repaired_code="def test(): return 1",
        repair_reasoning="Syntactical fix completed.",
        verified=True,
    )
    assert traj.repaired_code == "def test(): return 1"
    assert traj.verified is True
    
    # Save
    filepath = collector.save_trajectory(traj)
    assert filepath.exists()
    
    # Load
    loaded = collector.load_trajectory(traj.trajectory_id)
    assert loaded.trajectory_id == traj.trajectory_id
    assert len(loaded.steps) == 1
    assert loaded.verified is True
    
    # List
    traj_ids = collector.list_trajectories()
    assert traj.trajectory_id in traj_ids
    
    # Convert to TrainingExample
    example = collector.to_training_example(
        traj,
        task="Test task",
        plan="Plan",
        failed_code="def test(): pass",
        failure_cluster="Syntax",
    )
    assert example.problem == "Test task"
    assert example.plan == "Plan"
    assert example.failed_code == "def test(): pass"
    assert example.repaired_code == "def test(): return 1"
    assert example.failure_cluster == FailureCluster.SYNTAX
