"""Trajectory collection and replay buffer.

Captures full repair trajectories (problem → plan → failure → repair)
and manages a priority replay buffer for adaptive curriculum training.
Higher-failure clusters are oversampled so the student focuses on weaknesses.
"""

from afdad.trajectories.collector import TrajectoryCollector
from afdad.trajectories.replay_buffer import ReplayBuffer

__all__ = ["TrajectoryCollector", "ReplayBuffer"]
