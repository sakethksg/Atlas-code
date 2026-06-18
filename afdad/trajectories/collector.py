"""
Trajectory collector — captures full repair trajectories for distillation.

Records the complete path: problem → plan → failed_code → failure analysis
→ repair reasoning → repaired_code.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.utils.logging import get_logger
from afdad.utils.models import RepairStep, RepairTrajectory, TrainingExample


class TrajectoryCollector:
    """Captures and serialises repair trajectories.

    Parameters
    ----------
    cfg:
        Trajectories configuration with ``save_dir`` field.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def create_trajectory(self) -> RepairTrajectory:
        """Create a new empty trajectory with a unique ID."""
        return RepairTrajectory(
            trajectory_id=str(uuid.uuid4()),
            steps=[],
        )

    def add_step(
        self,
        trajectory: RepairTrajectory,
        agent: str,
        action: str,
        reasoning: str,
        output: str,
    ) -> RepairTrajectory:
        """Add a step to an existing trajectory.

        Parameters
        ----------
        trajectory:
            The trajectory to extend.
        agent:
            Name of the agent (e.g., "Planner", "Debugger", "Critic").
        action:
            Action taken (e.g., "analyse_failure", "generate_repair").
        reasoning:
            Reasoning behind the action.
        output:
            Output produced by the step.

        Returns
        -------
        RepairTrajectory
            Updated trajectory.
        """
        step = RepairStep(
            agent=agent,
            action=action,
            reasoning=reasoning,
            output=output,
        )
        trajectory.steps.append(step)
        trajectory.num_attempts = len(
            [s for s in trajectory.steps if s.action == "generate_repair"]
        )
        return trajectory

    def finalise(
        self,
        trajectory: RepairTrajectory,
        repaired_code: str,
        repair_reasoning: str,
        verified: bool = False,
    ) -> RepairTrajectory:
        """Finalise a trajectory with the repaired code.

        Parameters
        ----------
        trajectory:
            The trajectory to finalise.
        repaired_code:
            Final corrected code.
        repair_reasoning:
            Summary of repair reasoning.
        verified:
            Whether the critic verified the repair.

        Returns
        -------
        RepairTrajectory
            Finalised trajectory.
        """
        trajectory.repaired_code = repaired_code
        trajectory.repair_reasoning = repair_reasoning
        trajectory.verified = verified
        return trajectory

    def to_training_example(
        self,
        trajectory: RepairTrajectory,
        task: str,
        plan: str,
        failed_code: str,
        failure_cluster: str = "Unknown",
    ) -> TrainingExample:
        """Convert a trajectory into a training example for distillation.

        Parameters
        ----------
        trajectory:
            The completed repair trajectory.
        task:
            Original coding problem.
        plan:
            Algorithmic plan.
        failed_code:
            Student's failed code.
        failure_cluster:
            Assigned failure cluster.

        Returns
        -------
        TrainingExample
        """
        from afdad.utils.models import FailureCluster

        try:
            cluster = FailureCluster(failure_cluster)
        except ValueError:
            cluster = FailureCluster.UNKNOWN

        return TrainingExample(
            problem=task,
            plan=plan,
            failed_code=failed_code,
            repair_reasoning=trajectory.repair_reasoning,
            repaired_code=trajectory.repaired_code,
            failure_cluster=cluster,
        )

    def save_trajectory(self, trajectory: RepairTrajectory) -> Path:
        """Save a trajectory to disk as JSON.

        Parameters
        ----------
        trajectory:
            The trajectory to save.

        Returns
        -------
        Path
            Path to the saved file.
        """
        filepath = self.save_dir / f"{trajectory.trajectory_id}.json"
        filepath.write_text(
            trajectory.model_dump_json(indent=2), encoding="utf-8"
        )
        self.logger.debug(f"Saved trajectory to {filepath}")
        return filepath

    def load_trajectory(self, trajectory_id: str) -> RepairTrajectory:
        """Load a trajectory from disk.

        Parameters
        ----------
        trajectory_id:
            UUID of the trajectory.

        Returns
        -------
        RepairTrajectory
        """
        filepath = self.save_dir / f"{trajectory_id}.json"
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return RepairTrajectory(**data)

    def list_trajectories(self) -> list[str]:
        """List all saved trajectory IDs."""
        return [
            p.stem for p in self.save_dir.glob("*.json")
        ]
