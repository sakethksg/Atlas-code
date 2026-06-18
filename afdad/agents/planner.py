"""
Planner Agent — understands problems and generates algorithmic plans.

Uses the student model to decompose problems, identify edge cases,
and propose algorithm strategies before code generation begins.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from afdad.agents.base import BaseAgent
from afdad.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from afdad.utils.models import PlanOutput


class PlannerAgent(BaseAgent):
    """Planner agent: task → algorithmic plan."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__(model_cfg)

    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def build_user_prompt(self, **kwargs: Any) -> str:
        task: str = kwargs["task"]
        return PLANNER_USER_TEMPLATE.format(task=task)

    def parse_response(self, raw: str) -> PlanOutput:
        """Parse the planner's JSON response into a PlanOutput model."""
        data = self.try_parse_json(raw)

        if "raw_response" in data:
            # Fallback: treat the whole response as the strategy
            return PlanOutput(
                algorithm_strategy=raw,
                complexity_estimate="Unknown",
                edge_cases=[],
                key_observations="",
            )

        return PlanOutput(
            algorithm_strategy=data.get("algorithm_strategy", ""),
            complexity_estimate=data.get("complexity_estimate", "Unknown"),
            edge_cases=data.get("edge_cases", []),
            key_observations=data.get("key_observations", ""),
        )
