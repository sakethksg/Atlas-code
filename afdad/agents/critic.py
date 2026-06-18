"""
Critic Agent — validates repairs and detects hallucinated changes.

Part of the agentic repair team. Ensures repaired code actually
addresses the root cause and would pass the test cases.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from afdad.agents.base import BaseAgent
from afdad.prompts.critic import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE


class CriticAgent(BaseAgent):
    """Critic agent: repair → validation verdict."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__(model_cfg)

    @property
    def system_prompt(self) -> str:
        return CRITIC_SYSTEM_PROMPT

    def build_user_prompt(self, **kwargs: Any) -> str:
        return CRITIC_USER_TEMPLATE.format(
            task=kwargs["task"],
            failed_code=kwargs["failed_code"],
            error_type=kwargs.get("error_type", "Unknown"),
            stderr=kwargs.get("stderr", ""),
            repaired_code=kwargs["repaired_code"],
            repair_reasoning=kwargs.get("repair_reasoning", ""),
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the critic's JSON verdict."""
        data = self.try_parse_json(raw)

        if "raw_response" in data:
            # Conservative fallback: mark as not validated
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues_found": ["Could not parse critic response"],
                "reasoning": raw[:500],
                "suggested_improvements": "",
            }

        return {
            "is_valid": data.get("is_valid", False),
            "confidence": float(data.get("confidence", 0.0)),
            "issues_found": data.get("issues_found", []),
            "reasoning": data.get("reasoning", ""),
            "suggested_improvements": data.get("suggested_improvements", ""),
        }
