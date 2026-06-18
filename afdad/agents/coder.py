"""
Coder Agent — generates code from task + plan using the student SLM.

This is the student model's primary code-generation interface.
No expert involvement at this stage.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from afdad.agents.base import BaseAgent
from afdad.prompts.coder import CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE


class CoderAgent(BaseAgent):
    """Student coder agent: task + plan → code."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__(model_cfg)

    @property
    def system_prompt(self) -> str:
        return CODER_SYSTEM_PROMPT

    def build_user_prompt(self, **kwargs: Any) -> str:
        task: str = kwargs["task"]
        plan: str = kwargs.get("plan", "No plan provided.")
        return CODER_USER_TEMPLATE.format(task=task, plan=plan)

    def parse_response(self, raw: str) -> str:
        """Extract the code from the response."""
        return self.extract_code_block(raw)
