"""
Base agent class with async OpenAI-compatible client.

All AFDAD agents (Planner, Coder, Debugger, Critic) inherit from this
class and override the system prompt and output parsing logic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI
from omegaconf import DictConfig
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from afdad.utils.logging import get_logger


class BaseAgent(ABC):
    """Abstract base for all AFDAD agents.

    Parameters
    ----------
    model_cfg:
        Model configuration (name, base_url, api_key, generation settings).
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        self.model_name: str = model_cfg.name
        self.client = AsyncOpenAI(
            base_url=model_cfg.base_url,
            api_key=model_cfg.api_key,
        )
        self.gen_cfg = model_cfg.generation
        self.logger = get_logger()

    # ── Abstract interface ────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""

    @abstractmethod
    def build_user_prompt(self, **kwargs: Any) -> str:
        """Build the user message from task-specific arguments."""

    def parse_response(self, raw: str) -> Any:
        """Parse the raw LLM response.  Override for structured output."""
        return raw

    # ── Generation ────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def generate(self, **kwargs: Any) -> Any:
        """Call the LLM and return parsed output.

        Keyword arguments are forwarded to :meth:`build_user_prompt`.
        """
        user_prompt = self.build_user_prompt(**kwargs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        stop = (
            list(self.gen_cfg.stop_sequences)
            if self.gen_cfg.get("stop_sequences")
            else None
        )

        self.logger.debug(
            f"[{self.__class__.__name__}] Calling {self.model_name}"
        )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.gen_cfg.temperature,
            max_tokens=self.gen_cfg.max_tokens,
            top_p=self.gen_cfg.get("top_p", 0.95),
            stop=stop,
        )

        raw_text = response.choices[0].message.content or ""
        self.logger.debug(
            f"[{self.__class__.__name__}] Response length: {len(raw_text)} chars"
        )

        return self.parse_response(raw_text)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def extract_code_block(text: str) -> str:
        """Extract the first fenced code block from text."""
        lines = text.split("\n")
        in_block = False
        code_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip().startswith("```") and in_block:
                break
            elif in_block:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else text

    @staticmethod
    def try_parse_json(text: str) -> dict[str, Any]:
        """Attempt to parse JSON from text, extracting from code blocks if needed."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        extracted = BaseAgent.extract_code_block(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # Return as raw string wrapped in dict
        return {"raw_response": text}
