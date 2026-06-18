"""
Debugger Agent — identifies root causes and proposes repairs using the expert model.

Part of the agentic repair team. Receives failed code + execution output,
performs root-cause analysis, and generates corrected code.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from afdad.agents.base import BaseAgent
from afdad.prompts.debugger import DEBUGGER_SYSTEM_PROMPT, DEBUGGER_USER_TEMPLATE
from afdad.utils.models import FailureAnalysis


class DebuggerAgent(BaseAgent):
    """Debugger agent: failed_code + execution_result → analysis + repaired_code."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__(model_cfg)

    @property
    def system_prompt(self) -> str:
        return DEBUGGER_SYSTEM_PROMPT

    def build_user_prompt(self, **kwargs: Any) -> str:
        return DEBUGGER_USER_TEMPLATE.format(
            task=kwargs["task"],
            plan=kwargs.get("plan", ""),
            failed_code=kwargs["failed_code"],
            returncode=kwargs.get("returncode", -1),
            passed=kwargs.get("passed", 0),
            total=kwargs.get("total", 0),
            error_type=kwargs.get("error_type", "Unknown"),
            stderr=kwargs.get("stderr", ""),
            stdout=kwargs.get("stdout", ""),
            traceback=kwargs.get("traceback", ""),
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the debugger's JSON response."""
        data = self.try_parse_json(raw)

        if "raw_response" in data:
            # Fallback: try to extract code from raw text
            code = self.extract_code_block(raw)
            return {
                "analysis": FailureAnalysis(
                    root_cause="Could not parse structured response",
                    error_category="Unknown",
                    detailed_explanation=raw[:500],
                    suggested_fix_strategy="See raw response",
                    relevant_code_section="",
                ),
                "repaired_code": code,
            }

        analysis = FailureAnalysis(
            root_cause=data.get("root_cause", ""),
            error_category=data.get("error_category", "Unknown"),
            detailed_explanation=data.get("detailed_explanation", ""),
            suggested_fix_strategy=data.get("suggested_fix_strategy", ""),
            relevant_code_section=data.get("relevant_code_section", ""),
        )

        repaired_code = data.get("repaired_code", "")
        # If repaired_code is in a code block, extract it
        if "```" in repaired_code:
            repaired_code = self.extract_code_block(repaired_code)

        return {
            "analysis": analysis,
            "repaired_code": repaired_code,
        }
