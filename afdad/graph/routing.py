"""
Conditional edge routing for the AFDAD LangGraph workflow.

Determines control flow after execution and repair nodes.
"""

from __future__ import annotations

from afdad.graph.state import AFDADState
from afdad.utils.logging import get_logger


def route_after_execution(state: AFDADState) -> str:
    """Route after the execution node.

    - If success → ``success_store`` (terminal node)
    - If failure → ``failure_analysis`` (begin repair pipeline)

    Parameters
    ----------
    state:
        Current graph state.

    Returns
    -------
    str
        Next node name.
    """
    logger = get_logger()

    if state.get("success", False):
        logger.info("Routing → [green]success_store[/green]")
        return "success_store"
    else:
        logger.info("Routing → [red]failure_analysis[/red]")
        return "failure_analysis"


def route_after_repair(state: AFDADState) -> str:
    """Route after the expert repair node.

    - If repair verified → ``re_execute`` (re-run tests on repaired code)
    - If not verified and attempts remaining → ``expert_repair`` (retry)
    - If max attempts reached → ``distillation`` (distill anyway)

    Parameters
    ----------
    state:
        Current graph state.

    Returns
    -------
    str
        Next node name.
    """
    logger = get_logger()

    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    verified = state.get("repair_verified", False)

    if verified:
        logger.info("Repair verified → [yellow]re_execute[/yellow]")
        return "re_execute"
    elif attempt < max_attempts:
        logger.info(
            f"Repair unverified (attempt {attempt}/{max_attempts}) "
            "→ [red]expert_repair[/red] (retry)"
        )
        return "expert_repair"
    else:
        logger.info(
            f"Max attempts reached ({max_attempts}) "
            "→ [white]distillation[/white]"
        )
        return "distillation"


def route_after_re_execution(state: AFDADState) -> str:
    """Route after re-executing repaired code.

    - If success → ``distillation`` (capture the successful repair)
    - If still failing and attempts remaining → ``failure_analysis`` (retry cycle)
    - If max attempts → ``distillation`` (distill what we have)

    Parameters
    ----------
    state:
        Current graph state.

    Returns
    -------
    str
        Next node name.
    """
    logger = get_logger()

    if state.get("success", False):
        logger.info("Repaired code passes → [white]distillation[/white]")
        return "distillation"

    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)

    if attempt < max_attempts:
        logger.info(
            f"Repaired code still fails (attempt {attempt}/{max_attempts}) "
            "→ [red]failure_analysis[/red]"
        )
        return "failure_analysis"
    else:
        logger.info(
            "Max attempts reached — distilling partial repair "
            "→ [white]distillation[/white]"
        )
        return "distillation"
