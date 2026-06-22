"""Unit tests for the LangGraph routing logic."""

from __future__ import annotations

from afdad.graph.routing import (
    route_after_execution,
    route_after_re_execution,
    route_after_repair,
)


def test_route_after_execution() -> None:
    # Success routes to success_store
    assert route_after_execution({"success": True}) == "success_store"

    # Failure routes to failure_analysis
    assert route_after_execution({"success": False}) == "failure_analysis"


def test_route_after_repair() -> None:
    # Verified routes to re_execute
    assert route_after_repair({"repair_verified": True}) == "re_execute"

    # Unverified with attempts remaining routes to expert_repair
    state = {"repair_verified": False, "attempt": 1, "max_attempts": 3}
    assert route_after_repair(state) == "expert_repair"

    # Unverified with no attempts remaining routes to distillation
    state = {"repair_verified": False, "attempt": 3, "max_attempts": 3}
    assert route_after_repair(state) == "distillation"


def test_route_after_re_execution() -> None:
    # Success routes to distillation
    assert route_after_re_execution({"success": True}) == "distillation"

    # Failure with attempts remaining routes to failure_analysis
    state = {"success": False, "attempt": 1, "max_attempts": 3}
    assert route_after_re_execution(state) == "failure_analysis"

    # Failure with no attempts remaining routes to distillation
    state = {"success": False, "attempt": 3, "max_attempts": 3}
    assert route_after_re_execution(state) == "distillation"
