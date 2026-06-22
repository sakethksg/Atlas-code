"""Unit tests for Pydantic data models."""

from __future__ import annotations

from afdad.utils.models import ExecutionResult, FailureCluster, FailureRecord


def test_failure_cluster_from_string() -> None:
    # Exact and case-insensitive conversion
    assert FailureCluster.from_string("Syntax") == FailureCluster.SYNTAX
    assert FailureCluster.from_string("syntax ") == FailureCluster.SYNTAX
    assert FailureCluster.from_string("RUNTIME") == FailureCluster.RUNTIME
    assert FailureCluster.from_string("unknown_val") == FailureCluster.UNKNOWN


def test_execution_result_is_success() -> None:
    # Successful execution
    res = ExecutionResult(passed=3, failed=0, total=3, timeout=False)
    assert res.is_success is True

    # Failed assertion
    res = ExecutionResult(passed=2, failed=1, total=3, timeout=False)
    assert res.is_success is False

    # Timeout
    res = ExecutionResult(passed=3, failed=0, total=3, timeout=True)
    assert res.is_success is False

    # Empty tests
    res = ExecutionResult(passed=0, failed=0, total=0)
    assert res.is_success is False


def test_failure_record_serialization() -> None:
    record = FailureRecord(
        task="Write a test function",
        code="def test(): pass",
        traceback="RuntimeError",
        failure_cluster=FailureCluster.RUNTIME,
    )
    data = record.model_dump()
    assert data["failure_cluster"] == "Runtime"
    assert data["task"] == "Write a test function"
