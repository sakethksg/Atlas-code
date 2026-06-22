"""Unit tests for the failure text builder and encoding."""

from __future__ import annotations

from afdad.failures.encoder import FailureEncoder


def test_build_failure_text_default_limits() -> None:
    task = "A long task prompt that needs to be truncated because it is very long and has lots of details."
    code = "def check(x):\n    return x + 1"
    error_type = "TypeError"
    stderr = "Error output stderr"
    traceback = "Traceback lines"

    res = FailureEncoder.build_failure_text(
        task=task,
        code=code,
        error_type=error_type,
        stderr=stderr,
        traceback=traceback,
        task_limit=10,
        traceback_limit=9,
        stderr_limit=5,
        code_limit=12,
    )

    assert "Task: A long tas" in res
    assert "Error Type: TypeError" in res
    assert "Traceback: Traceback" in res
    assert "Stderr: Error" in res
    assert "Code Snippet: def check(x)" in res

