"""Unit tests for the subprocess CodeExecutor internals."""

from __future__ import annotations

from pathlib import Path
import pytest
from omegaconf import DictConfig, OmegaConf

from afdad.execution.executor import CodeExecutor


@pytest.fixture
def exec_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create({
        "timeout_seconds": 2,
        "temp_dir": str(tmp_path),
    })


def test_build_script(exec_cfg: DictConfig) -> None:
    executor = CodeExecutor(exec_cfg)
    
    # 1. With test code and entry point
    script = executor._build_script(
        code="def sum(a, b):\n    return a + b",
        test_code="assert sum(1, 2) == 3",
        entry_point="sum",
    )
    assert "def sum(a, b):" in script
    assert "assert sum(1, 2) == 3" in script
    assert "check(sum)" in script
    assert "RESULTS: passed=" in script

    # 2. Without test code
    script_no_test = executor._build_script(
        code="def sum(a, b):\n    return a + b",
        test_code="",
        entry_point="",
    )
    assert "def sum(a, b):" in script_no_test
    assert "No test code provided" in script_no_test
    assert "RESULTS: passed=1 failed=0 total=1" in script_no_test


def test_parse_test_output(exec_cfg: DictConfig) -> None:
    executor = CodeExecutor(exec_cfg)
    
    # 1. Success case
    res = executor._parse_test_output(
        stdout="RESULTS: passed=1 failed=0 total=1",
        stderr="",
        returncode=0,
        elapsed_ms=10.0,
    )
    assert res.is_success is True
    assert res.passed == 1
    assert res.failed == 0
    assert res.total == 1
    assert res.returncode == 0
    assert res.error_type is None

    # 2. Failed test case
    res_failed = executor._parse_test_output(
        stdout="RESULTS: passed=0 failed=1 total=1",
        stderr="FAIL: assertion failed\nAssertionError\n",
        returncode=0,
        elapsed_ms=10.0,
    )
    assert res_failed.is_success is False
    assert res_failed.passed == 0
    assert res_failed.failed == 1
    assert res_failed.total == 1
    assert res_failed.error_type == "AssertionError"

    # 3. Crash / non-zero returncode without test results
    res_crash = executor._parse_test_output(
        stdout="",
        stderr="SyntaxError: invalid syntax",
        returncode=1,
        elapsed_ms=10.0,
    )
    assert res_crash.is_success is False
    assert res_crash.passed == 0
    assert res_crash.failed == 1
    assert res_crash.total == 1
    assert res_crash.error_type == "SyntaxError"


def test_detect_error_type(exec_cfg: DictConfig) -> None:
    executor = CodeExecutor(exec_cfg)
    
    assert executor._detect_error_type("ZeroDivisionError: division by zero") == "ZeroDivisionError"
    assert executor._detect_error_type("NameError: name 'foo' is not defined") == "NameError"
    assert executor._detect_error_type("Some unexpected message") == "UnknownError"
    assert executor._detect_error_type("") is None
