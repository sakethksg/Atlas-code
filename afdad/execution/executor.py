"""
Code executor — runs generated code in a subprocess.

Provides timeout enforcement and stdout/stderr capture.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.utils.logging import get_logger
from afdad.utils.models import ExecutionResult


class CodeExecutor:
    """Subprocess-based executor for running generated Python code.

    Parameters
    ----------
    cfg:
        Execution configuration (timeout_seconds, temp_dir).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.timeout = cfg.timeout_seconds
        self.temp_dir = Path(cfg.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    async def execute(
        self,
        code: str,
        test_code: str = "",
        entry_point: str = "",
    ) -> ExecutionResult:
        """Execute code with optional test cases in a subprocess.

        Parameters
        ----------
        code:
            The generated Python code (function definitions).
        test_code:
            Test code to run against the generated code.
        entry_point:
            The function name to test (used for constructing check calls).

        Returns
        -------
        ExecutionResult
            Structured execution result.
        """
        # Build the full script: code + tests
        full_script = self._build_script(code, test_code, entry_point)

        # Write to temp file
        script_path = self.temp_dir / f"exec_{os.getpid()}_{id(code)}.py"
        script_path.write_text(full_script, encoding="utf-8")

        start_time = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                elapsed = (time.perf_counter() - start_time) * 1000
                self.logger.warning(
                    f"Execution timed out after {self.timeout}s"
                )
                return ExecutionResult(
                    timeout=True,
                    returncode=-1,
                    stderr=f"Execution timed out after {self.timeout} seconds",
                    execution_time_ms=elapsed,
                )

            elapsed = (time.perf_counter() - start_time) * 1000
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = proc.returncode or 0

            # Parse test results from stdout
            result = self._parse_test_output(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                elapsed_ms=elapsed,
            )

            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Execution error: {exc}")
            return ExecutionResult(
                stderr=str(exc),
                returncode=-1,
                execution_time_ms=elapsed,
                error_type=type(exc).__name__,
            )
        finally:
            # Cleanup temp file
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _build_script(
        self, code: str, test_code: str, entry_point: str
    ) -> str:
        """Construct the full executable script."""
        parts = [
            "import sys",
            "import traceback as tb_module",
            "",
            "# Generated Code",
            code,
            "",
        ]

        if test_code:
            indented_test = "\n".join("        " + line for line in test_code.strip().split("\n"))
            parts.extend([
                "# Test Code",
                "passed = 0",
                "failed = 0",
                "total = 0",
                "",
                "def check(candidate):",
                "    global passed, failed, total",
                "    try:",
                indented_test,
                "        passed += 1",
                "    except Exception as e:",
                "        failed += 1",
                '        print(f"FAIL: {e}", file=sys.stderr)',
                "    total += 1",
                "",
                f"check({entry_point})" if entry_point else "",
                "",
                'print(f"RESULTS: passed={passed} failed={failed} total={total}")',
            ])
        else:
            # If no test code, just try to run/import the code
            parts.extend([
                '# No test code provided — checking for syntax/import errors only',
                'print("RESULTS: passed=1 failed=0 total=1")',
            ])

        return "\n".join(parts)

    def _parse_test_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        elapsed_ms: float,
    ) -> ExecutionResult:
        """Parse test results from stdout."""
        passed = 0
        failed = 0
        total = 0

        for line in stdout.split("\n"):
            if line.startswith("RESULTS:"):
                parts = line.split()
                for part in parts[1:]:
                    key, val = part.split("=")
                    if key == "passed":
                        passed = int(val)
                    elif key == "failed":
                        failed = int(val)
                    elif key == "total":
                        total = int(val)
                break

        # Detect error type from stderr
        error_type = self._detect_error_type(stderr)

        # Extract traceback
        traceback_str = ""
        if "Traceback" in stderr:
            tb_start = stderr.index("Traceback")
            traceback_str = stderr[tb_start:]

        # If returncode != 0 and no tests parsed, mark as failed
        if returncode != 0 and total == 0:
            total = 1
            failed = 1

        return ExecutionResult(
            passed=passed,
            failed=failed,
            total=total,
            stderr=stderr,
            stdout=stdout,
            returncode=returncode,
            execution_time_ms=elapsed_ms,
            error_type=error_type,
            traceback=traceback_str,
        )

    @staticmethod
    def _detect_error_type(stderr: str) -> str | None:
        """Detect the Python error type from stderr."""
        error_types = [
            "SyntaxError",
            "IndentationError",
            "NameError",
            "TypeError",
            "ValueError",
            "IndexError",
            "KeyError",
            "AttributeError",
            "ZeroDivisionError",
            "RecursionError",
            "MemoryError",
            "TimeoutError",
            "RuntimeError",
            "AssertionError",
            "ImportError",
        ]
        for err in error_types:
            if err in stderr:
                return err
        return None if not stderr else "UnknownError"
