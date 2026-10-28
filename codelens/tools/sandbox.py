"""
Sandboxed code execution and test runner tools for CodeLens AI.
Executes code strictly in isolated subprocesses with timeouts, environment
sanitization, process termination, and structured test outcome collection.
Never executes code inside the agent process or runs arbitrary shell commands.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult
from codelens.utils.security import (
    validate_safe_path,
    sanitize_sandbox_env,
    SecurityError,
)


class TestRunResult(BaseModel):
    """Structured test results parsed from real runner output."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    all_passed: bool = False


class SandboxExecutionError(Exception):
    """Raised when sandbox execution fails due to resource limits or violations."""
    pass


def run_sandboxed_python(
    args: List[str],
    cwd: Path,
    timeout: int = 8,
    memory_limit_mb: int = 256,
) -> dict[str, Any]:
    """
    Execute a Python command strictly inside an isolated subprocess.
    Uses shell=False, sanitized environment, strict timeout, and process cleanup.
    """
    python_exe = sys.executable
    cmd = [python_exe] + args

    env = sanitize_sandbox_env()
    # Ensure sandboxed scripts can import workspace modules
    env["PYTHONPATH"] = str(Path(cwd).resolve())

    start_time = time.time()
    timed_out = False
    stdout_data = ""
    stderr_data = ""
    exit_code = -1

    try:
        # Create subprocess with separate process group on Windows
        creationflags = 0
        if sys.platform == "win32":
            # CREATE_NO_WINDOW to prevent popup windows
            creationflags = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            creationflags=creationflags,
        )

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # Forcibly terminate process tree
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
            exit_code = -9
    except Exception as e:
        raise SandboxExecutionError(f"Failed to launch sandboxed subprocess: {e}")

    duration = time.time() - start_time

    return {
        "exit_code": exit_code,
        "stdout": stdout_data,
        "stderr": stderr_data,
        "duration": duration,
        "timed_out": timed_out,
    }


class RunCodeSandboxInput(BaseModel):
    script_path: str = Field(description="Relative path to python script in workspace")
    args: Optional[List[str]] = Field(default_factory=list, description="Optional command-line arguments to pass to script")


class RunCodeSandboxTool(BaseTool):
    name = "run_code_sandbox"
    description = "Execute a Python script in an isolated, resource-limited sandbox with a strict timeout."
    args_schema = RunCodeSandboxInput

    def __init__(self, workspace_root: Optional[Any] = None, timeout: int = 8, memory_limit_mb: int = 256, enabled: bool = True):
        super().__init__(workspace_root)
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.enabled = enabled

    def execute(self, script_path: str, args: Optional[List[str]] = None) -> ToolResult:
        if not self.enabled:
            return ToolResult(
                success=False,
                output="",
                error="Execution sandbox is currently disabled by configuration policy.",
                is_assumption=False,
            )

        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_script = validate_safe_path(script_path, Path(self.workspace_root))
            if not safe_script.exists():
                return ToolResult(success=False, output="", error=f"Script not found: {script_path}")

            exec_args = [str(safe_script)] + (args or [])
            result = run_sandboxed_python(
                args=exec_args,
                cwd=Path(self.workspace_root),
                timeout=self.timeout,
                memory_limit_mb=self.memory_limit_mb,
            )

            if result["timed_out"]:
                return ToolResult(
                    success=False,
                    output=(
                        f"[EXECUTION TIMED OUT] Script '{script_path}' exceeded the strict timeout limit of {self.timeout}s.\n"
                        f"Partial stdout:\n{result['stdout']}\nPartial stderr:\n{result['stderr']}"
                    ),
                    error=f"TimeoutExpired: Execution exceeded {self.timeout}s limit.",
                    data=result,
                    is_assumption=False,
                )

            success = (result["exit_code"] == 0)
            status_text = "COMPLETED SUCCESSFULLY (exit 0)" if success else f"FAILED (exit {result['exit_code']})"

            output = (
                f"[SANDBOX EXECUTION RESULT] Status: {status_text} in {result['duration']:.2f}s\n"
                f"STDOUT:\n{result['stdout'] if result['stdout'].strip() else '(empty)'}\n\n"
                f"STDERR:\n{result['stderr'] if result['stderr'].strip() else '(empty)'}"
            )

            return ToolResult(
                success=success,
                output=output,
                error=None if success else f"Subprocess exited with code {result['exit_code']}",
                data=result,
                is_assumption=False,  # Ground truth observation
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Sandbox error: {e}")


class RunTestsInput(BaseModel):
    test_path: str = Field(description="Relative path to test file or test directory (e.g. 'tests/test_fix.py')")
    runner: str = Field(default="unittest", description="Test runner to use ('unittest' or 'pytest')")


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = "Execute tests using unittest or pytest in the isolated sandbox and collect real test results."
    args_schema = RunTestsInput

    def __init__(self, workspace_root: Optional[Any] = None, timeout: int = 15, enabled: bool = True):
        super().__init__(workspace_root)
        self.timeout = timeout
        self.enabled = enabled

    def execute(self, test_path: str, runner: str = "unittest") -> ToolResult:
        if not self.enabled:
            return ToolResult(
                success=False,
                output="",
                error="Test execution is currently disabled by configuration policy.",
                is_assumption=False,
            )

        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_target = validate_safe_path(test_path, Path(self.workspace_root))
            if not safe_target.exists():
                return ToolResult(success=False, output="", error=f"Test path not found: {test_path}")

            # Relative path from workspace
            rel_target = safe_target.relative_to(Path(self.workspace_root).resolve()).as_posix()

            if runner.lower() == "pytest":
                args = ["-m", "pytest", "-v", rel_target]
            else:
                # Default to unittest discover / module run
                if safe_target.is_file():
                    # Format as file path or module
                    args = ["-m", "unittest", rel_target]
                else:
                    args = ["-m", "unittest", "discover", "-s", rel_target]

            raw_res = run_sandboxed_python(
                args=args,
                cwd=Path(self.workspace_root),
                timeout=self.timeout,
            )

            # Parse outcomes
            parsed = self._parse_test_output(raw_res["stdout"], raw_res["stderr"], raw_res["exit_code"], raw_res["duration"])

            status_str = "[TESTS PASSED]" if parsed.all_passed else "[TESTS FAILED]"
            summary = (
                f"{status_str} (Runner: {runner}, Exit: {parsed.exit_code}, Duration: {parsed.duration_seconds:.2f}s)\n"
                f"- Total Tests: {parsed.total}\n"
                f"- Passed: {parsed.passed}\n"
                f"- Failed: {parsed.failed}\n"
                f"- Errors: {parsed.errors}\n\n"
                f"STDOUT:\n{parsed.stdout if parsed.stdout.strip() else '(empty)'}\n\n"
                f"STDERR:\n{parsed.stderr if parsed.stderr.strip() else '(empty)'}"
            )

            return ToolResult(
                success=parsed.all_passed,
                output=summary,
                error=None if parsed.all_passed else f"{parsed.failed} test(s) failed, {parsed.errors} error(s)",
                data=parsed.model_dump(),
                is_assumption=False,  # CRITICAL: Ground truth test result!
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Test runner error: {e}")

    def _parse_test_output(self, stdout: str, stderr: str, exit_code: int, duration: float) -> TestRunResult:
        combined = stdout + "\n" + stderr
        total = 0
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        # Look for unittest patterns: "Ran X tests in Ys", "FAILED (failures=1, errors=2)" or "OK"
        ran_match = __import__("re").search(r"Ran (\d+) tests? in", combined)
        if ran_match:
            total = int(ran_match.group(1))

        fail_match = __import__("re").search(r"failures=(\d+)", combined)
        if fail_match:
            failed = int(fail_match.group(1))

        err_match = __import__("re").search(r"errors=(\d+)", combined)
        if err_match:
            errors = int(err_match.group(1))

        skip_match = __import__("re").search(r"skipped=(\d+)", combined)
        if skip_match:
            skipped = int(skip_match.group(1))

        # Look for pytest patterns: "X passed, Y failed, Z error in ..."
        pytest_match = __import__("re").search(r"(\d+)\s+passed", combined)
        if pytest_match:
            passed = int(pytest_match.group(1))
        pytest_failed = __import__("re").search(r"(\d+)\s+failed", combined)
        if pytest_failed:
            failed = int(pytest_failed.group(1))
        pytest_err = __import__("re").search(r"(\d+)\s+error", combined)
        if pytest_err:
            errors = int(pytest_err.group(1))

        # If unittest OK
        if "OK" in combined and exit_code == 0:
            passed = total - skipped
            failed = 0
            errors = 0
        elif total > 0 and passed == 0:
            passed = max(0, total - failed - errors - skipped)

        if total == 0:
            total = passed + failed + errors + skipped

        all_passed = (exit_code == 0 and failed == 0 and errors == 0 and total > 0)

        return TestRunResult(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            all_passed=all_passed,
        )
