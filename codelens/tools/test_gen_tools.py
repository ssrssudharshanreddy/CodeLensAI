"""
Test generation tool for CodeLens AI.
Safely creates isolated unit test cases within workspace to verify bug fixes.
"""

from __future__ import annotations

import ast
from pathlib import Path
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult
from codelens.utils.security import validate_safe_path, SecurityError


class GenerateTestsInput(BaseModel):
    test_file_path: str = Field(description="Relative path where the test file should be saved (e.g. 'tests/test_fix.py')")
    test_code: str = Field(description="Complete executable Python unit test code (unittest or pytest format)")


class GenerateTestsTool(BaseTool):
    name = "generate_tests"
    description = "Generate and save a Python test file in the workspace to reproduce a bug or verify a fix."
    args_schema = GenerateTestsInput

    def execute(self, test_file_path: str, test_code: str) -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(test_file_path, Path(self.workspace_root))

            # Validate syntax of generated test code before saving
            try:
                ast.parse(test_code, filename=test_file_path)
            except SyntaxError as syn_err:
                return ToolResult(
                    success=False,
                    output=f"Generated test code has syntax error: {syn_err.msg} at line {syn_err.lineno}",
                    error=f"SyntaxError in generated tests: {syn_err.msg}",
                )

            # Ensure parent directories exist
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # Write test file
            safe_path.write_text(test_code, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Successfully generated and saved test suite to '{test_file_path}'.",
                data={
                    "path": test_file_path,
                    "bytes_written": len(test_code.encode("utf-8")),
                    "lines": len(test_code.splitlines()),
                },
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to generate test file: {e}")
