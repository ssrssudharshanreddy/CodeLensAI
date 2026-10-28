"""
Static syntax checking and language detection tools for CodeLens AI.
Performs AST-based static syntax parsing without executing untrusted code.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult
from codelens.utils.security import validate_safe_path, SecurityError


EXTENSION_MAP = {
    ".py": "Python",
    ".pyw": "Python",
    ".json": "JSON",
    ".md": "Markdown",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".txt": "Text",
    ".sh": "Shell",
    ".ps1": "PowerShell",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
}


def detect_language(path: Path, content: str) -> str:
    """Detect language from file suffix or content header."""
    ext = path.suffix.lower()
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]
    
    # Check shebang
    first_line = content.splitlines()[0] if content else ""
    if first_line.startswith("#!"):
        if "python" in first_line:
            return "Python"
        if "sh" in first_line or "bash" in first_line:
            return "Shell"

    return "Unknown/Plain Text"


class CheckSyntaxInput(BaseModel):
    path: str = Field(description="Relative path of file to check")


class CheckSyntaxTool(BaseTool):
    name = "check_syntax"
    description = "Perform static syntax verification and AST analysis on a file without executing it."
    args_schema = CheckSyntaxInput

    def execute(self, path: str) -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(path, Path(self.workspace_root))
            if not safe_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")

            code_text = safe_path.read_text(encoding="utf-8", errors="replace")
            language = detect_language(safe_path, code_text)

            if language == "Python":
                return self._check_python_syntax(path, code_text)
            elif language == "JSON":
                return self._check_json_syntax(path, code_text)
            else:
                return ToolResult(
                    success=True,
                    output=f"File detected as {language}. Static syntax analysis is currently optimized for Python and JSON.",
                    data={"language": language, "status": "skipped_non_python"},
                )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Syntax check error: {e}")

    def _check_python_syntax(self, path: str, code_text: str) -> ToolResult:
        try:
            tree = ast.parse(code_text, filename=path)
        except SyntaxError as syn_err:
            line_no = syn_err.lineno or 0
            offset = syn_err.offset or 0
            error_line = syn_err.text.rstrip() if syn_err.text else ""
            pointer = " " * (max(0, offset - 1)) + "^" if offset > 0 else ""

            msg = (
                f"[SYNTAX ERROR] in '{path}' at line {line_no}, col {offset}:\n"
                f"  {error_line}\n"
                f"  {pointer}\n"
                f"Message: {syn_err.msg}"
            )
            return ToolResult(
                success=False,
                output=msg,
                error=syn_err.msg,
                data={
                    "language": "Python",
                    "valid_syntax": False,
                    "line": line_no,
                    "column": offset,
                    "error_text": error_line,
                    "message": syn_err.msg,
                },
            )

        # Collect basic structural stats
        functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        summary = (
            f"[VALID SYNTAX] Python file '{path}' is syntactically sound.\n"
            f"- Total lines: {len(code_text.splitlines())}\n"
            f"- Classes defined: {', '.join(classes) if classes else 'None'}\n"
            f"- Functions defined: {', '.join(functions) if functions else 'None'}\n"
            f"- Modules imported: {', '.join(set(imports)) if imports else 'None'}"
        )

        return ToolResult(
            success=True,
            output=summary,
            data={
                "language": "Python",
                "valid_syntax": True,
                "classes": classes,
                "functions": functions,
                "imports": list(set(imports)),
            },
        )

    def _check_json_syntax(self, path: str, text: str) -> ToolResult:
        try:
            json.loads(text)
            return ToolResult(
                success=True,
                output=f"[VALID SYNTAX] JSON file '{path}' is valid.",
                data={"language": "JSON", "valid_syntax": True},
            )
        except json.JSONDecodeError as jde:
            msg = f"[SYNTAX ERROR] JSON decode failed in '{path}' at line {jde.lineno}, col {jde.colno}: {jde.msg}"
            return ToolResult(
                success=False,
                output=msg,
                error=jde.msg,
                data={"language": "JSON", "valid_syntax": False, "line": jde.lineno, "column": jde.colno},
            )
