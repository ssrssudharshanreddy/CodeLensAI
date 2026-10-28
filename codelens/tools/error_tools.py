"""
Error-message and traceback analysis tool for CodeLens AI.
Parses stack traces, extracts exception hierarchies, failing source locations,
and classifies error categories with debugging heuristics.
"""

from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult


ERROR_HEURISTICS = {
    "ZeroDivisionError": (
        "Division or modulo by zero detected. "
        "Check denominator values, guard with 'if divisor != 0:' or handle empty collections."
    ),
    "IndexError": (
        "Sequence index out of range. "
        "Commonly caused by off-by-one loop conditions (e.g. range(len(seq) + 1)), "
        "accessing index == len(list), or indexing into an empty list."
    ),
    "KeyError": (
        "Dictionary key not found. "
        "Verify key existence using 'in' or 'dict.get(key, default)'."
    ),
    "TypeError": (
        "Operation performed on inappropriate type (e.g. concatenating str and int, "
        "calling an uncallable object, or missing required function arguments)."
    ),
    "AttributeError": (
        "Object has no such attribute. "
        "Frequently caused by variables being None unexpectedly or misnamed methods/fields."
    ),
    "NameError": (
        "Variable or function name is referenced before declaration or misspelled."
    ),
    "ValueError": (
        "Function received an argument of correct type but inappropriate value "
        "(e.g. int('invalid_string') or unpacking wrong number of elements)."
    ),
    "RecursionError": (
        "Maximum recursion depth exceeded. "
        "Inspect recursion base cases and termination criteria."
    ),
    "FileNotFoundError": (
        "Path does not exist. Check relative vs absolute paths and working directory."
    ),
}


class AnalyzeErrorInput(BaseModel):
    error_text: str = Field(description="Raw error message, traceback, or exception output")


class AnalyzeErrorTool(BaseTool):
    name = "analyze_error"
    description = "Parse and analyze a traceback or error message to extract failing location, exception type, and root cause hints."
    args_schema = AnalyzeErrorInput

    def execute(self, error_text: str) -> ToolResult:
        if not error_text or not error_text.strip():
            return ToolResult(success=False, output="", error="Empty error text provided.")

        clean_text = error_text.strip()

        # Find stack frames: File "...", line 123, in func
        frame_pattern = re.compile(
            r'File\s+["\'](?P<file>[^"\']+)["\'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>\w+))?(?:\n\s+(?P<code>.+))?'
        )
        frames = []
        for match in frame_pattern.finditer(clean_text):
            frames.append({
                "file": match.group("file"),
                "line": int(match.group("line")),
                "func": match.group("func") or "module",
                "code": (match.group("code") or "").strip(),
            })

        # Find exception line: e.g. "ZeroDivisionError: division by zero" or "AssertionError: ..."
        exc_pattern = re.compile(r"([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception|AssertionError):\s*(.*)")
        exc_match = None
        for line in reversed(clean_text.splitlines()):
            m = exc_pattern.search(line)
            if m:
                exc_match = m
                break

        exc_type = exc_match.group(1) if exc_match else "UnknownError"
        exc_msg = exc_match.group(2).strip() if exc_match else clean_text.splitlines()[-1]

        heuristic = ERROR_HEURISTICS.get(
            exc_type,
            "Inspect the deepest stack frame and check variable states leading up to the failure."
        )

        failing_frame = frames[-1] if frames else None

        output_lines = [
            f"[ERROR DIAGNOSIS] Exception: {exc_type}",
            f"Details: {exc_msg}",
        ]
        if failing_frame:
            output_lines.append(
                f"Failing Location: {failing_frame['file']} at line {failing_frame['line']} in {failing_frame['func']}()"
            )
            if failing_frame['code']:
                output_lines.append(f"Offending Line: {failing_frame['code']}")
        output_lines.append(f"\nHeuristic Guidance: {heuristic}")

        return ToolResult(
            success=True,
            output="\n".join(output_lines),
            data={
                "exception_type": exc_type,
                "exception_message": exc_msg,
                "frames": frames,
                "failing_frame": failing_frame,
                "heuristic": heuristic,
            },
        )
