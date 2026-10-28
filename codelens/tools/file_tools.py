"""
Workspace file inspection, listing, and search tools for CodeLens AI.
All operations strictly respect workspace containment guardrails.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult
from codelens.utils.security import validate_safe_path, SecurityError


class InspectFileInput(BaseModel):
    path: str = Field(description="Relative path to file within workspace")
    start_line: Optional[int] = Field(default=1, description="1-indexed line number to start reading from")
    end_line: Optional[int] = Field(default=None, description="1-indexed line number to stop reading (inclusive)")


class InspectFileTool(BaseTool):
    name = "inspect_file"
    description = "Read the contents of a file in the workspace with line numbers."
    args_schema = InspectFileInput

    def execute(self, path: str, start_line: Optional[int] = 1, end_line: Optional[int] = None) -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(path, Path(self.workspace_root))
            if not safe_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            if not safe_path.is_file():
                return ToolResult(success=False, output="", error=f"Path is not a regular file: {path}")

            # Read content with utf-8
            text = safe_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total_lines = len(lines)

            s_line = max(1, start_line or 1)
            e_line = min(total_lines, end_line or total_lines) if end_line else total_lines

            if s_line > total_lines:
                return ToolResult(
                    success=True,
                    output=f"File '{path}' has only {total_lines} lines (requested start line {s_line}).",
                    data={"total_lines": total_lines, "lines": []},
                )

            selected_lines = lines[s_line - 1 : e_line]
            formatted_lines = [
                f"{i:4d} | {line}" for i, line in enumerate(selected_lines, start=s_line)
            ]
            header = f"--- Content of '{path}' (lines {s_line}-{e_line} of {total_lines}) ---\n"
            output = header + "\n".join(formatted_lines)

            return ToolResult(
                success=True,
                output=output,
                data={
                    "path": path,
                    "total_lines": total_lines,
                    "start_line": s_line,
                    "end_line": e_line,
                    "raw_content": "\n".join(selected_lines),
                },
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to inspect file '{path}': {e}")


class ListFilesInput(BaseModel):
    directory: str = Field(default=".", description="Relative directory path to inspect (defaults to root '.')")
    pattern: str = Field(default="*", description="Glob pattern filter, e.g. '*.py' or '*'")


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files and directories in the workspace, ignoring VCS and virtualenv directories."
    args_schema = ListFilesInput

    IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".idea", ".vscode", "build", "dist"}

    def execute(self, directory: str = ".", pattern: str = "*") -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_dir = validate_safe_path(directory, Path(self.workspace_root))
            if not safe_dir.exists() or not safe_dir.is_dir():
                return ToolResult(success=False, output="", error=f"Directory not found: {directory}")

            workspace_path = Path(self.workspace_root).resolve()
            results = []

            for root_str, dirs, files in safe_dir.walk() if hasattr(safe_dir, "walk") else __import__("os").walk(safe_dir):
                root_path = Path(root_str)
                # Filter out ignored directories in-place
                dirs[:] = [d for d in dirs if d not in self.IGNORED_DIRS]

                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        full_path = root_path / file
                        try:
                            rel_path = full_path.relative_to(workspace_path)
                            size = full_path.stat().st_size
                            results.append(f"{rel_path.as_posix()} ({size} bytes)")
                        except ValueError:
                            continue

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No files matching pattern '{pattern}' found in '{directory}'.",
                    data={"files": []},
                )

            results.sort()
            output = f"Files found in '{directory}' matching '{pattern}':\n" + "\n".join(f"- {f}" for f in results)
            return ToolResult(success=True, output=output, data={"files": results})
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to list files in '{directory}': {e}")


class SearchWorkspaceInput(BaseModel):
    query: str = Field(description="Search string or keyword to find across files")
    file_pattern: str = Field(default="*.py", description="File glob filter, e.g. '*.py'")


class SearchWorkspaceTool(BaseTool):
    name = "search_workspace"
    description = "Search for a keyword or string across workspace source code files."
    args_schema = SearchWorkspaceInput

    IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}

    def execute(self, query: str, file_pattern: str = "*.py") -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            workspace_path = Path(self.workspace_root).resolve()
            matches = []

            for root_str, dirs, files in __import__("os").walk(workspace_path):
                dirs[:] = [d for d in dirs if d not in self.IGNORED_DIRS]
                for file in files:
                    if fnmatch.fnmatch(file, file_pattern):
                        full_path = Path(root_str) / file
                        try:
                            rel_path = full_path.relative_to(workspace_path)
                            text = full_path.read_text(encoding="utf-8", errors="replace")
                            for line_no, line in enumerate(text.splitlines(), start=1):
                                if query in line:
                                    matches.append(f"{rel_path.as_posix()}:{line_no} | {line.strip()}")
                        except Exception:
                            continue

            if not matches:
                return ToolResult(
                    success=True,
                    output=f"No occurrences of '{query}' found matching '{file_pattern}'.",
                    data={"matches": []},
                )

            output = f"Search results for '{query}' ({len(matches)} matches):\n" + "\n".join(matches[:50])
            return ToolResult(success=True, output=output, data={"matches": matches})
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Search failed: {e}")
