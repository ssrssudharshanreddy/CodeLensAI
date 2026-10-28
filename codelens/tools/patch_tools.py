"""
Patch generation, diff visualization, and safe modification tools for CodeLens AI.
Requires explicit user approval before modifying existing files, creates
automatic backups, and verifies syntax before saving.
"""

from __future__ import annotations

import ast
import difflib
import shutil
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from codelens.tools.base import BaseTool, ToolResult
from codelens.utils.security import (
    validate_safe_path,
    request_user_approval,
    SecurityError,
    ApprovalDeniedError,
)


class CreatePatchInput(BaseModel):
    target_path: str = Field(description="Relative path of file to patch")
    proposed_content: str = Field(description="The complete new content proposed for the file")


class CreatePatchTool(BaseTool):
    name = "create_patch"
    description = "Compute and preview a unified diff between current file content and proposed changes without modifying the file."
    args_schema = CreatePatchInput

    def execute(self, target_path: str, proposed_content: str) -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(target_path, Path(self.workspace_root))
            if not safe_path.exists():
                return ToolResult(success=False, output="", error=f"Target file does not exist: {target_path}")

            original_text = safe_path.read_text(encoding="utf-8", errors="replace")
            orig_lines = original_text.splitlines(keepends=True)
            new_lines = proposed_content.splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    orig_lines,
                    new_lines,
                    fromfile=f"a/{target_path}",
                    tofile=f"b/{target_path}",
                    n=3,
                )
            )

            if not diff_lines:
                return ToolResult(
                    success=True,
                    output=f"No differences found: proposed content is identical to existing content of '{target_path}'.",
                    data={"target_path": target_path, "diff": "", "has_changes": False},
                )

            diff_text = "".join(diff_lines)
            output = f"Proposed Diff for '{target_path}':\n\n{diff_text}"

            return ToolResult(
                success=True,
                output=output,
                data={
                    "target_path": target_path,
                    "diff": diff_text,
                    "has_changes": True,
                    "lines_added": sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")),
                    "lines_removed": sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---")),
                },
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Patch generation error: {e}")


class ApplyPatchInput(BaseModel):
    target_path: str = Field(description="Relative path to file being modified")
    new_content: str = Field(description="The validated new content to write to the file")
    reason: str = Field(default="Applying bug fix", description="Explanation of why this patch is being applied")


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = "Apply a proposed patch to a file. Requires user approval, creates an automatic backup (.bak), and validates syntax."
    args_schema = ApplyPatchInput

    def __init__(self, workspace_root: Optional[Any] = None, require_approval: bool = True):
        super().__init__(workspace_root)
        self.require_approval = require_approval

    def execute(self, target_path: str, new_content: str, reason: str = "Applying bug fix") -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(target_path, Path(self.workspace_root))
            if not safe_path.exists():
                return ToolResult(success=False, output="", error=f"Target file does not exist: {target_path}")

            original_text = safe_path.read_text(encoding="utf-8", errors="replace")

            # Check Python syntax before writing
            if safe_path.suffix.lower() == ".py":
                try:
                    ast.parse(new_content, filename=target_path)
                except SyntaxError as syn_err:
                    return ToolResult(
                        success=False,
                        output=f"Proposed patch contains syntax error: {syn_err.msg} at line {syn_err.lineno}",
                        error=f"SyntaxError in patch: {syn_err.msg}",
                    )

            # Compute diff for preview
            orig_lines = original_text.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff_lines = list(
                difflib.unified_diff(
                    orig_lines,
                    new_lines,
                    fromfile=f"a/{target_path}",
                    tofile=f"b/{target_path}",
                    n=3,
                )
            )
            diff_text = "".join(diff_lines)

            # Request user approval if required
            try:
                request_user_approval(
                    action_summary=f"Modify '{target_path}' ({reason})",
                    details=f"Unified Diff:\n{diff_text}" if diff_text else "No content changes",
                    auto_approve=not self.require_approval,
                )
            except ApprovalDeniedError as ade:
                return ToolResult(
                    success=False,
                    output=f"Patch application rejected by user: {ade}",
                    error="ApprovalDenied",
                    data={"rejected": True},
                )

            # Create backup file (.bak)
            backup_path = safe_path.with_name(safe_path.name + ".bak")
            shutil.copy2(safe_path, backup_path)

            # Write new content
            safe_path.write_text(new_content, encoding="utf-8")

            output = (
                f"[PATCH APPLIED] Successfully updated '{target_path}'.\n"
                f"Automatic backup preserved at '{backup_path.name}'.\n\n"
                f"Diff applied:\n{diff_text if diff_text else '(no net changes)'}"
            )

            return ToolResult(
                success=True,
                output=output,
                data={
                    "target_path": target_path,
                    "backup_path": str(backup_path),
                    "diff": diff_text,
                },
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to apply patch: {e}")


class RollbackPatchInput(BaseModel):
    target_path: str = Field(description="Relative path of file to restore from .bak backup")


class RollbackPatchTool(BaseTool):
    name = "rollback_patch"
    description = "Restore a file from its .bak backup created during a previous patch."
    args_schema = RollbackPatchInput

    def execute(self, target_path: str) -> ToolResult:
        if not self.workspace_root:
            return ToolResult(success=False, output="", error="Workspace root not configured.")

        try:
            safe_path = validate_safe_path(target_path, Path(self.workspace_root))
            backup_path = safe_path.with_name(safe_path.name + ".bak")

            if not backup_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No backup file '{backup_path.name}' found for '{target_path}'.",
                )

            shutil.copy2(backup_path, safe_path)
            return ToolResult(
                success=True,
                output=f"[ROLLBACK SUCCESS] Restored '{target_path}' from backup '{backup_path.name}'.",
                data={"restored": target_path},
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=f"Security violation: {se}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Rollback failed: {e}")
