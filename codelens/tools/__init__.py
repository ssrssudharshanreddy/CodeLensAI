"""
CodeLens AI Tools package.
"""

from codelens.tools.base import BaseTool, ToolResult
from codelens.tools.file_tools import InspectFileTool, ListFilesTool, SearchWorkspaceTool
from codelens.tools.syntax_tools import CheckSyntaxTool
from codelens.tools.error_tools import AnalyzeErrorTool
from codelens.tools.test_gen_tools import GenerateTestsTool
from codelens.tools.sandbox import RunCodeSandboxTool, RunTestsTool, TestRunResult
from codelens.tools.patch_tools import CreatePatchTool, ApplyPatchTool, RollbackPatchTool
from codelens.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolResult",
    "InspectFileTool",
    "ListFilesTool",
    "SearchWorkspaceTool",
    "CheckSyntaxTool",
    "AnalyzeErrorTool",
    "GenerateTestsTool",
    "RunCodeSandboxTool",
    "RunTestsTool",
    "TestRunResult",
    "CreatePatchTool",
    "ApplyPatchTool",
    "RollbackPatchTool",
    "ToolRegistry",
]
