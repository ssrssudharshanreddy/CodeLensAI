"""
Tool registry and execution dispatcher for CodeLens AI.
Validates tool existence, checks schemas, and manages tool discovery.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from codelens.tools.base import BaseTool, ToolResult
from codelens.tools.file_tools import InspectFileTool, ListFilesTool, SearchWorkspaceTool
from codelens.tools.syntax_tools import CheckSyntaxTool
from codelens.tools.error_tools import AnalyzeErrorTool
from codelens.tools.test_gen_tools import GenerateTestsTool
from codelens.tools.sandbox import RunCodeSandboxTool, RunTestsTool
from codelens.tools.patch_tools import CreatePatchTool, ApplyPatchTool, RollbackPatchTool


class ToolRegistry:
    """Central registry of executable agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Lookup a registered tool by name."""
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> Dict[str, Type[BaseModel]]:
        """Return mapping of tool name to Pydantic args schema class."""
        return {name: tool.args_schema for name, tool in self._tools.items()}

    def get_tools_prompt_description(self) -> str:
        """Generate formatted documentation of tools and schemas for LLM system prompts."""
        lines = ["AVAILABLE TOOLS:"]
        for name, tool in sorted(self._tools.items()):
            schema = tool.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            lines.append(f"\n- `{name}`: {tool.description}")
            lines.append("  Arguments schema:")
            for prop_name, prop_meta in properties.items():
                is_req = "required" if prop_name in required else "optional"
                p_type = prop_meta.get("type", "any")
                desc = prop_meta.get("description", "")
                lines.append(f"    * `{prop_name}` ({p_type}, {is_req}): {desc}")

        # Also document 'finish' action
        lines.append("\n- `finish`: Complete the task and provide final solution summary.")
        lines.append("  Arguments schema:")
        lines.append("    * `root_cause` (string, required): Root cause of the issue")
        lines.append("    * `solution_summary` (string, required): Explanation of the fix")
        lines.append("    * `test_results_summary` (string, required): Actual test verification evidence")
        lines.append("    * `remaining_uncertainty` (string, required): Any caveats or edge cases")

        return "\n".join(lines)

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool with error encapsulation."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' is not registered. Available tools: {', '.join(self.list_names())}",
            )

        try:
            return tool.execute(**args)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Unhandled exception executing tool '{name}': {e}",
            )

    @classmethod
    def create_default(
        cls,
        workspace_root: Any,
        sandbox_timeout: int = 8,
        sandbox_memory_mb: int = 256,
        sandbox_enabled: bool = True,
        require_approval: bool = True,
    ) -> ToolRegistry:
        """Instantiate and populate standard tool set configured with workspace."""
        reg = cls()
        reg.register(InspectFileTool(workspace_root=workspace_root))
        reg.register(ListFilesTool(workspace_root=workspace_root))
        reg.register(SearchWorkspaceTool(workspace_root=workspace_root))
        reg.register(CheckSyntaxTool(workspace_root=workspace_root))
        reg.register(AnalyzeErrorTool(workspace_root=workspace_root))
        reg.register(GenerateTestsTool(workspace_root=workspace_root))
        reg.register(
            RunCodeSandboxTool(
                workspace_root=workspace_root,
                timeout=sandbox_timeout,
                memory_limit_mb=sandbox_memory_mb,
                enabled=sandbox_enabled,
            )
        )
        reg.register(
            RunTestsTool(
                workspace_root=workspace_root,
                timeout=sandbox_timeout * 2,
                enabled=sandbox_enabled,
            )
        )
        reg.register(CreatePatchTool(workspace_root=workspace_root))
        reg.register(ApplyPatchTool(workspace_root=workspace_root, require_approval=require_approval))
        reg.register(RollbackPatchTool(workspace_root=workspace_root))
        return reg
