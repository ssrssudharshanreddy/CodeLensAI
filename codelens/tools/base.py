"""
Base tool interfaces and result abstractions for CodeLens AI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Encapsulates the outcome of a tool execution."""
    success: bool = Field(description="True if tool executed successfully without uncaught errors")
    output: str = Field(description="Human and model readable output/evidence")
    error: Optional[str] = Field(default=None, description="Detailed error description if execution failed")
    data: Optional[dict[str, Any]] = Field(default=None, description="Structured payload or metadata")
    is_assumption: bool = Field(default=False, description="Flag indicating if this represents an unverified assumption")

    def to_observation_text(self) -> str:
        """Format as an observation string for LLM history."""
        prefix = "[VERIFIED OBSERVATION]" if not self.is_assumption else "[UNVERIFIED ASSUMPTION]"
        status = "SUCCESS" if self.success else "FAILURE"
        content = self.output if self.success else f"Error: {self.error}\nDetails:\n{self.output}"
        return f"{prefix} Status: {status}\n{content}"


class BaseTool(ABC):
    """Abstract base class for all CodeLens agent tools."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    def __init__(self, workspace_root: Optional[Any] = None):
        self.workspace_root = workspace_root

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with validated arguments."""
        pass

    def get_schema_dict(self) -> dict[str, Any]:
        """Return the JSON schema dictionary for this tool's arguments."""
        return self.args_schema.model_json_schema()
