"""
Structured tool-calling protocol and response validation for CodeLens AI.
Translates model text/JSON responses into validated tool execution calls.
Does not assume native tool calling; handles JSON parsing, markdown stripping,
schema validation, and informative retry diagnostics.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError


class ToolProtocolError(Exception):
    """Raised when model response cannot be parsed or fails schema validation."""
    pass


class ActionDecision(BaseModel):
    """Parsed model decision containing thought, plan, tool_name, and arguments."""
    thought: str = Field(description="Step-by-step reasoning about the task and current observations")
    plan: str = Field(default="", description="High-level plan or current step description")
    tool_name: str = Field(description="Selected tool name or 'finish'")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="Validated arguments for the tool")
    raw_response: str = Field(default="", description="The raw model output")


class ToolProtocol:
    """Manages prompt schemas and response validation."""

    SYSTEM_INSTRUCTION = """You are CodeLens AI, an autonomous software engineering pair-programmer and debugging agent.
You solve coding problems through a disciplined loop:
1. Formulate a short plan and state your reasoning in "thought".
2. Select a single appropriate tool from the registered tools list.
3. Pass valid arguments matching the tool's schema.
4. Read actual tool observations (distinguishing facts from assumptions).
5. If changes are needed, propose unified diffs, run tests, and verify outcomes.
6. When the task is verified and solved, invoke tool "finish".

STRICT OUTPUT FORMAT:
You MUST respond with a single, valid JSON object and nothing else.
Do NOT include preamble, markdown fences, or post-commentary outside the JSON.

JSON schema:
{
  "thought": "Your reasoning about current evidence and why this action is chosen",
  "plan": "Current step in the plan",
  "tool_name": "<exact_tool_name_or_finish>",
  "tool_args": {
    "<arg_name>": "<arg_value>"
  }
}

Special 'finish' tool arguments when done:
{
  "thought": "Verification complete. Tests passed.",
  "plan": "Complete",
  "tool_name": "finish",
  "tool_args": {
    "root_cause": "Exact bug description and why it occurred",
    "solution_summary": "Explanation of fix applied",
    "test_results_summary": "Summary of real tests executed and their verified outcomes",
    "remaining_uncertainty": "Any assumptions, edge cases, or potential risks"
  }
}
"""

    @classmethod
    def clean_json_text(cls, text: str) -> str:
        """Strip markdown formatting, code fences, or surrounding conversational text."""
        cleaned = text.strip()

        # Remove ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # If still not starting with {, search for outermost JSON object
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        return cleaned

    @classmethod
    def normalize_tool_arguments(cls, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize common parameter name variations from different LLMs."""
        normalized = dict(args)

        # Path / File aliases
        if tool_name in ("inspect_file", "check_syntax"):
            if "path" not in normalized:
                for alias in ("file_path", "filepath", "file", "filename", "target_file"):
                    if alias in normalized:
                        normalized["path"] = str(normalized[alias])
                        break

        elif tool_name in ("create_patch", "apply_patch", "rollback_patch", "generate_tests"):
            if "target_file" not in normalized:
                for alias in ("path", "file_path", "filepath", "file", "filename"):
                    if alias in normalized:
                        normalized["target_file"] = str(normalized[alias])
                        break

        elif tool_name == "run_code_sandbox":
            if "target_script" not in normalized and "code" not in normalized:
                for alias in ("script", "path", "file_path", "filepath", "file", "target_file"):
                    if alias in normalized:
                        normalized["target_script"] = str(normalized[alias])
                        break

        elif tool_name == "run_tests":
            if "test_target" not in normalized:
                for alias in ("test_file", "test_script", "path", "file_path", "target", "file"):
                    if alias in normalized:
                        normalized["test_target"] = str(normalized[alias])
                        break

        elif tool_name == "analyze_error":
            if "traceback_text" not in normalized:
                for alias in ("error", "traceback", "error_message", "output"):
                    if alias in normalized:
                        normalized["traceback_text"] = str(normalized[alias])
                        break

        return normalized

    @classmethod
    def parse_response(
        cls,
        raw_response: str,
        valid_tools: Optional[list[str]] = None,
        tool_schemas: Optional[dict[str, type[BaseModel]]] = None,
    ) -> ActionDecision:
        """
        Parse and validate raw model response string into an ActionDecision.

        Raises:
            ToolProtocolError: If JSON decoding fails, required keys are missing,
                               tool_name is unrecognized, or tool_args violate the tool's schema.
        """
        if not raw_response or not raw_response.strip():
            raise ToolProtocolError("Model returned an empty response.")

        cleaned = cls.clean_json_text(raw_response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ToolProtocolError(
                f"Invalid JSON in model output: {e.msg} at line {e.lineno}, col {e.colno}.\n"
                f"Raw output received was:\n{raw_response[:300]}"
            )

        if not isinstance(data, dict):
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                raise ToolProtocolError(f"Expected a JSON object, but got {type(data).__name__}.")

        # Unwrap if encapsulated under container keys like 'decision', 'action', 'response'
        for wrap_key in ("decision", "action_decision", "action", "tool_call", "response"):
            if wrap_key in data and isinstance(data[wrap_key], dict) and any(k in data[wrap_key] for k in ("tool_name", "tool", "name")):
                data = data[wrap_key]
                break

        thought = data.get("thought")
        if not thought or not isinstance(thought, str):
            # Fallback if thought is missing
            thought = str(data.get("reasoning", data.get("explanation", data.get("rationale", "Proceeding with next tool action."))))

        plan = str(data.get("plan", data.get("step", "")))

        tool_name = (
            data.get("tool_name")
            or data.get("tool")
            or data.get("name")
            or data.get("action")
            or data.get("function")
        )
        if not tool_name or not isinstance(tool_name, str):
            raise ToolProtocolError("Missing required field 'tool_name' in JSON response.")

        tool_name = tool_name.strip()

        tool_args = data.get("tool_args")
        if tool_args is None:
            tool_args = data.get("args") or data.get("arguments") or data.get("parameters") or {}

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {}

        if not isinstance(tool_args, dict):
            tool_args = {}

        # Normalize argument aliases across LLMs
        tool_args = cls.normalize_tool_arguments(tool_name, tool_args)

        # Validate against tool list if provided
        if valid_tools is not None:
            allowed = set(valid_tools) | {"finish"}
            if tool_name not in allowed:
                raise ToolProtocolError(
                    f"Unknown tool '{tool_name}'. Allowed tools are: {', '.join(sorted(allowed))}"
                )

        # Validate arguments against Pydantic schema if registered
        if tool_schemas and tool_name in tool_schemas:
            schema_cls = tool_schemas[tool_name]
            try:
                # Instantiate schema to validate arguments
                validated = schema_cls(**tool_args)
                tool_args = validated.model_dump()
            except ValidationError as ve:
                errors = []
                for err in ve.errors():
                    field = ".".join(str(loc) for loc in err["loc"])
                    errors.append(f"- Argument '{field}': {err['msg']}")
                raise ToolProtocolError(
                    f"Invalid arguments for tool '{tool_name}':\n" + "\n".join(errors)
                )

        return ActionDecision(
            thought=thought,
            plan=plan,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_response=raw_response,
        )

    @classmethod
    def build_format_reminder(cls, error_msg: str) -> str:
        """Generate a corrective prompt when the model violates the JSON protocol."""
        return (
            f"Your previous response had a formatting/validation error:\n{error_msg}\n\n"
            "Please correct this immediately. Output ONLY a valid JSON object matching:\n"
            "{\n"
            '  "thought": "<reasoning>",\n'
            '  "plan": "<current step>",\n'
            '  "tool_name": "<tool_name_or_finish>",\n'
            '  "tool_args": { ... }\n'
            "}"
        )
