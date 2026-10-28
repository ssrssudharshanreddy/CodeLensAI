"""
Unit tests for structured tool calling protocol and response parser.
"""

import pytest
from pydantic import BaseModel, Field

from codelens.llm.protocol import ToolProtocol, ToolProtocolError, ActionDecision


class DummyArgsSchema(BaseModel):
    path: str = Field(description="Target file")
    count: int = Field(default=1, description="Count value")


def test_clean_json_text_markdown_stripping():
    wrapped = "```json\n{\n  \"thought\": \"Thinking\",\n  \"tool_name\": \"finish\",\n  \"tool_args\": {}\n}\n```"
    cleaned = ToolProtocol.clean_json_text(wrapped)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")


def test_clean_json_text_surrounding_commentary():
    text = "Here is my response:\n{\"thought\": \"Analyze\", \"tool_name\": \"finish\", \"tool_args\": {}}\nHope that helps!"
    cleaned = ToolProtocol.clean_json_text(text)
    assert cleaned == '{"thought": "Analyze", "tool_name": "finish", "tool_args": {}}'


def test_parse_response_valid():
    raw = '{"thought": "Inspect code", "plan": "Step 1", "tool_name": "inspect_file", "tool_args": {"path": "main.py"}}'
    decision = ToolProtocol.parse_response(raw, valid_tools=["inspect_file"])
    assert decision.thought == "Inspect code"
    assert decision.plan == "Step 1"
    assert decision.tool_name == "inspect_file"
    assert decision.tool_args == {"path": "main.py"}


def test_parse_response_unknown_tool():
    raw = '{"thought": "Execute command", "tool_name": "shell_exec", "tool_args": {"cmd": "ls"}}'
    with pytest.raises(ToolProtocolError) as exc_info:
        ToolProtocol.parse_response(raw, valid_tools=["inspect_file", "check_syntax"])
    assert "Unknown tool 'shell_exec'" in str(exc_info.value)


def test_parse_response_schema_validation():
    schemas = {"dummy_tool": DummyArgsSchema}
    # Missing required argument 'path'
    raw_invalid = '{"thought": "Running", "tool_name": "dummy_tool", "tool_args": {"count": 5}}'
    with pytest.raises(ToolProtocolError) as exc_info:
        ToolProtocol.parse_response(raw_invalid, valid_tools=["dummy_tool"], tool_schemas=schemas)
    assert "Invalid arguments for tool 'dummy_tool'" in str(exc_info.value)

    # Valid arguments
    raw_valid = '{"thought": "Running", "tool_name": "dummy_tool", "tool_args": {"path": "test.txt", "count": 2}}'
    decision = ToolProtocol.parse_response(raw_valid, valid_tools=["dummy_tool"], tool_schemas=schemas)
    assert decision.tool_args["path"] == "test.txt"
    assert decision.tool_args["count"] == 2


def test_parse_response_aliases_and_unwrapping():
    # Test action key alias, args alias, and argument name alias for inspect_file
    raw_alias = '{"action": "inspect_file", "args": {"file_path": "foo.py"}, "explanation": "Reading file"}'
    decision = ToolProtocol.parse_response(raw_alias, valid_tools=["inspect_file"])
    assert decision.tool_name == "inspect_file"
    assert decision.tool_args["path"] == "foo.py"
    assert decision.thought == "Reading file"

    # Test nested container wrapper
    raw_nested = '{"decision": {"tool_name": "inspect_file", "tool_args": {"path": "bar.py"}}, "thought": "Checking"}'
    decision_nested = ToolProtocol.parse_response(raw_nested, valid_tools=["inspect_file"])
    assert decision_nested.tool_name == "inspect_file"
    assert decision_nested.tool_args["path"] == "bar.py"
