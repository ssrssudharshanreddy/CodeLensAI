"""
Unit tests for all CodeLens AI tools.
"""

from pathlib import Path
import pytest

from codelens.tools.file_tools import InspectFileTool, ListFilesTool, SearchWorkspaceTool
from codelens.tools.syntax_tools import CheckSyntaxTool
from codelens.tools.error_tools import AnalyzeErrorTool
from codelens.tools.test_gen_tools import GenerateTestsTool
from codelens.tools.sandbox import RunCodeSandboxTool, RunTestsTool
from codelens.tools.patch_tools import CreatePatchTool, ApplyPatchTool, RollbackPatchTool


def test_file_inspection_and_listing(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("line 1\nline 2\nline 3\n")

    inspect_tool = InspectFileTool(workspace_root=tmp_path)
    res = inspect_tool.execute(path="sample.py", start_line=1, end_line=2)
    assert res.success is True
    assert "line 1" in res.output
    assert "line 2" in res.output
    assert res.data["total_lines"] == 3

    list_tool = ListFilesTool(workspace_root=tmp_path)
    res_list = list_tool.execute()
    assert res_list.success is True
    assert any("sample.py" in x for x in res_list.data["files"])


def test_search_workspace(tmp_path: Path):
    f1 = tmp_path / "module_a.py"
    f1.write_text("def find_me():\n    return 42\n")

    search_tool = SearchWorkspaceTool(workspace_root=tmp_path)
    res = search_tool.execute(query="find_me")
    assert res.success is True
    assert len(res.data["matches"]) >= 1
    assert "module_a.py:1" in res.data["matches"][0]


def test_syntax_checker_valid_and_invalid(tmp_path: Path):
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("def add(a, b):\n    return a + b\n")

    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("def broken(\n")

    syntax_tool = CheckSyntaxTool(workspace_root=tmp_path)

    # Valid syntax
    res_valid = syntax_tool.execute(path="valid.py")
    assert res_valid.success is True
    assert "add" in res_valid.data["functions"]

    # Invalid syntax
    res_invalid = syntax_tool.execute(path="invalid.py")
    assert res_invalid.success is False
    assert res_invalid.data["valid_syntax"] is False
    assert "line" in res_invalid.data


def test_error_analyzer():
    traceback_sample = (
        'Traceback (most recent call last):\n'
        '  File "calculator.py", line 12, in divide\n'
        '    return a / b\n'
        'ZeroDivisionError: division by zero'
    )
    error_tool = AnalyzeErrorTool()
    res = error_tool.execute(error_text=traceback_sample)
    assert res.success is True
    assert res.data["exception_type"] == "ZeroDivisionError"
    assert res.data["failing_frame"]["line"] == 12
    assert "Division or modulo by zero detected" in res.data["heuristic"]


def test_test_generation_tool(tmp_path: Path):
    gen_tool = GenerateTestsTool(workspace_root=tmp_path)
    code = (
        "import unittest\n"
        "class TestSample(unittest.TestCase):\n"
        "    def test_pass(self):\n"
        "        self.assertTrue(True)\n"
    )
    res = gen_tool.execute(test_file_path="tests/test_sample.py", test_code=code)
    assert res.success is True
    assert (tmp_path / "tests" / "test_sample.py").exists()


def test_patch_creation_application_and_rollback(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text("x = 1\ny = 2\n")

    patch_creator = CreatePatchTool(workspace_root=tmp_path)
    res_diff = patch_creator.execute(target_path="app.py", proposed_content="x = 1\ny = 99\n")
    assert res_diff.success is True
    assert res_diff.data["has_changes"] is True
    assert "+y = 99" in res_diff.data["diff"]

    patch_applier = ApplyPatchTool(workspace_root=tmp_path, require_approval=False)
    res_apply = patch_applier.execute(target_path="app.py", new_content="x = 1\ny = 99\n")
    assert res_apply.success is True
    assert target.read_text() == "x = 1\ny = 99\n"

    # Backup file should exist
    backup = tmp_path / "app.py.bak"
    assert backup.exists()
    assert backup.read_text() == "x = 1\ny = 2\n"

    # Test Rollback
    rollback_tool = RollbackPatchTool(workspace_root=tmp_path)
    res_rb = rollback_tool.execute(target_path="app.py")
    assert res_rb.success is True
    assert target.read_text() == "x = 1\ny = 2\n"


def test_sandbox_execution_and_timeout(tmp_path: Path):
    # Quick normal script
    script = tmp_path / "quick.py"
    script.write_text("print('HELLO_SANDBOX')")

    sandbox_tool = RunCodeSandboxTool(workspace_root=tmp_path, timeout=5)
    res = sandbox_tool.execute(script_path="quick.py")
    assert res.success is True
    assert "HELLO_SANDBOX" in res.data["stdout"]

    # Infinite loop script to test timeout
    loop_script = tmp_path / "infinite.py"
    loop_script.write_text("import time\nwhile True:\n    time.sleep(0.1)\n")

    quick_timeout_tool = RunCodeSandboxTool(workspace_root=tmp_path, timeout=1)
    res_timeout = quick_timeout_tool.execute(script_path="infinite.py")
    assert res_timeout.success is False
    assert "TIMED OUT" in res_timeout.output
    assert res_timeout.data["timed_out"] is True


def test_run_tests_tool(tmp_path: Path):
    test_file = tmp_path / "test_math.py"
    test_file.write_text(
        "import unittest\n"
        "class TestMath(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(1 + 1, 2)\n"
    )
    runner_tool = RunTestsTool(workspace_root=tmp_path, timeout=10)
    res = runner_tool.execute(test_path="test_math.py", runner="unittest")
    assert res.success is True
    assert res.data["all_passed"] is True
    assert res.data["passed"] == 1
    assert res.is_assumption is False  # Verified fact
