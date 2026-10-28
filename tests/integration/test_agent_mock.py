"""
Integration tests for the complete CodeLens AI autonomous agent loop.
Uses mocked LLM responses with real tools and real sandbox execution.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from codelens.config import CodeLensConfig
from codelens.llm.ollama_client import OllamaClient, ModelInfo
from codelens.orchestrator.agent import CodeLensAgent
from codelens.orchestrator.state import AgentStatus
from codelens.utils.logger import AgentLogger


def test_agent_loop_end_to_end_mock(tmp_path: Path):
    # Setup buggy file in temporary workspace
    buggy_code = (
        "def divide(a: float, b: float) -> float:\n"
        "    return a / b\n"
    )
    (tmp_path / "calc.py").write_text(buggy_code, encoding="utf-8")

    # Setup unit test that asserts handling zero denominator
    test_code = (
        "import unittest\n"
        "from calc import divide\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_normal(self):\n"
        "        self.assertEqual(divide(10, 2), 5.0)\n\n"
        "    def test_zero_denominator(self):\n"
        "        self.assertEqual(divide(10, 0), 0.0)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    (tmp_path / "test_calc.py").write_text(test_code, encoding="utf-8")

    # Patched version of buggy_code
    fixed_code = (
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        return 0.0\n"
        "    return a / b\n"
    )

    # Mock OllamaClient responses simulating the agent's disciplined reasoning steps
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.default_model = "gemma4:e2b"
    mock_client.timeout = 30
    mock_client.check_connection.return_value = True
    mock_client.verify_model.return_value = ModelInfo(
        name="gemma4:e2b", parameter_size="4.6B", family="gemma4", context_length=131072
    )

    steps = [
        # Step 1: inspect buggy file
        {
            "thought": "I will inspect calc.py to understand how divide() is implemented.",
            "plan": "Inspect source code",
            "tool_name": "inspect_file",
            "tool_args": {"path": "calc.py"},
        },
        # Step 2: run tests in sandbox to observe failure
        {
            "thought": "I will run test_calc.py to observe the failure.",
            "plan": "Run initial tests",
            "tool_name": "run_tests",
            "tool_args": {"test_path": "test_calc.py", "runner": "unittest"},
        },
        # Step 3: create patch
        {
            "thought": "The test fails due to unhandled zero denominator. I will formulate a patch.",
            "plan": "Create patch",
            "tool_name": "create_patch",
            "tool_args": {"target_path": "calc.py", "proposed_content": fixed_code},
        },
        # Step 4: apply patch
        {
            "thought": "Applying the validated fix.",
            "plan": "Apply patch",
            "tool_name": "apply_patch",
            "tool_args": {"target_path": "calc.py", "new_content": fixed_code, "reason": "Guard against zero division"},
        },
        # Step 5: rerun tests in sandbox
        {
            "thought": "Rerunning test suite to verify all assertions pass.",
            "plan": "Verify fix with tests",
            "tool_name": "run_tests",
            "tool_args": {"test_path": "test_calc.py", "runner": "unittest"},
        },
        # Step 6: finish
        {
            "thought": "Tests passed in sandbox. Task is resolved.",
            "plan": "Complete task",
            "tool_name": "finish",
            "tool_args": {
                "root_cause": "Unhandled zero denominator in divide() caused ZeroDivisionError.",
                "solution_summary": "Added a guard condition 'if b == 0: return 0.0' to prevent zero division.",
                "test_results_summary": "All 2 unit tests passed successfully in sandbox.",
                "remaining_uncertainty": "Returning 0.0 is one convention; callers might alternatively expect ValueError.",
            },
        },
    ]

    mock_client.chat.side_effect = [json.dumps(s) for s in steps]

    config = CodeLensConfig(
        workspace_dir=tmp_path,
        max_iterations=8,
        require_approval=False,  # auto-approve in test
        sandbox_enabled=True,
    )
    logger = AgentLogger(verbose=False)

    agent = CodeLensAgent(
        config=config,
        client=mock_client,
        logger=logger,
    )

    state = agent.run(task_description="Fix zero division in calc.py and verify with test_calc.py")

    assert state.status == AgentStatus.SOLVED
    assert state.final_report is not None
    assert "ZeroDivisionError" in state.final_report.root_cause
    assert state.has_verified_test_pass() is True
    assert len(state.steps) == 6

    # Verify real file modification and backup preservation
    current_content = (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert "if b == 0:" in current_content
    backup_file = tmp_path / "calc.py.bak"
    assert backup_file.exists()
    assert "return a / b" in backup_file.read_text(encoding="utf-8")
