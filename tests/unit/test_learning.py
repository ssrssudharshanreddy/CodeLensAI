"""
Unit tests for student learning mode.
"""

from unittest.mock import MagicMock
from codelens.orchestrator.learning import LearningSession
from codelens.llm.ollama_client import OllamaClient


def test_learning_session_hint_guidance_solution():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.default_model = "gemma4:e2b"
    mock_client.chat.side_effect = [
        "Think about what index represents the middle item in a 0-indexed list.",  # Hint
        "Inspect line 12. Notice that mid + 1 skips past the median element.",    # Guidance
        "The fix is to change mid + 1 to mid in the odd condition.",              # Solution
    ]

    session = LearningSession(client=mock_client)

    # 1. Hint
    hint = session.generate_hint("code", "problem")
    assert "0-indexed" in hint
    # Verify chat was called with prompt demanding no solution
    hint_call = mock_client.chat.call_args_list[0]
    assert "conceptual clue or hint" in hint_call.kwargs["messages"][0]["content"]

    # 2. Guidance
    guidance = session.generate_guidance("code", "problem", hint)
    assert "line 12" in guidance
    guidance_call = mock_client.chat.call_args_list[1]
    assert "targeted next step" in guidance_call.kwargs["messages"][0]["content"]

    # 3. Solution
    solution = session.generate_solution("code", "problem")
    assert "mid + 1" in solution
    solution_call = mock_client.chat.call_args_list[2]
    assert "complete solution" in solution_call.kwargs["messages"][0]["content"]
