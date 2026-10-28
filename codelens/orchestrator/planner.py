"""
Task analysis and diagnostic planning for CodeLens AI.
Synthesizes the initial problem statement, file context, and execution strategy.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path


class TaskPlanner:
    """Generates structured initial plans before launching the agent loop."""

    @classmethod
    def create_initial_plan(cls, task_description: str, target_file: Optional[str] = None) -> str:
        """Formulate a concise 7-step engineering plan."""
        target_clause = f" for '{target_file}'" if target_file else ""
        return (
            f"DIAGNOSTIC PLAN{target_clause}:\n"
            f"1. Discover & Inspect: Examine workspace structure and inspect source code.\n"
            f"2. Static Analysis: Run AST syntax verification and language checks.\n"
            f"3. Failure Reproduction: Execute code or existing tests in isolated sandbox to collect real output.\n"
            f"4. Error Diagnosis: Analyze tracebacks, line numbers, and classify root cause.\n"
            f"5. Test & Patch Generation: Author an isolated test case and create a unified diff patch.\n"
            f"6. Verification: Apply patch with backup and rerun tests in sandbox to verify resolution.\n"
            f"7. Report & Explanation: Detail root cause, fix rationale, real test evidence, and edge cases."
        )
