"""
Agent state tracking, memory history, and observation classification.
"""

from __future__ import annotations

import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from codelens.tools.base import ToolResult
from codelens.tools.sandbox import TestRunResult


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    SOLVED = "SOLVED"
    FAILED = "FAILED"
    STOPPED_LIMIT = "STOPPED_LIMIT"


class AgentStep(BaseModel):
    """A single cycle in the agent loop: thought -> tool call -> verified observation."""
    iteration: int
    thought: str
    plan: str = ""
    tool_name: str
    tool_args: Dict[str, Any]
    observation: ToolResult
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class FinalReport(BaseModel):
    """The structured solution generated upon task completion."""
    root_cause: str
    solution_summary: str
    test_results_summary: str
    remaining_uncertainty: str


class AgentState(BaseModel):
    """Full execution state of the CodeLens AI agent."""
    task: str
    workspace_dir: str
    iteration: int = 0
    max_iterations: int = 8
    status: AgentStatus = AgentStatus.IDLE
    steps: List[AgentStep] = Field(default_factory=list)
    final_report: Optional[FinalReport] = None
    applied_patches: List[str] = Field(default_factory=list)
    test_runs: List[Dict[str, Any]] = Field(default_factory=list)

    def add_step(
        self,
        iteration: int,
        thought: str,
        plan: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        observation: ToolResult,
    ) -> AgentStep:
        step = AgentStep(
            iteration=iteration,
            thought=thought,
            plan=plan,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
        )
        self.steps.append(step)

        # Track applied patches
        if tool_name == "apply_patch" and observation.success:
            target = tool_args.get("target_path", "")
            if target and target not in self.applied_patches:
                self.applied_patches.append(target)

        # Track test executions
        if tool_name == "run_tests" and observation.data:
            self.test_runs.append(observation.data)

        return step

    def has_verified_test_pass(self) -> bool:
        """Check if at least one verified test run completed with all tests passing."""
        for run in self.test_runs:
            if run.get("all_passed") is True:
                return True
        return False

    def get_conversation_history(self, max_recent: int = 6) -> List[Dict[str, str]]:
        """
        Format recent steps into LLM message history,
        clearly labeling verified tool observations vs model thoughts.
        """
        messages = []
        recent_steps = self.steps[-max_recent:] if len(self.steps) > max_recent else self.steps

        for s in recent_steps:
            # Assistant decision
            assistant_payload = {
                "thought": s.thought,
                "plan": s.plan,
                "tool_name": s.tool_name,
                "tool_args": s.tool_args,
            }
            messages.append({
                "role": "assistant",
                "content": __import__("json").dumps(assistant_payload),
            })

            # Tool observation returned as user/environment observation
            obs_text = s.observation.to_observation_text()
            messages.append({
                "role": "user",
                "content": f"[TOOL OBSERVATION: {s.tool_name}]\n{obs_text}",
            })

        return messages
