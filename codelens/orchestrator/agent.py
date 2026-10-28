"""
Autonomous agent loop and orchestration engine for CodeLens AI.
Executes multi-step reasoning, plan formulation, tool execution,
observation reading, patch verification, and final explanation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from codelens.config import CodeLensConfig
from codelens.llm.ollama_client import OllamaClient, OllamaClientError
from codelens.llm.cloud_client import CloudLLMClient
from codelens.llm.protocol import ToolProtocol, ToolProtocolError, ActionDecision
from codelens.tools.registry import ToolRegistry
from codelens.tools.base import ToolResult
from codelens.orchestrator.state import AgentState, AgentStatus, AgentStep, FinalReport
from codelens.orchestrator.planner import TaskPlanner
from codelens.utils.logger import AgentLogger


class CodeLensAgent:
    """The core autonomous engineering agent."""

    def __init__(
        self,
        config: Optional[CodeLensConfig] = None,
        registry: Optional[ToolRegistry] = None,
        client: Optional[Any] = None,
        logger: Optional[AgentLogger] = None,
    ):
        self.config = config or CodeLensConfig()
        self.workspace_root = self.config.validate_workspace()
        self.logger = logger or AgentLogger(verbose=self.config.verbose)
        from codelens.llm.factory import create_llm_client
        self.client = client or create_llm_client(self.config)
        self.registry = registry or ToolRegistry.create_default(
            workspace_root=self.workspace_root,
            sandbox_timeout=self.config.sandbox_timeout,
            sandbox_memory_mb=self.config.sandbox_memory_limit_mb,
            sandbox_enabled=self.config.sandbox_enabled,
            require_approval=self.config.require_approval,
        )

    def run(self, task_description: str, target_file: Optional[str] = None) -> AgentState:
        """
        Execute the autonomous agent loop for a given programming task or bug report.
        """
        # Determine active model based on client type
        active_model = (
            self.config.cloud_model
            if isinstance(self.client, CloudLLMClient)
            else self.config.model_name
        )
        provider_name = "Cloud LLM" if isinstance(self.client, CloudLLMClient) else "Ollama"

        self.logger.banner(
            title="CodeLens AI - Autonomous Engineering Agent",
            subtitle=f"{provider_name}: {active_model} | Workspace: {self.workspace_root}",
        )

        # 1. Pre-flight connectivity & model check
        self.logger.log_info(f"Performing pre-flight connectivity verification for {provider_name} ({active_model})...")
        try:
            self.client.check_connection()
            model_info = self.client.verify_model(active_model)
            self.logger.log_info(
                f"Verified model '{model_info.name}' (family: {model_info.family or 'local'}, "
                f"parameters: {model_info.parameter_size or 'unknown'})."
            )
        except Exception as e:
            self.logger.log_error(f"Pre-flight verification failed: {e}")
            state = AgentState(
                task=task_description,
                workspace_dir=str(self.workspace_root),
                status=AgentStatus.FAILED,
            )
            return state

        # 2. Formulate Initial Plan
        initial_plan = TaskPlanner.create_initial_plan(task_description, target_file)
        self.logger.log_thought(
            iteration=0,
            max_iter=self.config.max_iterations,
            thought=f"Received task: '{task_description}'. Initializing workspace analysis.",
            plan=initial_plan,
        )

        state = AgentState(
            task=task_description,
            workspace_dir=str(self.workspace_root),
            max_iterations=self.config.max_iterations,
            status=AgentStatus.RUNNING,
        )

        # Build system prompt with tools documentation
        tools_doc = self.registry.get_tools_prompt_description()
        system_prompt = (
            f"{ToolProtocol.SYSTEM_INSTRUCTION}\n\n"
            f"WORKSPACE DIRECTORY: {self.workspace_root}\n"
            f"SANDBOX STATUS: {'ENABLED' if self.config.sandbox_enabled else 'DISABLED'}\n"
            f"APPROVAL REQUIRED BEFORE PATCHING: {'YES' if self.config.require_approval else 'NO (AUTO-APPROVED)'}\n\n"
            f"{tools_doc}\n"
        )

        # Initial user prompt
        initial_user_prompt = (
            f"TASK DESCRIPTION:\n{task_description}\n"
        )
        if target_file:
            initial_user_prompt += f"TARGET FILE: {target_file}\n"
        initial_user_prompt += (
            f"\nINITIAL ENGINEERING PLAN:\n{initial_plan}\n\n"
            "Begin by inspecting the workspace or checking the target file. "
            "Remember to respond ONLY with a single JSON object."
        )

        # Main Agent Loop
        retry_count = 0
        max_retries_per_step = 3

        while state.iteration < state.max_iterations:
            state.iteration += 1
            iter_num = state.iteration

            # Construct message context
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": initial_user_prompt})

            # Append historical turns
            messages.extend(state.get_conversation_history(max_recent=6))

            # Query LLM (Ollama or Cloud)
            decision: Optional[ActionDecision] = None
            for attempt in range(max_retries_per_step):
                raw_resp = ""
                try:
                    raw_resp = self.client.chat(
                        messages=messages,
                        model=active_model,
                        format_json=True,
                        temperature=0.1,
                    )
                    decision = ToolProtocol.parse_response(
                        raw_response=raw_resp,
                        valid_tools=self.registry.list_names(),
                        tool_schemas=self.registry.get_schemas(),
                    )
                    break  # Successfully parsed and validated
                except (ToolProtocolError, OllamaClientError) as err:
                    self.logger.log_warning(
                        f"Protocol validation error on attempt {attempt + 1}/{max_retries_per_step}: {err}"
                    )
                    # Add corrective message to prompt for next attempt
                    if raw_resp:
                        messages.append({
                            "role": "assistant",
                            "content": raw_resp,
                        })
                    messages.append({
                        "role": "user",
                        "content": ToolProtocol.build_format_reminder(str(err)),
                    })

            if not decision:
                self.logger.log_error(
                    f"Failed to obtain valid tool decision from model after {max_retries_per_step} attempts."
                )
                state.status = AgentStatus.FAILED
                break

            # Log model thought and planned action
            self.logger.log_thought(
                iteration=iter_num,
                max_iter=state.max_iterations,
                thought=decision.thought,
                plan=decision.plan,
            )

            # Check if agent invoked 'finish'
            if decision.tool_name == "finish":
                root_cause = str(decision.tool_args.get("root_cause", "Unspecified root cause."))
                solution_summary = str(decision.tool_args.get("solution_summary", "Fix applied."))
                test_summary = str(decision.tool_args.get("test_results_summary", ""))
                uncertainty = str(decision.tool_args.get("remaining_uncertainty", "None noted."))

                # Safety check: Did tests actually run and pass?
                verified_pass = state.has_verified_test_pass()
                if not verified_pass:
                    if not test_summary or "passed" in test_summary.lower():
                        test_summary = (
                            "[WARNING: NO VERIFIED TEST PASSED IN SANDBOX] "
                            + (test_summary or "No tests were executed.")
                        )
                    uncertainty += " [NOTE: Fix was not backed by a passing sandbox test suite.]"

                report = FinalReport(
                    root_cause=root_cause,
                    solution_summary=solution_summary,
                    test_results_summary=test_summary,
                    remaining_uncertainty=uncertainty,
                )
                state.final_report = report
                state.status = AgentStatus.SOLVED

                # Record finish step
                state.add_step(
                    iteration=iter_num,
                    thought=decision.thought,
                    plan=decision.plan,
                    tool_name="finish",
                    tool_args=decision.tool_args,
                    observation=ToolResult(
                        success=True,
                        output="Task concluded by agent.",
                        data=report.model_dump(),
                        is_assumption=False,
                    ),
                )

                # Render final summary
                self.logger.log_solution(
                    root_cause=root_cause,
                    explanation=solution_summary,
                    test_summary=test_summary,
                    uncertainty=uncertainty,
                )
                break

            # Execute the selected tool
            self.logger.log_tool_call(tool_name=decision.tool_name, args=decision.tool_args)

            obs = self.registry.execute(name=decision.tool_name, args=decision.tool_args)

            # Log observation clearly distinguishing facts from assumptions
            self.logger.log_tool_observation(
                tool_name=decision.tool_name,
                success=obs.success,
                output=obs.output or (obs.error or ""),
                is_assumption=obs.is_assumption,
            )

            # If tool was create_patch, display diff specifically
            if decision.tool_name == "create_patch" and obs.data and obs.data.get("diff"):
                self.logger.log_diff(
                    filename=decision.tool_args.get("target_path", "unknown"),
                    diff_text=obs.data["diff"],
                )

            # Record step in state
            state.add_step(
                iteration=iter_num,
                thought=decision.thought,
                plan=decision.plan,
                tool_name=decision.tool_name,
                tool_args=decision.tool_args,
                observation=obs,
            )

        # Check if stopped due to iteration limit
        if state.iteration >= state.max_iterations and state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.STOPPED_LIMIT
            self.logger.log_warning(
                f"Agent reached maximum iteration limit ({state.max_iterations}). Stopping execution."
            )
            state.final_report = FinalReport(
                root_cause="Task incomplete: reached maximum permitted iterations.",
                solution_summary="Partial analysis conducted. Check step history for progress.",
                test_results_summary=f"Completed {len(state.test_runs)} test runs.",
                remaining_uncertainty="Unresolved due to iteration boundary. Review logs before manual intervention.",
            )
            self.logger.log_solution(
                root_cause=state.final_report.root_cause,
                explanation=state.final_report.solution_summary,
                test_summary=state.final_report.test_results_summary,
                uncertainty=state.final_report.remaining_uncertainty,
            )

        return state
