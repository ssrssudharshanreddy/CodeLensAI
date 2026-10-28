"""
FastAPI application for CodeLens AI web interface.
Provides real-time Server-Sent Events (SSE) streaming of agent thought processes,
interactive patch approvals, file browsing, and student learning modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codelens.config import CodeLensConfig
from codelens.llm.factory import create_llm_client
from codelens.orchestrator.agent import CodeLensAgent
from codelens.orchestrator.learning import LearningSession
from codelens.orchestrator.state import AgentStatus
from codelens.tools.registry import ToolRegistry
from codelens.utils.logger import AgentLogger


app = FastAPI(title="CodeLens AI Web Interface", version="0.1.0")

# In-memory run tracking and approval coordination
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}
APPROVAL_EVENTS: Dict[str, threading.Event] = {}
APPROVAL_DECISIONS: Dict[str, bool] = {}


class WebAgentLogger(AgentLogger):
    """Custom logger that intercepts agent events and pushes them to the SSE queue."""

    def __init__(self, run_id: str, event_queue: queue.Queue):
        super().__init__(verbose=True)
        self.run_id = run_id
        self.queue = event_queue

    def log_thought(self, iteration: int, max_iter: int, thought: str, plan: Optional[str] = None):
        super().log_thought(iteration, max_iter, thought, plan)
        self.queue.put({
            "type": "thought",
            "iteration": iteration,
            "max_iter": max_iter,
            "thought": thought,
            "plan": plan or "",
        })

    def log_tool_call(self, tool_name: str, args: dict[str, Any]):
        super().log_tool_call(tool_name, args)
        self.queue.put({
            "type": "tool_call",
            "tool_name": tool_name,
            "args": args,
        })

    def log_tool_observation(self, tool_name: str, success: bool, output: str, is_assumption: bool = False):
        super().log_tool_observation(tool_name, success, output, is_assumption)
        self.queue.put({
            "type": "observation",
            "tool_name": tool_name,
            "success": success,
            "output": output,
            "is_assumption": is_assumption,
        })

    def log_diff(self, filename: str, diff_text: str):
        super().log_diff(filename, diff_text)
        self.queue.put({
            "type": "diff",
            "filename": filename,
            "diff": diff_text,
        })

    def log_solution(self, root_cause: str, explanation: str, test_summary: str, uncertainty: str):
        super().log_solution(root_cause, explanation, test_summary, uncertainty)
        self.queue.put({
            "type": "solution",
            "root_cause": root_cause,
            "explanation": explanation,
            "test_summary": test_summary,
            "uncertainty": uncertainty,
        })

    def log_warning(self, message: str):
        super().log_warning(message)
        self.queue.put({"type": "warning", "message": message})

    def log_error(self, message: str):
        super().log_error(message)
        self.queue.put({"type": "error", "message": message})

    def log_info(self, message: str):
        super().log_info(message)
        self.queue.put({"type": "info", "message": message})


class RunTaskRequest(BaseModel):
    task: str
    target_file: Optional[str] = None
    max_iterations: int = 8
    auto_approve: bool = False
    provider: Optional[str] = None
    model_name: Optional[str] = None


class ApprovalRequest(BaseModel):
    run_id: str
    approved: bool


class SettingsUpdateRequest(BaseModel):
    provider: str = Field(description="'ollama' or 'cloud' or 'auto'")
    cloud_api_key: Optional[str] = None
    cloud_base_url: Optional[str] = None
    cloud_model: Optional[str] = None
    local_model: Optional[str] = None


class LearnRequest(BaseModel):
    code: str
    problem: str
    stage: str = Field(description="'hint', 'guidance', or 'solution'")
    prior_hint: Optional[str] = None


# Static directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>CodeLens AI UI loading...</h1>", status_code=200)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    cfg = CodeLensConfig()
    client = create_llm_client(cfg)
    is_connected = False
    model_details = {}

    from codelens.llm.cloud_client import CloudLLMClient
    active_provider = "cloud" if isinstance(client, CloudLLMClient) else "ollama"
    active_model = cfg.cloud_model if isinstance(client, CloudLLMClient) else cfg.model_name
    if isinstance(client, CloudLLMClient) and client.is_gemini(active_model):
        if not CloudLLMClient.is_text_generation_model(active_model):
            active_model = "gemini-2.0-flash"

    try:
        is_connected = client.check_connection()
        info = client.verify_model(active_model)
        model_details = info.model_dump()
    except Exception as e:
        model_details = {"error": str(e)}

    return {
        "status": "online",
        "provider": active_provider if cfg.provider == "auto" else cfg.provider,
        "is_connected": is_connected,
        "model_details": model_details,
        "workspace": str(cfg.workspace_dir),
        "sandbox_enabled": cfg.sandbox_enabled,
        "active_runs_count": len([r for r in ACTIVE_RUNS.values() if r.get("status") == "running"]),
    }


@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest):
    os.environ["CODELENS_PROVIDER"] = req.provider
    if req.cloud_api_key is not None:
        os.environ["CODELENS_CLOUD_API_KEY"] = req.cloud_api_key
        os.environ["GEMINI_API_KEY"] = req.cloud_api_key
    if req.cloud_base_url is not None:
        os.environ["CODELENS_CLOUD_BASE_URL"] = req.cloud_base_url
    if req.cloud_model is not None:
        os.environ["CODELENS_CLOUD_MODEL"] = req.cloud_model
    if req.local_model is not None:
        os.environ["CODELENS_MODEL"] = req.local_model

    return {"status": "updated", "provider": req.provider}


@app.get("/api/files")
async def list_workspace_files():
    cfg = CodeLensConfig()
    ws = cfg.workspace_dir
    reg = ToolRegistry.create_default(workspace_root=ws)
    list_tool = reg.get("list_files")
    res = list_tool.execute() if list_tool else None
    files = res.data.get("files", []) if res and res.data else []
    return {"workspace": str(ws), "files": files}


@app.get("/api/file")
async def read_file(path: str):
    cfg = CodeLensConfig()
    ws = cfg.workspace_dir
    reg = ToolRegistry.create_default(workspace_root=ws)
    inspect_tool = reg.get("inspect_file")
    if not inspect_tool:
        raise HTTPException(status_code=500, detail="InspectFileTool not registered")
    res = inspect_tool.execute(path=path)
    if not res.success:
        raise HTTPException(status_code=404, detail=res.error or "File could not be inspected")
    return {"path": path, "content": res.data.get("raw_content", res.output)}


def run_agent_in_thread(run_id: str, task: str, target_file: Optional[str], config: CodeLensConfig, event_queue: queue.Queue):
    """Worker executing agent in background thread and pushing events."""
    logger = WebAgentLogger(run_id=run_id, event_queue=event_queue)

    # Custom approval hook for web
    def web_approval_hook(action_summary: str, details: Optional[str] = None, auto_approve: bool = False) -> bool:
        if auto_approve:
            return True
        event_queue.put({
            "type": "approval_required",
            "run_id": run_id,
            "action": action_summary,
            "details": details or "",
        })
        # Wait on event
        event = threading.Event()
        APPROVAL_EVENTS[run_id] = event
        event.wait(timeout=120)  # Wait up to 2 minutes for user click
        decision = APPROVAL_DECISIONS.get(run_id, False)
        return decision

    # Patch security approval function dynamically for this run
    import codelens.utils.security as sec
    orig_approval = sec.request_user_approval
    sec.request_user_approval = web_approval_hook

    try:
        agent = CodeLensAgent(config=config, logger=logger)
        state = agent.run(task_description=task, target_file=target_file)
        ACTIVE_RUNS[run_id]["status"] = state.status.value
        event_queue.put({"type": "done", "status": state.status.value, "steps_count": len(state.steps)})
    except Exception as e:
        ACTIVE_RUNS[run_id]["status"] = "FAILED"
        event_queue.put({"type": "error", "message": f"Execution failed: {e}"})
        event_queue.put({"type": "done", "status": "FAILED", "steps_count": 0})
    finally:
        sec.request_user_approval = orig_approval


@app.post("/api/run")
async def start_run(req: RunTaskRequest):
    run_id = str(uuid.uuid4())[:8]
    event_queue = queue.Queue()

    cfg = CodeLensConfig(
        max_iterations=req.max_iterations,
        require_approval=not req.auto_approve,
        provider=req.provider or os.getenv("CODELENS_PROVIDER", "auto"),
    )
    if req.model_name:
        if cfg.provider == "cloud":
            cfg.cloud_model = req.model_name
        else:
            cfg.model_name = req.model_name

    ACTIVE_RUNS[run_id] = {
        "id": run_id,
        "task": req.task,
        "status": "running",
        "queue": event_queue,
    }

    thread = threading.Thread(
        target=run_agent_in_thread,
        args=(run_id, req.task, req.target_file, cfg, event_queue),
        daemon=True,
    )
    thread.start()

    return {"run_id": run_id, "status": "started"}


@app.post("/api/approve")
async def submit_approval(req: ApprovalRequest):
    run_id = req.run_id
    if run_id in APPROVAL_EVENTS:
        APPROVAL_DECISIONS[run_id] = req.approved
        APPROVAL_EVENTS[run_id].set()
        return {"status": "ok", "decision": req.approved}
    raise HTTPException(status_code=404, detail="No pending approval for this run ID")


@app.get("/api/stream/{run_id}")
async def stream_run_events(run_id: str):
    if run_id not in ACTIVE_RUNS:
        raise HTTPException(status_code=404, detail="Run ID not found")

    event_queue: queue.Queue = ACTIVE_RUNS[run_id]["queue"]

    async def event_generator():
        # Stream initial ping
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"
        while True:
            try:
                # Fetch next event without blocking async loop
                event = await asyncio.to_thread(event_queue.get, timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except queue.Empty:
                # Send periodic keep-alive
                yield f": keep-alive\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/learn")
async def learn_stage(req: LearnRequest):
    cfg = CodeLensConfig()
    client = create_llm_client(cfg)
    from codelens.llm.cloud_client import CloudLLMClient
    active_model = cfg.cloud_model if isinstance(client, CloudLLMClient) else cfg.model_name
    session = LearningSession(client=client, model=active_model)

    try:
        if req.stage == "hint":
            text = session.generate_hint(code=req.code, problem_description=req.problem)
        elif req.stage == "guidance":
            text = session.generate_guidance(
                code=req.code,
                problem_description=req.problem,
                hint=req.prior_hint or "Examine the boundary conditions.",
            )
        elif req.stage == "solution":
            text = session.generate_solution(code=req.code, problem_description=req.problem)
        else:
            raise HTTPException(status_code=400, detail="Invalid stage. Must be 'hint', 'guidance', or 'solution'.")

        return {"stage": req.stage, "content": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/demo")
async def trigger_demo():
    req = RunTaskRequest(
        task="Inspect examples/off_by_one.py, identify the bug in median calculation, apply a patch, and verify with tests.",
        target_file="examples/off_by_one.py",
        auto_approve=True,
    )
    return await start_run(req)
