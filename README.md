# CodeLens AI: Autonomous Local Coding Agent

**CodeLens AI** is a real, autonomous AI pair-programming and debugging agent built in Python and powered locally by Ollama using Google's **Gemma 4** (`gemma4:e2b`). It operates directly on your local Windows workstation through a command-line interface, enforcing strict workspace containment, process sandboxing, and interactive approval guards.

Unlike passive chat interfaces or static dashboards, CodeLens AI executes a disciplined autonomous engineering loop: formulating plans, selecting and invoking tools via a validated JSON schema protocol, analyzing real execution observations, synthesizing minimal patches with unified diff previews, and verifying results using isolated test suites.

---

## Key Capabilities

- **100% Local & Private**: Connects exclusively to local Ollama at `http://127.0.0.1:11434` running `gemma4:e2b`. Never transmits your code to third-party cloud AI providers.
- **Disciplined Autonomous Loop**: Plans, observes, hypotheses, patches, and verifies across bounded iterations (configurable `max_iterations`, default 8).
- **Verified Observations vs. Assumptions**: Distinguishes facts from tool execution observations from model assumptions. Never claims a test passed without real runner exit codes and test report assertions.
- **Robust Sandboxing on Windows**: Untrusted code and tests execute in isolated subprocesses with sanitized environments (stripping API keys and secrets), strict timeouts (default 8s), and process tree termination.
- **Safety First & Safe Patching**: Generates unified diffs, prompts for explicit user approval before modifying files, generates automatic `.bak` backups, and supports one-click rollback.
- **Student Learning Mode**: 3-stage educational mode (Hint → Guidance → Solution) allowing student developers to learn debugging at their own pace without premature spoilers.

---

## Architecture

```
+------------------------------------------------------------------------------------+
|                                    USER CLI                                        |
|         (codelens run / fix / learn / check / tools / demo  |  .\run.ps1)          |
+------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|                         CODELENS AGENT ORCHESTRATOR                                |
|  - AgentState & History Tracking (Distinguishes Fact vs. Model Assumption)         |
|  - Bounded Multi-Step Iteration Control (Max Iterations, Approval Guardrails)      |
|  - Initial Diagnostic Planner & Reflection Engine                                  |
+------------------------------------------------------------------------------------+
       |                                                               |
       | Prompt + History (JSON Schema)                                | Exec Validated Tools
       v                                                               v
+--------------------------------+           +---------------------------------------+
|        LOCAL OLLAMA LLM        |           |          MODULAR TOOL REGISTRY        |
|  - Endpoint: 127.0.0.1:11434   |           |  - inspect_file (Workspace read)      |
|  - Model: gemma4:e2b           |           |  - list_files (Workspace discovery)   |
|  - Structured Tool Protocol    |           |  - search_workspace (Code search)     |
|  - JSON Extraction & Retry     |           |  - check_syntax (AST static parsing)  |
+--------------------------------+           |  - analyze_error (Traceback parser)   |
       |                                     |  - generate_tests (Unit test author)  |
       | Validated ToolCall(name, args)      |  - create_patch (Unified diff engine) |
       +-----------------------------------> |  - apply_patch (Safe patch + .bak)    |
                                             |  - rollback_patch (Backup restore)    |
                                             |  - finish (Root cause summary)        |
                                             +---------------------------------------+
                                                                 |
                                                                 v
                                             +---------------------------------------+
                                             |         ISOLATED SANDBOX RUNNER       |
                                             |  - run_code_sandbox (Subprocess exec) |
                                             |  - run_tests (unittest / pytest)      |
                                             |  - Sanitized Env (No leaked secrets)  |
                                             |  - Wall-Clock Timeout & Termination   |
                                             +---------------------------------------+
```

---

## Directory Structure

```
CodeLensAI/
├── codelens/                    # Core Python package
│   ├── __init__.py
│   ├── config.py                # Environment configuration & Pydantic settings
│   ├── cli.py                   # Click + Rich command-line interface
│   ├── orchestrator/            # Autonomous agent orchestration
│   │   ├── __init__.py
│   │   ├── agent.py             # 10-step autonomous agent loop
│   │   ├── state.py             # Agent memory, history, & observation model
│   │   ├── planner.py           # Diagnostic planner
│   │   └── learning.py          # 3-stage student learning tutor
│   ├── llm/                     # Ollama client & structured tool protocol
│   │   ├── __init__.py
│   │   ├── ollama_client.py     # Local HTTP client & health checks
│   │   └── protocol.py          # JSON schema validation & retry repair
│   ├── tools/                   # Modular agent tools
│   │   ├── __init__.py
│   │   ├── base.py              # BaseTool & ToolResult abstractions
│   │   ├── file_tools.py        # Workspace file inspection & search
│   │   ├── syntax_tools.py      # AST static syntax & language detection
│   │   ├── error_tools.py       # Traceback & exception categorization
│   │   ├── test_gen_tools.py    # Test generation tool
│   │   ├── patch_tools.py       # Unified diffs, patching, & rollback
│   │   ├── sandbox.py           # Subprocess execution & test runner
│   │   └── registry.py          # Tool discovery & execution dispatcher
│   └── utils/                   # Security, logging, & guardrails
│       ├── __init__.py
│       ├── security.py          # Path validation & environment sanitization
│       └── logger.py            # Rich terminal tables, panels, & diffs
├── examples/                    # Sample buggy programs & tests for demo
│   ├── off_by_one.py            # Median calculation boundary bug
│   ├── zero_division.py         # Empty list division error
│   ├── type_mismatch.py         # Integer/string concatenation error
│   └── tests/
│       ├── test_off_by_one.py   # Test suite reproducing median bug
│       └── test_zero_division.py
├── tests/                       # Complete automated test suite
│   ├── unit/                    # Unit tests for security, tools, protocol
│   │   ├── test_security.py
│   │   ├── test_protocol.py
│   │   ├── test_tools.py
│   │   └── test_learning.py
│   └── integration/             # End-to-end integration tests
│       ├── test_agent_mock.py   # Deterministic mocked loop test
│       └── test_agent_live.py   # Live integration test against Ollama
├── .env.example                 # Environment configuration template
├── .gitignore                   # Standard git exclusions
├── pyproject.toml               # Modern build & pytest configuration
├── requirements.txt             # Frozen dependency manifest
├── setup.ps1                    # One-click Windows setup PowerShell script
├── run.ps1                      # Windows launcher PowerShell script
└── README.md                    # This document
```

---

## Getting Started on Windows

### 1. Prerequisites
- **Python 3.10+** installed and available in PATH.
- **Ollama** installed on Windows and running at `http://127.0.0.1:11434`.
- The **`gemma4:e2b`** model downloaded in Ollama:
  ```powershell
  ollama list
  # If not installed:
  ollama pull gemma4:e2b
  ```

### 2. Automated Setup (Recommended)
Run the provided setup script in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
This script will:
1. Verify Python 3.10+ availability.
2. Initialize the local virtual environment at `.venv`.
3. Install all required dependencies from `requirements.txt`.
4. Run a health check against Ollama and confirm that `gemma4:e2b` is ready.

### 3. Quick Health Check
Verify your local setup with:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 check
```
Sample output:
```
+--------------------------- CodeLens Health Check ---------------------------+
| Checking Ollama at http://127.0.0.1:11434...                                |
+-----------------------------------------------------------------------------+
[OK] Ollama server is responding at http://127.0.0.1:11434
  Verified Model: gemma4:e2b   
+-----------------------------+
| Property       | Value      |
|----------------+------------|
| Model Name     | gemma4:e2b |
| Family         | gemma4     |
| Parameter Size | 4.6B       |
| Quantization   | Q4_K_M     |
| Context Length | 131072     |
+-----------------------------+
```

---

## Usage Guide & Commands

### 1. Inspect Available Agent Tools
List all 11 registered agent tools and their input argument schemas:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 tools
```

### 2. Autonomous Agent Task Execution (`run`)
Instruct the agent to analyze a task, inspect files, formulate tests, create patches, and verify:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 run "Fix the bug in examples/off_by_one.py and verify with tests" --target examples/off_by_one.py
```
**Flags:**
- `--target, -t`: Focus agent on a specific file.
- `--workspace, -w`: Set explicit workspace directory (default: current directory).
- `--max-iter, -i`: Configure maximum iteration limit (default: 8).
- `--auto-approve`: Automatically approve patch application without prompt (default: prompts user).
- `--disable-sandbox`: Disable isolated execution if desired.
- `--verbose, -v`: Enable verbose debug logging.

### 3. Targeted Bug Repair (`fix`)
Quick command targeting a specific buggy file:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 fix examples/zero_division.py
```

### 4. Student Learning Mode (`learn`)
Step through debugging progressively with the 3-stage tutor without spoiling the answer:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 learn examples/off_by_one.py --problem "find_median() fails on odd-length lists"
```
**How it works:**
1. **Stage 1 (Hint)**: The agent provides a conceptual clue (e.g., zero-based indexing invariant).
2. The user is prompted: `Would you like guidance for the next step? [next/solve/quit]`.
3. **Stage 2 (Guidance)**: The agent points to the specific condition or variable to inspect.
4. The user is prompted: `Would you like to reveal the full solution? [solve/quit]`.
5. **Stage 3 (Solution)**: The agent reveals the complete root cause, corrected code, and explanation.

### 5. Launch the Web Interface (`web`)
Launch the modern full-stack web application with real-time SSE streaming, visual diff previews, and interactive approval modals:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 web
```
This launches the server at `http://127.0.0.1:8000` and automatically opens your browser.

### 6. Automated Demonstration (`demo`)
Run an end-to-end automated demonstration on a sample buggy program:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 demo
```

---

## Web Publishing & Cloud Deployment

CodeLens AI supports a **hybrid execution architecture**:
- **Locally on Windows**: Runs with local Ollama (`gemma4:e2b`) without requiring internet access or API fees.
- **Published on the Web**: Deploys as a lightweight FastAPI service that connects seamlessly to cloud LLM providers (e.g. **Google Gemini API**, **Groq**, or **OpenRouter**) via an API key.

### 1-Click Publishing to Render.com
1. Push your repository to GitHub.
2. In [Render.com](https://render.com), click **New +** > **Web Service** and link your repo.
3. Render automatically reads [`render.yaml`](file:///c:/Users/srsud/OneDrive/CodeLensAI/render.yaml):
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn codelens.web.app:app --host 0.0.0.0 --port $PORT`
4. Add your `GEMINI_API_KEY` or `GROQ_API_KEY` under Environment Variables in the Render dashboard.
5. Your application is live worldwide!

### Docker / Hugging Face Spaces Deployment
Build and run the portable container locally or on any cloud server:
```bash
docker build -t codelens-ai .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key" codelens-ai
```

---

## Safety & Sandboxing Architecture

CodeLens AI incorporates multiple layers of security to protect your workstation:

| Guardrail | Implementation |
|---|---|
| **Filesystem Isolation** | Path traversal prevention via `validate_safe_path()`. Strictly confines all file reads, writes, and searches to the configured workspace root. Rejects `../`, absolute paths outside workspace, and symlink escapes. |
| **Protected Secrets** | Automatically blocks reading or modification of `.env`, `id_rsa`, `*.pem`, `*.key`, and VCS internal directories. |
| **Credential Sanitization** | Sandboxed subprocesses run with stripped environments via `sanitize_sandbox_env()`, purging sensitive environment variables (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `AWS_*`, `GOOGLE_*`, etc.). |
| **Python Safe Path** | Sandboxed processes run with `PYTHONSAFEPATH=1`, `PYTHONDONTWRITEBYTECODE=1`, and explicit `PYTHONPATH` restricted strictly to the workspace. |
| **Execution Sandboxing** | Code and tests are never executed within the agent process. Subprocesses are executed without shell (`shell=False`) with strict timeouts (default 8s) and explicit tree termination upon timeout. |
| **User Approval & Backups** | File modifications require interactive confirmation showing a unified diff. Every patch automatically creates a timestamped `.bak` backup file, allowing immediate rollback via `rollback_patch`. |
| **Ground Truth Verification** | Distinguishes verified tool observations (`[VERIFIED TOOL OBSERVATION]`) from model hypotheses. Never claims tests passed without real runner exit code 0 and parsed assertion counts. |

---

## Testing

Run the full automated test suite using `pytest`:
```powershell
.venv\Scripts\pytest.exe -v
```

### Test Coverage Highlights:
- **`tests/unit/test_security.py`**: Verifies path traversal blocking, secret file protection, and environment variable sanitization.
- **`tests/unit/test_protocol.py`**: Verifies JSON extraction, code block fence stripping, schema validation, and retry diagnostic generation.
- **`tests/unit/test_tools.py`**: Comprehensive unit tests for all 11 tools (syntax check, error traceback parser, test generator, sandbox execution, timeout handling, patch creation, backup, and rollback).
- **`tests/unit/test_learning.py`**: Verifies 3-stage hint, guidance, and solution prompt constraints.
- **`tests/integration/test_agent_mock.py`**: Deterministic end-to-end 10-step agent loop verification covering problem analysis, test failure observation, patch creation, file patching with `.bak` preservation, sandbox test verification, and structured final reporting.
- **`tests/integration/test_agent_live.py`**: Live integration test communicating directly with the local Ollama server and `gemma4:e2b` model.

---

## Configuration via `.env`

You can customize agent behavior by copying `.env.example` to `.env`:
```ini
CODELENS_OLLAMA_URL=http://127.0.0.1:11434
CODELENS_MODEL=gemma4:e2b
CODELENS_WORKSPACE=.
CODELENS_MAX_ITERATIONS=8
CODELENS_LLM_TIMEOUT=90
CODELENS_SANDBOX_TIMEOUT=8
CODELENS_SANDBOX_MEMORY_MB=256
CODELENS_AUTO_APPROVE=0
CODELENS_SANDBOX_ENABLED=1
CODELENS_VERBOSE=0
```
