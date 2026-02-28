# Agent-Force

Agent-Force is a local safety-evaluation platform for AI agents. It runs scenario-based safety checks against tool-using agents, produces structured judge-based scoring, and stores full run artifacts for audit and debugging.

Current focus is on **agent safety outcomes** (tool misuse, policy violations, guardrails), not formal governance reporting.

## Features

- **Multi-page dashboard**: Overview, Evaluation, Results, Remediation.
- **Standard evaluation** for built-in agents across scenario sets.
- **Adaptive evaluation** that generates additional adversarial scenarios for weak areas.
- **Safety scoring** using a separate LLM judge model.
- **Sandboxed execution** for world simulations (email/web search/code execution).
- **Attack campaign mode** with deterministic scenario generation and turn-level scoring.
- **MCP support** for real server-backed agents (e.g., Jira MCP).
- **MCP registry link support** on API runs (runtime-resolved manifests).
- **Live progress stream** via SSE while runs execute.
- **Persisted artifacts** in JSON with secret redaction.

## Current Agents and Scenario Sets

The API registry currently supports these built-in agent keys:

- `email` -> `email_safety_scenarios`
- `web_search` -> `web_search_safety_scenarios`
- `code_exec` -> `code_exec_safety_scenarios`
- `jira` -> `jira_safety_scenarios` (requires Jira MCP configuration)

Custom HTTP-only targets are available in attack mode via `target_agent.type: "http"`.

## Requirements

- Python 3.11+
- API key for your provider (or Ollama)
- `mcp` package for MCP agents (optional, but required for real MCP server usage)

## Installation

```bash
# from project root
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
# API server + attack endpoints also require:
pip install fastapi "uvicorn[standard]" httpx
```

## Environment Configuration

Create `.env` in repo root.

```bash
# Provider keys
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...

# Default models
AGENT_MODEL=openai/gpt-4o-mini
SCORER_MODEL=openai/gpt-4o-mini
ADAPTIVE_MODEL=openai/gpt-4o-mini

# Optional run storage overrides
AGENTFORCE_RUNS_FILE=artifacts/runs.json
AGENTFORCE_RUNS_DIR=artifacts/run_logs

# Optional Jira MCP setup (for `jira` agent)
# Preferred:
JIRA_MCP_ARGS_JSON=["mcp-atlassian","--jira-url","https://your-org.atlassian.net","--jira-username","you@example.com","--jira-token","your-token"]
# Optional compatibility helpers
JIRA_MCP_COMMAND=uvx
JIRA_URL=https://your-org.atlassian.net
JIRA_USERNAME=you@example.com
JIRA_API_TOKEN=your-token

# Safety hardening for Jira (optional)
JIRA_EVAL_READ_ONLY=true
```

For a full list of environment variables, check `.env.example`.

## Running Locally

### 1) Start the API server

```bash
# Recommended:
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000

# Or on POSIX shell using wrapper:
./run_server.sh
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2) Open the UI

```bash
streamlit run app.py
```

The UI uses the API at `http://localhost:8000` by default (`config.py`).

### 3) Run CLI examples

```bash
# Run all standard agents
python examples/example.py

# Run specific built-in agents
python examples/example.py email web_search

# Run adaptive eval
python examples/adaptive_example.py

# On shell wrappers:
./run_eval.sh
./run_adaptive.sh
```

## API Reference

### POST `/runs`

Start a safety run.

Body schema (example):

```json
{
  "agents": ["email", "jira"],
  "adaptive": false,
  "agent_model": "openai/gpt-4o-mini",
  "scorer_model": "openai/gpt-4o-mini",
  "adaptive_model": "openai/gpt-4o-mini",
  "samples_per_round": 4,
  "max_rounds": 3,
  "sandbox_mode": "world_stateful",
  "world_pack": "acme_corp_v1",
  "demo_mode": "live_hybrid",
  "trace_level": "full",
  "mcp_registry_links": []
}
```

Notes:
- `mcp_registry_links` accepts a list of MCP registry URLs.
- If `adaptive: true`, `agent_model` drives execution, `adaptive_model` drives scenario generation.

### GET `/runs`

List completed and in-progress run summaries.

### GET `/runs/{run_id}`

Get full run payload (scorecards, rule hits, metadata, config summary).

### GET `/runs/{run_id}/stream`

SSE stream of live execution events:

```bash
curl -N http://localhost:8000/runs/<run_id>/stream
```

### POST `/attack/runs`

Start defensive campaign (sandbox mode by default).

```json
{
  "target_agent": {
    "type": "world_sandbox",
    "sandbox_agent": "email",
    "world_pack": "acme_corp_v1",
    "demo_mode": "deterministic",
    "trace_level": "full",
    "mcp_registry_links": []
  },
  "agent_card": {
    "use_case": "Personal email assistant",
    "tools": ["email.read", "email.send"],
    "tool_specs": [{"name": "email.send", "description": "Send email", "side_effect": true}],
    "memory_mode": "session",
    "autonomy_level": "act_with_confirm",
    "operation_mode": "shadow",
    "approval_token": "APPROVE_ACTION",
    "bulk_action_threshold": 250
  },
  "policies": [
    "Never reveal secrets",
    "Never call tools outside allowlist"
  ],
  "categories": ["context_compaction_failure", "confirmation_gate_failure", "stop_failsafe_failure"],
  "scenario_pack": "resilience_stress",
  "require_sandbox": true,
  "max_turns": 8,
  "budget": {"max_tests": 6, "max_tokens": 8000}
}
```

### POST `/attack/scenarios/generate`

Generate scenario previews without executing an attack run.

### HTTP target mode for attack campaigns

For your own custom HTTP target agent:

```json
"target_agent": {
  "type": "http",
  "endpoint": "https://your-agent.example.com/invoke",
  "auth": "Bearer <token>"
}
```

In this mode set `require_sandbox=false` when calling `/attack/runs`.

## Provider and Model Strings

Models use provider prefixes in `provider/model` format.

Supported providers: `openai`, `groq`, `gemini`, `ollama`.

Examples:
- `openai/gpt-5.2`
- `openai/gpt-5.2-pro`
- `openai/gpt-5-mini`
- `openai/gpt-5-nano`
- `openai/gpt-4.1`
- `openai/gpt-4.1-mini`
- `openai/gpt-4.1-nano`
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `groq/llama-3.1-8b-instant`
- `ollama/llama3.2`

The exact model availability depends on your provider account.

## Sandbox & Tooling Behavior

- `sandbox_mode` can be set to `world_stateful` (default in server) or `none`.
- `world_pack` selects synthetic fixtures (`acme_corp_v1` default).
- `demo_mode` controls fallback behavior:
  - `live_hybrid`: real agent execution with deterministic fallback on failure.
  - `deterministic`: no live LLM execution.
- `trace_level`: `summary` or `full`.

You should expect:
- tool calls generated by agents
- rule hits and confirmations in run metadata
- fallback indicators if execution switched from live to deterministic mode

## Output and Logging

### Global run index

- `artifacts/runs.json`: list and summary for all runs.

### Per-run archive

- `artifacts/run_logs/run_<run_id>.json`: full run payload after completion (results + events).

### Deterministic CLI reports

- `artifacts/reports/` for example CLI runs.

## Secret Safety in Logs

The run store redacts tokens and sensitive-looking arguments when persisting:

- token-like strings (OpenAI/ATATT/etc.)
- API keys in argument lists
- keys containing sensitive names (`token`, `secret`, `password`, etc.)

If you pass credentials anywhere, prefer environment variables over JSON body fields.

## Notes

- The UI includes framework checkboxes and labels; scenario scoring is derived from configured scenarios and judge/sandbox outputs.
- Attack endpoints and MCP capabilities are independent from standard mode. If you are only evaluating standard runs, keep configurations focused on `AGENT_MODEL`, `SCORER_MODEL`, and `mcp_registry_links`.

## Project Layout

- `app.py` � Streamlit dashboard entry point
- `api/server.py` � FastAPI service and run orchestration
- `api/store.py` � Run persistence + redaction + SSE event log
- `agents/` � Built-in agent builders (email, web_search, code_exec, jira)
- `safety_kit/` � Core runtime, providers, adaptive engine, scoring, sandbox, attack kit
- `sandbox_env/` � Scenario world, deterministic fallback runner, MCP manifest resolver
- `components/` � Streamlit UI components
- `examples/` � CLI entry examples
- `scripts/` � Wrapper shell scripts
- `artifacts/` � Local outputs and run archives

## Runbook

- Start backend first, verify `/health`.
- Start UI and run standard or adaptive tests.
- For attack coverage, open the Evaluation page Attack tab and generate/run scenarios.
- Inspect latest run in UI Results page, then open `artifacts/run_logs/run_<id>.json` for raw traces.

