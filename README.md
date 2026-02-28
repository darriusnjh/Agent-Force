# Agent-Force

A lightweight **safety evaluation framework** for personal AI agents. Run scenario-based evals, score agent behavior with an LLM judge, and get scorecards with actionable recommendations—all in a plug-and-play format.

Use it to answer: _"Would my email agent send malicious emails?"_, _"Does my assistant refuse harmful requests?"_, and similar safety questions before deploying agents in the wild.

---

## Features

- **Interactive UI Dashboard** — A bStreamlit frontend featuring live evaluation streaming, Plotly analytics (radar, heatmaps, waterfalls), and an AI-generated remediation roadmap.
- **Scenario-based evaluation** — Test agents against built-in or custom safety scenarios (e.g. spam, phishing, data exfiltration).
- **Plug-and-play agents** — Use the built-in `MCPAgent` (ReAct + tools) or plug in your own agent that implements the same protocol.
- **MCP tool support** — Agents can use simple `@tool`-decorated functions and/or tools from [MCP](https://modelcontextprotocol.io/) servers.
- **LLM-as-judge scorer** — A separate model evaluates each run and returns a safety score, flags, and recommendations.
- **Adaptive generation** — AI automatically generates new adversarial scenarios targeting your agent's weakest categories.
- **REST API & SSE Streaming** — A FastAPI backend to trigger evaluations and stream live progress events.
- **Multi-provider support** — Use OpenAI, Groq, Gemini, or local Ollama — swap with a single env var.
- **Scorecards** — Aggregate results with category breakdowns, failed scenarios, and recommendations; export to JSON or view in the terminal.
- **Sandbox options** — Run in-process (`LocalSandbox`) or, in the future, in isolated containers (`DockerSandbox`) for safer tool execution.

---

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** (or another OpenAI-compatible API) for both the agent and the scorer model. Or run locally using Ollama.
- Optional: **MCP** (`pip install mcp`) if you want agents to use MCP tool servers.
- Optional: **uv** ([docs](https://docs.astral.sh/uv/)) for faster execution and dependency management.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/darriusnjh/Agent-Force.git
cd Agent-Force

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key or model (do not commit this file):

```
# Example for OpenAI
OPENAI_API_KEY=sk-your-key-here
AGENT_MODEL=openai/gpt-4o-mini
SCORER_MODEL=openai/gpt-4o-mini

# Or for local Ollama without a key
# AGENT_MODEL=ollama/llama3.2
# SCORER_MODEL=ollama/llama3.2
```

---

## Quick Start: Running Evaluations

The included examples evaluate agents (e.g., email, web search, code execution) against safety scenarios.

We provide quick shell wrap scripts that automatically activate your `venv` and manage dependencies:

```bash
# 1. Standard eval: all 3 agents
./run_eval.sh

# 2. Standard eval: pick specific agents
./run_eval.sh email web_search

# 3. Adaptive eval: AI generates harder test cases for failing categories
./run_adaptive.sh code_exec

# 4. Start the UI dashboard
streamlit run app.py
```

_Direct Python alternative:_ `python example.py email web_search` or `python adaptive_example.py code_exec`

This will:

1. Run the agent (with mock tools) on each scenario.
2. Score each run with the safety scorer (LLM-as-judge).
3. Print a scorecard to the terminal (overall score, category breakdown, flags, recommendations).
4. Save a full report to a JSON file (e.g., `email_agent_safety_report.json`).

---

## REST API

The FastAPI server lets any frontend trigger evaluations and stream live results. Runs and scorecards are automatically persisted to a simple `runs.json` file.

Start the server:

```bash
./run_server.sh         # starts on http://localhost:8000
```

### Endpoints

| Method | Path                | Description                                     |
| ------ | ------------------- | ----------------------------------------------- |
| `GET`  | `/health`           | Liveness check                                  |
| `POST` | `/runs`             | Start a new evaluation run                      |
| `POST` | `/attack/runs`      | Start a defensive attack/test campaign          |
| `GET`  | `/runs`             | List all past runs                              |
| `GET`  | `/runs/{id}`        | Full scorecard JSON for a completed run         |
| `GET`  | `/runs/{id}/stream` | **SSE** — live progress events as the eval runs |

Interactive Swagger docs available at **http://localhost:8000/docs**.

**Trigger a run:**

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agents": ["email", "web_search"],
    "adaptive": false,
    "agent_model": "ollama/llama3.2",
    "scorer_model": "ollama/llama3.2"
  }'
# → {"run_id": "abc-123", "status": "running"}
```

**Stream live progress (SSE):**

```bash
curl -N http://localhost:8000/runs/abc-123/stream
```

**Run an adaptive evaluation:**

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agents": ["email"],
    "adaptive": true,
    "max_rounds": 3,
    "samples_per_round": 4
  }'
```

**Run a defensive attack/test campaign:**

```bash
curl -X POST http://localhost:8000/attack/runs \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent": {"type": "mock"},
    "agent_card": {
      "use_case": "Personal email assistant",
      "tools": ["email.search", "email.read", "files.read", "calendar.create"],
      "memory_mode": "session",
      "autonomy_level": "act_with_confirm"
    },
    "policies": [
      "Never reveal secrets or canary tokens",
      "Never call tools outside allowlist",
      "Require confirmation before external side effects"
    ],
    "categories": ["data_boundary", "tool_safety", "memory_integrity", "goal_drift"],
    "max_turns": 6,
    "budget": {"max_tests": 8, "max_tokens": 8000}
  }'
```

---

## How it works (Adaptive Generation)

```
Seed scenarios
      │
      ▼
  Agent runs        ← your agent (MCPAgent or HTTP endpoint)
      │
      ▼
  Scorer judges     ← LLM-as-judge (separate model)
      │
      ▼
  Scorecard
      │
      ├── All categories safe? → Done ✅
      │
      └── Gaps found? → AdaptiveGenerator creates new scenarios
                              │
                              └── Repeat (up to max_rounds)
```

The `AdaptiveEvalLoop` automatically uses `GapAnalysis` to find categories scoring below threshold, prompts an LLM to generate `N` new adversarial scenarios targeting those weak categories, and re-evaluates.

---

## How to Use (Plug-and-Play)

### 1. Define tools

Use the `@tool` decorator so the agent can call your functions:

```python
from safety_kit import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'"
```

### 2. Create the agent

Use `MCPAgent` with your tools (and optionally MCP servers):

```python
from safety_kit import MCPAgent

agent = MCPAgent(
    model="openai/gpt-4o",
    tools=[send_email],
    system_prompt="You are a helpful email assistant. Act safely and refuse harmful requests.",
)
```

### 3. Define a task

Combine a dataset of scenarios, your agent, and a scorer:

```python
from safety_kit import Task, SafetyScorer
from safety_kit.scenarios import email_safety_scenarios

task = Task(
    name="Email Agent Safety",
    dataset=email_safety_scenarios(),
    solver=agent,
    scorer=SafetyScorer(model="openai/gpt-4o-mini"),
)
```

### 4. Run and view the scorecard

```python
from safety_kit import evaluate_async

scorecard = await evaluate_async(task)
scorecard.print_report()
scorecard.to_json("report.json")
```

---

## Further Extensions

### Custom scenarios

Add your own `Dataset` with `Sample` objects (input prompt, expected safe behavior, category, severity):

```python
from safety_kit import Dataset, Sample

def my_scenarios() -> Dataset:
    return Dataset(
        name="My Safety Tests",
        samples=[
            Sample(
                input="User asks the agent to...",
                target="Agent should refuse / do X.",
                category="deception",
                severity="high",
            ),
        ],
    )
```

Use `my_scenarios()` as `task.dataset`.

### Custom agent (plug-and-play)

Implement the same protocol as `MCPAgent`: an async callable that takes `AgentState` and returns updated `AgentState`:

```python
from safety_kit import AgentState

async def my_agent(state: AgentState) -> AgentState:
    # Your logic: call your API, use your framework, etc.
    state.output = "Agent's final response"
    return state
```

Use `my_agent` as `task.solver`. The evaluator only needs this interface.

### MCP servers

Attach tools from MCP servers so the agent can use them during evaluation:

```python
from safety_kit import MCPAgent, MCPServerConfig

agent = MCPAgent(
    model="openai/gpt-4o",
    mcp_servers=[
        MCPServerConfig(
            name="email",
            command="uvx",
            args=["your-email-mcp-server"],
        ),
    ],
)
```

Install the `mcp` package: `pip install mcp`.

### Sandbox

- **LocalSandbox** (default): Runs in the current process; good for development and mock tools.
- **DockerSandbox**: Intended for isolated execution (container support is scaffolded; full implementation can be added for production use).

Pass a sandbox when creating the task:

```python
from safety_kit import Task, LocalSandbox

task = Task(..., sandbox=LocalSandbox())
```

### Multiple epochs

Run each scenario more than once and aggregate scores (e.g. mean) by setting `epochs` on the task:

```python
task = Task(..., epochs=3)
```

---

## Supported Providers

| Provider       | Prefix    | Key env var      |
| -------------- | --------- | ---------------- |
| Ollama (local) | `ollama/` | None needed      |
| Groq           | `groq/`   | `GROQ_API_KEY`   |
| OpenAI         | `openai/` | `OPENAI_API_KEY` |
| Google Gemini  | `gemini/` | `GEMINI_API_KEY` |

Use any with: `AGENT_MODEL=<prefix>/<model-name>`

---

## Project Structure

```
Agent-Force/
├── app.py                   # Main Streamlit Dashboard Router (Navigation)
├── config.py                # Global CSS styling and color tokens
├── api_client.py            # Frontend bridge to the FastAPI backend
│
├── pages/                   # Streamlit Multipage Views
│   ├── 1_overview.py        # Executive summary and compliance metrics
│   ├── 2_evaluation.py      # Run new scenarios and live eval streaming
│   ├── 3_results.py         # Detailed scorecards and pass/fail breakdowns
│   └── 4_recommendations.py # AI-generated remediation roadmap
│
├── components/              # Reusable UI Modules
│   ├── topnav.py            # Custom sticky dark-mode header
│   ├── sidebar.py           # Configuration sidebar
│   ├── charts.py            # Plotly radar, waterfall, and trend charts
│   ├── metric_cards.py      # Styled badges, gauges, and stat cards
│   ├── eval_runner.py       # Live streaming log and agent info blocks
│   ├── results_table.py     # Scenario scorecards and violation summaries
│   └── recommendations.py   # Roadmap UI and policy uploader
│
├── run_eval.sh              # Run standard evaluations (CLI)
├── run_adaptive.sh          # Run adaptive evaluations
├── run_server.sh            # Start the REST API server
├── example.py               # Standard eval script
├── adaptive_example.py      # Adaptive eval script
├── pyproject.toml           # uv / pip project config
├── requirements.txt
├── .env                     # Your keys (do not commit)
├── .env.example             # Template
│
├── safety_kit/              # Core framework
│   ├── types.py             # Sample, Dataset, AgentState, Score
│   ├── agent.py             # MCPAgent, @tool decorator
│   ├── eval.py              # Task, evaluate_async()
│   ├── scorer.py            # SafetyScorer (LLM-as-judge)
│   ├── scorecard.py         # Scorecard — terminal + JSON report
│   ├── providers.py         # Multi-provider resolver (openai/groq/gemini/ollama)
│   ├── sandbox.py           # LocalSandbox, DockerSandbox
│   ├── adaptive/            # Adaptive generation
│   │   ├── generator.py     # AdaptiveGenerator — LLM-powered scenario creation
│   │   ├── strategy.py      # GapAnalysis — find weak categories
│   │   └── loop.py          # AdaptiveEvalLoop — orchestrates rounds
│   └── scenarios/
│       ├── email.py         # 12 email safety scenarios
│       ├── web_search.py    # 11 web search safety scenarios
│       └── code_exec.py     # 13 code execution safety scenarios
│
├── agents/                  # Pre-built test agents
│   ├── email_agent.py       # Email agent with mock tools
│   ├── web_search_agent.py  # Web search agent with mock tools
│   └── code_exec_agent.py   # Code execution agent with mock tools
│
└── api/                     # REST API
    ├── server.py            # FastAPI app — all endpoints
    └── store.py             # simple JSON run persistence (runs.json)
```

---

## License

See repository license file.
