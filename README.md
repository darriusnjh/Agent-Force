# Agent-Force

A lightweight **safety evaluation framework** for personal AI agents. Run scenario-based evals, score agent behavior with an LLM judge, and get scorecards with actionable recommendations—all in a plug-and-play format.

Use it to answer: *"Would my email agent send malicious emails?"*, *"Does my assistant refuse harmful requests?"*, and similar safety questions before deploying agents in the wild.

---

## Features

- **Scenario-based evaluation** — Test agents against built-in or custom safety scenarios (e.g. spam, phishing, data exfiltration).
- **Plug-and-play agents** — Use the built-in `MCPAgent` (ReAct + tools) or plug in your own agent that implements the same protocol.
- **MCP tool support** — Agents can use simple `@tool`-decorated functions and/or tools from [MCP](https://modelcontextprotocol.io/) servers.
- **LLM-as-judge scorer** — A separate model evaluates each run and returns a safety score, flags, and recommendations.
- **Scorecards** — Aggregate results with category breakdowns, failed scenarios, and recommendations; export to JSON or view in the terminal.
- **Sandbox options** — Run in-process (`LocalSandbox`) or, in the future, in isolated containers (`DockerSandbox`) for safer tool execution.

---

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** (or another OpenAI-compatible API) for both the agent and the scorer model.
- Optional: **MCP** (`pip install mcp`) if you want agents to use MCP tool servers.

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

Create a `.env` file in the project root with your API key (do not commit this file):

```
OPENAI_API_KEY=sk-your-key-here
```

---

## Quick Start: Run the Example

The included example evaluates an **email agent** against 17 safety scenarios (legitimate use, spam, phishing, data privacy, harassment, deception, prompt injection).

```bash
# From the project root, with venv activated
python example.py
```

This will:

1. Run the agent (with mock `send_email`, `search_contacts`, `read_inbox` tools) on each scenario.
2. Score each run with the safety scorer (LLM-as-judge).
3. Print a scorecard to the terminal (overall score, category breakdown, flags, recommendations).
4. Save a full report to `email_safety_report.json`.

Example output:

```
╭──────────── Safety Scorecard ────────────╮
│ Email Agent Safety Evaluation            │
│ Overall Safety Score: 0.85 / 1.00 (SAFE)  │
│ Samples evaluated: 17                    │
╰──────────────────────────────────────────╯

Category Breakdown:
  legitimate_use  0.95  SAFE
  phishing        0.90  SAFE
  ...
```

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
    model="openai/gpt-4",
    tools=[send_email, search_contacts],
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
    scorer=SafetyScorer(model="openai/gpt-4"),
)
```

### 4. Run and view the scorecard

```python
from safety_kit import evaluate

scorecard = evaluate(task)
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
    model="openai/gpt-4",
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

### Different models

Use any OpenAI-compatible endpoint by passing `model` and optional `base_url`:

```python
# Local Ollama
agent = MCPAgent(model="ollama/llama3", base_url="http://localhost:11434/v1")

# Another API
scorer = SafetyScorer(model="openai/gpt-4", base_url="https://your-api.com/v1")
```

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

## Project Structure

```
Agent-Force/
├── README.md
├── requirements.txt
├── .env                    # Your API key (do not commit)
├── example.py              # Email safety evaluation example
├── email_safety_report.json # Generated after running example
├── safety_kit/
│   ├── __init__.py         # Public API
│   ├── types.py            # Sample, Dataset, AgentState, Score, protocols
│   ├── agent.py            # @tool, MCPAgent, MCPServerConfig
│   ├── eval.py             # Task, evaluate(), evaluate_async()
│   ├── scorer.py           # SafetyScorer (LLM-as-judge)
│   ├── scorecard.py        # Scorecard report + JSON export
│   ├── sandbox.py          # LocalSandbox, DockerSandbox
│   └── scenarios/
│       ├── __init__.py
│       └── email.py         # email_safety_scenarios()
├── inspect_ai/             # Inspect AI source (optional reference)
└── theory.py               # Inspect AI theory-of-mind example (separate)
```

---

## License

See repository license file.
