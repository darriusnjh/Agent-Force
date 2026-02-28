"""
Run a safety evaluation against a real Jira MCP server.

Usage:
    python jira_mcp_demo.py
    python jira_mcp_demo.py --max-samples 3 --output jira_demo_report.json

Required env vars (if JIRA_MCP_ARGS_JSON is not set):
    JIRA_URL
    JIRA_USERNAME
    JIRA_API_TOKEN

Safety env vars:
    JIRA_EVAL_READ_ONLY=true
    AGENTFORCE_FAIL_ON_MUTATING_TOOLS=true
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from agents import build_jira_mcp_agent
from safety_kit import Dataset, SafetyScorer, Task, ToolSafetyPolicy, evaluate_async
from safety_kit.scenarios import jira_safety_scenarios

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jira MCP safety demo evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optionally limit sample count for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        default="jira_mcp_safety_report.json",
        help="Path for JSON scorecard output.",
    )
    return parser.parse_args()


def _select_dataset(max_samples: int) -> Dataset:
    dataset = jira_safety_scenarios()
    if max_samples <= 0 or max_samples >= len(dataset.samples):
        return dataset
    return Dataset(
        name=f"{dataset.name} (subset)",
        description=dataset.description,
        samples=dataset.samples[:max_samples],
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


async def main() -> None:
    args = parse_args()

    agent_model = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
    scorer_model = os.getenv("SCORER_MODEL", "openai/gpt-4o-mini")
    block_writes = _env_bool("JIRA_EVAL_READ_ONLY", True)
    fail_on_write = _env_bool("AGENTFORCE_FAIL_ON_MUTATING_TOOLS", True)

    try:
        solver = build_jira_mcp_agent(model=agent_model)
    except ValueError as exc:
        print(f"Jira MCP config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    task = Task(
        name="Jira MCP Agent Safety",
        dataset=_select_dataset(args.max_samples),
        solver=solver,
        scorer=SafetyScorer(model=scorer_model),
        tool_policy=ToolSafetyPolicy.strict_read_only(
            fail_on_violation=fail_on_write,
            block_on_violation=block_writes,
        ),
    )

    print(f"Agent model  : {agent_model}")
    print(f"Scorer model : {scorer_model}")
    print(f"Scenarios    : {len(task.dataset)}")
    print(f"Read-only    : {block_writes}")
    print(f"Fail on write: {fail_on_write}")

    scorecard = await evaluate_async(task, verbose=True)
    scorecard.print_report()
    scorecard.to_json(args.output)
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
