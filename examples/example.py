"""
Agent-Force — Multi-agent safety evaluation example.

Usage:
    python example.py                        # run all 3 agents
    python example.py email                  # email agent only
    python example.py email web_search       # pick specific agents

Agent names: email, web_search, code_exec

Set your chosen provider key in .env (see .env.example):
    AGENT_MODEL=ollama/llama3.2
    SCORER_MODEL=ollama/llama3.2
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents import build_code_exec_agent, build_email_agent, build_web_search_agent
from safety_kit import SafetyScorer, Task, evaluate_async, list_providers
from safety_kit.scenarios import (
    code_exec_safety_scenarios,
    email_safety_scenarios,
    web_search_safety_scenarios,
)

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
SCORER_MODEL = os.getenv("SCORER_MODEL", "openai/gpt-4o-mini")
REPORTS_DIR = Path("artifacts/reports")

# Registry — add new agents here
ALL_AGENTS = {
    "email": {
        "name": "Email Agent Safety",
        "dataset": email_safety_scenarios,
        "builder": build_email_agent,
    },
    "web_search": {
        "name": "Web Search Agent Safety",
        "dataset": web_search_safety_scenarios,
        "builder": build_web_search_agent,
    },
    "code_exec": {
        "name": "Code Execution Agent Safety",
        "dataset": code_exec_safety_scenarios,
        "builder": build_code_exec_agent,
    },
}


def parse_args() -> list[str]:
    parser = argparse.ArgumentParser(
        description="Run Agent-Force safety evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available agents: {', '.join(ALL_AGENTS)}",
    )
    parser.add_argument(
        "agents",
        nargs="*",
        choices=list(ALL_AGENTS) + [[]],  # allow empty → all
        metavar="AGENT",
        help="Which agent(s) to evaluate. Omit to run all.",
    )
    args = parser.parse_args()
    selected = args.agents if args.agents else list(ALL_AGENTS)

    # Validate
    invalid = [a for a in selected if a not in ALL_AGENTS]
    if invalid:
        print(
            f"Unknown agent(s): {invalid}. Choose from: {list(ALL_AGENTS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    return selected


async def main() -> None:
    selected = parse_args()

    print(f"\nProviders available: {', '.join(list_providers())}")
    print(f"Agent model  : {AGENT_MODEL}")
    print(f"Scorer model : {SCORER_MODEL}")
    print(f"Running      : {', '.join(selected)}")

    scorer = SafetyScorer(model=SCORER_MODEL)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        Task(
            name=ALL_AGENTS[key]["name"],
            dataset=ALL_AGENTS[key]["dataset"](),
            solver=ALL_AGENTS[key]["builder"](model=AGENT_MODEL),
            scorer=scorer,
        )
        for key in selected
    ]

    all_scorecards = []
    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"  Running: {task.name}  ({len(task.dataset)} scenarios)")
        print(f"{'=' * 60}")
        scorecard = await evaluate_async(task, verbose=True)
        scorecard.print_report()
        report_path = REPORTS_DIR / f"{task.name.lower().replace(' ', '_')}_report.json"
        scorecard.to_json(str(report_path))
        print(f"Report saved -> {report_path}")
        all_scorecards.append(scorecard)

    if len(all_scorecards) > 1:
        print("\n" + "=" * 60)
        print("  OVERALL SUMMARY")
        print("=" * 60)
        for sc in all_scorecards:
            level_icon = {"safe": "v", "borderline": "~", "unsafe": "x"}.get(
                sc.overall_level.value, "?"
            )
            print(
                f"  [{level_icon}]  {sc.task_name:<35}  "
                f"{sc.overall_score:.2f}/1.00  ({sc.overall_level.value.upper()})"
            )
        print()


if __name__ == "__main__":
    asyncio.run(main())
