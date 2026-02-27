"""
Agent-Force — Multi-agent safety evaluation example.

Runs three agents (email, web search, code execution) through their respective
safety scenario suites and prints per-agent scorecards.

Usage:
    pip install -r requirements.txt
    python example.py

Set your OpenAI API key in .env:
    OPENAI_API_KEY=sk-...
"""

import asyncio
import os

from dotenv import load_dotenv

from agents import build_email_agent, build_web_search_agent, build_code_exec_agent
from safety_kit import Task, SafetyScorer, evaluate_async
from safety_kit.scenarios import (
    email_safety_scenarios,
    web_search_safety_scenarios,
    code_exec_safety_scenarios,
)

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
SCORER_MODEL = os.getenv("SCORER_MODEL", "gpt-4o-mini")


async def main() -> None:
    scorer = SafetyScorer(model=SCORER_MODEL)

    tasks = [
        Task(
            name="Email Agent Safety",
            dataset=email_safety_scenarios(),
            solver=build_email_agent(model=MODEL),
            scorer=scorer,
        ),
        Task(
            name="Web Search Agent Safety",
            dataset=web_search_safety_scenarios(),
            solver=build_web_search_agent(model=MODEL),
            scorer=scorer,
        ),
        Task(
            name="Code Execution Agent Safety",
            dataset=code_exec_safety_scenarios(),
            solver=build_code_exec_agent(model=MODEL),
            scorer=scorer,
        ),
    ]

    all_scorecards = []
    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"  Running: {task.name}  ({len(task.dataset)} scenarios)")
        print(f"{'=' * 60}")
        scorecard = await evaluate_async(task, verbose=True)
        scorecard.print_report()
        scorecard.to_json(f"{task.name.lower().replace(' ', '_')}_report.json")
        all_scorecards.append(scorecard)

    # Summary table
    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)
    for sc in all_scorecards:
        level_icon = {"safe": "✅", "borderline": "⚠️", "unsafe": "❌"}.get(
            sc.overall_level.value, "?"
        )
        print(
            f"  {level_icon}  {sc.task_name:<35}  "
            f"{sc.overall_score:.2f}/1.00  ({sc.overall_level.value.upper()})"
        )
    print()


if __name__ == "__main__":
    asyncio.run(main())
