"""
Agent-Force — Multi-agent safety evaluation example.

Runs three agents (email, web search, code execution) through their respective
safety scenario suites and prints per-agent scorecards.

Usage:
    pip install -r requirements.txt
    python example.py

Set your chosen provider key in .env (see .env.example):
    OPENAI_API_KEY=sk-...     # OpenAI (default)
    GROQ_API_KEY=gsk_...      # Groq   (fast, cheap)
    GEMINI_API_KEY=AIza...    # Google Gemini

Select a provider via env vars:
    AGENT_MODEL=groq/llama-3.1-8b-instant
    SCORER_MODEL=groq/llama-3.1-8b-instant

    AGENT_MODEL=gemini/gemini-1.5-flash
    SCORER_MODEL=gemini/gemini-1.5-flash
"""

import asyncio
import os

from dotenv import load_dotenv

from agents import build_email_agent, build_web_search_agent, build_code_exec_agent
from safety_kit import Task, SafetyScorer, evaluate_async, list_providers
from safety_kit.scenarios import (
    email_safety_scenarios,
    web_search_safety_scenarios,
    code_exec_safety_scenarios,
)

load_dotenv()

# ── Model selection ────────────────────────────────────────────────────────
# Format: "<provider>/<model-name>"
# Supported providers: openai, groq, gemini, ollama
# Defaults to OpenAI gpt-4o-mini if not set in .env
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
SCORER_MODEL = os.getenv("SCORER_MODEL", "openai/gpt-4o-mini")


async def main() -> None:
    print(f"\nProviders available: {', '.join(list_providers())}")
    print(f"Agent model  : {AGENT_MODEL}")
    print(f"Scorer model : {SCORER_MODEL}")

    scorer = SafetyScorer(model=SCORER_MODEL)

    tasks = [
        Task(
            name="Email Agent Safety",
            dataset=email_safety_scenarios(),
            solver=build_email_agent(model=AGENT_MODEL),
            scorer=scorer,
        ),
        Task(
            name="Web Search Agent Safety",
            dataset=web_search_safety_scenarios(),
            solver=build_web_search_agent(model=AGENT_MODEL),
            scorer=scorer,
        ),
        Task(
            name="Code Execution Agent Safety",
            dataset=code_exec_safety_scenarios(),
            solver=build_code_exec_agent(model=AGENT_MODEL),
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

    # ── Final summary ──────────────────────────────────────────────────────
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
