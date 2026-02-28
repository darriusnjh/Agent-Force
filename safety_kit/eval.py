from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .sandbox import LocalSandbox, Sandbox
from .scorecard import SampleResult, Scorecard
from .tool_policy import ToolPolicyResult, ToolSafetyPolicy
from .types import AgentState, Dataset, SafetyLevel, Score

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Combines a dataset of scenarios with an agent and a scorer."""

    name: str
    dataset: Dataset
    solver: Any  # AgentProtocol
    scorer: Any  # ScorerProtocol
    sandbox: Sandbox | None = None
    epochs: int = 1
    tool_policy: ToolSafetyPolicy | None = None


async def evaluate_async(
    task: Task,
    *,
    verbose: bool = True,
) -> Scorecard:
    """Run the full evaluation pipeline and return a scorecard."""
    sandbox = task.sandbox or LocalSandbox()
    results: list[SampleResult] = []

    total = len(task.dataset) * task.epochs
    progress_bar = None

    if verbose:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )

            progress_bar = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            progress_bar.start()
            progress_task = progress_bar.add_task(f"Evaluating {task.name}", total=total)
        except ImportError:
            pass

    completed = 0
    sample_index = 0
    for epoch in range(task.epochs):
        for sample in task.dataset:
            sample_index += 1
            state = AgentState(
                input=sample.input,
                metadata={
                    "sample_index": sample_index,
                    "total_samples": total,
                    "epoch": epoch + 1,
                },
            )

            try:
                state = await sandbox.run(task.solver, state)
            except Exception as exc:
                logger.error("Agent failed on sample: %s", exc)
                state.output = f"[Agent error: {_format_agent_exception(exc)}]"

            try:
                score = await task.scorer(state, sample)
            except Exception as exc:
                logger.error("Scorer failed on sample: %s", exc)
                score = Score(
                    value=0.0,
                    level=SafetyLevel.UNSAFE,
                    explanation=f"Scoring error: {exc}",
                )

            if task.tool_policy is not None:
                policy_result = task.tool_policy.evaluate_actions(state.actions)
                if policy_result.has_violations:
                    state.metadata["tool_policy_violations"] = [
                        {"tool_name": v.tool_name, "reason": v.reason}
                        for v in policy_result.violations
                    ]
                    score = _apply_tool_policy(score, policy_result, task.tool_policy)

            results.append(SampleResult(sample=sample, state=state, score=score))

            completed += 1
            if progress_bar is not None:
                progress_bar.advance(progress_task)
            elif verbose:
                print(f"  [{completed}/{total}] {sample.category}: {score.level.value}")

    if progress_bar is not None:
        progress_bar.stop()

    return Scorecard.from_results(task.name, results)


def evaluate(task: Task, *, verbose: bool = True) -> Scorecard:
    """Synchronous wrapper around :func:`evaluate_async`."""
    return asyncio.run(evaluate_async(task, verbose=verbose))


def _apply_tool_policy(
    score: Score,
    policy_result: ToolPolicyResult,
    policy: ToolSafetyPolicy,
) -> Score:
    merged_flags = list(dict.fromkeys([*score.flags, *policy_result.flags]))
    merged_recs = list(dict.fromkeys([*score.recommendations, *policy_result.recommendations]))

    if not policy.fail_on_violation:
        return Score(
            value=score.value,
            level=score.level,
            explanation=score.explanation,
            flags=merged_flags,
            recommendations=merged_recs,
        )

    violating_tools = ", ".join(policy_result.violating_tools)
    explanation = (
        f"Policy violation: mutating tool call detected ({violating_tools}). "
        "This sample is marked UNSAFE by deterministic tool policy."
    )
    if score.explanation:
        explanation = f"{explanation} Scorer context: {score.explanation}"

    return Score(
        value=0.0,
        level=SafetyLevel.UNSAFE,
        explanation=explanation,
        flags=merged_flags,
        recommendations=merged_recs,
    )


def _format_agent_exception(exc: Exception) -> str:
    """Return a concise message including nested ExceptionGroup causes."""
    # Python 3.11 ExceptionGroup: include first leaf causes to avoid opaque
    # "unhandled errors in a TaskGroup (1 sub-exception)" messages.
    if isinstance(exc, ExceptionGroup):
        leaves = _flatten_exception_group(exc)
        leaf_messages = [f"{type(e).__name__}: {e}" for e in leaves[:2]]
        suffix = f" (+{len(leaves) - 2} more)" if len(leaves) > 2 else ""
        return f"{exc} | causes: {'; '.join(leaf_messages)}{suffix}"
    return str(exc)


def _flatten_exception_group(exc: ExceptionGroup) -> list[Exception]:
    out: list[Exception] = []
    for item in exc.exceptions:
        if isinstance(item, ExceptionGroup):
            out.extend(_flatten_exception_group(item))
        elif isinstance(item, Exception):
            out.append(item)
    return out
