from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from .sandbox import LocalSandbox, Sandbox
from .scorecard import SampleResult, Scorecard
from .types import AgentState, Dataset

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
    for epoch in range(task.epochs):
        for sample in task.dataset:
            state = AgentState(input=sample.input)

            try:
                state = await sandbox.run(task.solver, state)
            except Exception as exc:
                logger.error("Agent failed on sample: %s", exc)
                state.output = f"[Agent error: {exc}]"

            try:
                score = await task.scorer(state, sample)
            except Exception as exc:
                logger.error("Scorer failed on sample: %s", exc)
                from .types import SafetyLevel, Score

                score = Score(
                    value=0.0,
                    level=SafetyLevel.UNSAFE,
                    explanation=f"Scoring error: {exc}",
                )

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
