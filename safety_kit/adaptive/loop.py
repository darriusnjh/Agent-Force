"""AdaptiveEvalLoop — orchestrates the generate → evaluate → adapt cycle.

Each round:
  1. Run the full evaluation on the current dataset.
  2. Analyse the scorecard for weak categories (GapAnalysis).
  3. Generate new adversarial samples targeting those gaps (AdaptiveGenerator).
  4. Append the new samples and repeat.

The loop stops when:
  - All categories meet the safety threshold (no more gaps), OR
  - ``max_rounds`` is reached.

The returned Scorecard carries a ``generation_log`` entry in its results
metadata so callers can trace which scenarios were AI-generated and when.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..eval import evaluate_async
from ..scorecard import Scorecard
from ..types import Dataset, Sample
from .generator import AdaptiveGenerator
from .strategy import CategoryGap, GapAnalysis

logger = logging.getLogger(__name__)


@dataclass
class RoundSummary:
    """Summary of a single adaptive evaluation round."""

    round_number: int
    scorecard: Scorecard
    gaps: list[CategoryGap]
    new_samples_generated: int
    converged: bool  # True if no gaps remained after this round


@dataclass
class AdaptiveResult:
    """Final result of an AdaptiveEvalLoop run."""

    final_scorecard: Scorecard
    rounds: list[RoundSummary]
    total_generated: int

    @property
    def converged(self) -> bool:
        """True if all categories met the safety threshold by the final round."""
        return bool(self.rounds) and self.rounds[-1].converged

    def print_summary(self) -> None:
        """Print a compact summary of the adaptive run to stdout."""
        print(f"\n{'=' * 60}")
        print(f"  Adaptive Evaluation Summary")
        print(f"{'=' * 60}")
        print(f"  Rounds run         : {len(self.rounds)}")
        print(f"  Samples generated  : {self.total_generated}")
        print(f"  Converged          : {'yes ✅' if self.converged else 'no ⚠️'}")
        print(
            f"  Final score        : {self.final_scorecard.overall_score:.2f} / 1.00"
            f"  ({self.final_scorecard.overall_level.value.upper()})"
        )
        print()
        for r in self.rounds:
            gap_str = (
                ", ".join(f"{g.category}({g.score:.2f})" for g in r.gaps) or "none"
            )
            print(
                f"  Round {r.round_number}: score={r.scorecard.overall_score:.2f}"
                f"  gaps=[{gap_str}]"
                f"  generated={r.new_samples_generated}"
            )
        print()


class AdaptiveEvalLoop:
    """Runs multiple evaluation rounds, generating new test cases between rounds.

    Args:
        generator: :class:`AdaptiveGenerator` instance to create new samples.
        strategy: :class:`GapAnalysis` instance to identify weak categories.
        samples_per_round: How many new samples to generate each round.
        max_rounds: Maximum number of evaluation rounds to run.
        verbose: Whether to print per-sample progress during evaluation.
    """

    def __init__(
        self,
        generator: AdaptiveGenerator,
        strategy: GapAnalysis | None = None,
        samples_per_round: int = 5,
        max_rounds: int = 3,
        verbose: bool = True,
    ) -> None:
        self.generator = generator
        self.strategy = strategy or GapAnalysis()
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.verbose = verbose

    async def run(self, task) -> AdaptiveResult:  # task: Task (avoid circular import)
        """Execute the adaptive evaluation loop.

        Args:
            task: A :class:`~safety_kit.eval.Task` defining the agent, scorer,
                  and initial dataset.

        Returns:
            :class:`AdaptiveResult` with per-round summaries and the final scorecard.
        """
        from ..eval import Task  # local import to avoid circular dependency

        # Work on a mutable copy of the samples list
        current_samples: list[Sample] = list(task.dataset.samples)
        rounds: list[RoundSummary] = []
        total_generated = 0

        for round_num in range(1, self.max_rounds + 1):
            logger.info(
                "AdaptiveEvalLoop: starting round %d / %d", round_num, self.max_rounds
            )

            # Build a fresh Task for this round's dataset
            round_dataset = Dataset(
                name=task.dataset.name,
                samples=current_samples,
                description=task.dataset.description,
            )
            round_task = Task(
                name=task.name,
                dataset=round_dataset,
                solver=task.solver,
                scorer=task.scorer,
                sandbox=task.sandbox,
                epochs=task.epochs,
            )

            if self.verbose:
                print(f"\n{'─' * 60}")
                print(
                    f"  Adaptive Round {round_num}/{self.max_rounds} "
                    f"| {len(current_samples)} samples"
                )
                print(f"{'─' * 60}")

            scorecard = await evaluate_async(round_task, verbose=self.verbose)

            # Analyse gaps
            gaps = self.strategy.analyse(scorecard)
            converged = len(gaps) == 0

            if self.verbose:
                if converged:
                    print(
                        f"\n  ✅ Round {round_num}: all categories meet safety threshold — converged!"
                    )
                else:
                    gap_str = ", ".join(f"{g.category}({g.score:.2f})" for g in gaps)
                    print(
                        f"\n  ⚠️  Round {round_num}: gaps in [{gap_str}] — generating new samples…"
                    )

            rounds.append(
                RoundSummary(
                    round_number=round_num,
                    scorecard=scorecard,
                    gaps=gaps,
                    new_samples_generated=0,  # updated below
                    converged=converged,
                )
            )

            if converged or round_num == self.max_rounds:
                # Final round complete — no more generation needed
                return AdaptiveResult(
                    final_scorecard=scorecard,
                    rounds=rounds,
                    total_generated=total_generated,
                )

            # Generate new targeted samples for next round
            target_categories = [g.category for g in gaps]
            new_samples = await self.generator.generate(
                target_categories=target_categories,
                seed_samples=current_samples,
                n=self.samples_per_round,
                generation_round=round_num,
            )

            rounds[-1].new_samples_generated = len(new_samples)
            total_generated += len(new_samples)
            current_samples = current_samples + new_samples

            if self.verbose and new_samples:
                print(
                    f"+ Generated {len(new_samples)} new samples targeting: {target_categories}"
                )

        # Should not reach here — covered by the return inside the loop
        last_scorecard = (
            rounds[-1].scorecard if rounds else Scorecard.from_results(task.name, [])
        )
        return AdaptiveResult(
            final_scorecard=last_scorecard,
            rounds=rounds,
            total_generated=total_generated,
        )
