import pytest

from safety_kit.eval import Task, evaluate_async
from safety_kit.tool_policy import ToolSafetyPolicy, is_likely_mutating_tool_name
from safety_kit.types import AgentState, Dataset, SafetyLevel, Sample, Score, ToolCall


async def _solver_with_write_call(state: AgentState) -> AgentState:
    state.actions.append(
        ToolCall(
            tool_name="create_issue",
            tool_input={"project": "OPS", "title": "demo"},
            tool_output="ok",
        )
    )
    state.output = "Done"
    return state


async def _safe_scorer(state: AgentState, sample: Sample) -> Score:
    return Score(
        value=0.95,
        level=SafetyLevel.SAFE,
        explanation="Scorer marked safe.",
    )


@pytest.mark.asyncio
async def test_tool_policy_forces_unsafe_on_mutating_call():
    task = Task(
        name="policy-test",
        dataset=Dataset(
            name="d",
            samples=[Sample(input="x", target="y")],
        ),
        solver=_solver_with_write_call,
        scorer=_safe_scorer,
        tool_policy=ToolSafetyPolicy.strict_read_only(),
    )

    scorecard = await evaluate_async(task, verbose=False)
    result = scorecard.results[0]

    assert result.score.level == SafetyLevel.UNSAFE
    assert result.score.value == 0.0
    assert "mutating_tool_call" in result.score.flags


@pytest.mark.asyncio
async def test_tool_policy_warn_only_mode_keeps_original_score():
    task = Task(
        name="policy-test-warn",
        dataset=Dataset(
            name="d",
            samples=[Sample(input="x", target="y")],
        ),
        solver=_solver_with_write_call,
        scorer=_safe_scorer,
        tool_policy=ToolSafetyPolicy.strict_read_only(fail_on_violation=False),
    )

    scorecard = await evaluate_async(task, verbose=False)
    result = scorecard.results[0]

    assert result.score.level == SafetyLevel.SAFE
    assert result.score.value == 0.95
    assert "mutating_tool_call" in result.score.flags


def test_mutating_tool_name_heuristic():
    assert is_likely_mutating_tool_name("create_issue")
    assert is_likely_mutating_tool_name("addComment")
    assert not is_likely_mutating_tool_name("list_issues")
    assert not is_likely_mutating_tool_name("getIssue")
