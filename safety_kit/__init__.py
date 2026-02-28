"""Safety Kit -- Lightweight safety evaluation framework for AI agents."""

from .adaptive import AdaptiveEvalLoop, AdaptiveGenerator, CategoryGap, GapAnalysis
from .adaptive.loop import AdaptiveResult
from .attack import (
    AttackAgent,
    Detectors,
    HeuristicPlanner,
    HttpTargetAdapter,
    MockScriptedTargetAdapter,
    RuleBasedJudge,
    SafeTemplateGenerator,
    SandboxToolProxy,
    SandboxWrapper,
    ToolPolicy,
    build_default_tool_policy,
    build_simulated_tools,
    run_attack,
)
from .agent import MCPAgent, MCPServerConfig, tool
from .eval import Task, evaluate, evaluate_async
from .providers import list_providers
from .providers import resolve as resolve_provider
from .sandbox import DockerSandbox, LocalSandbox
from .scorecard import Scorecard
from .scorer import SafetyScorer
from .types import (
    AgentProtocol,
    AgentState,
    Dataset,
    SafetyLevel,
    Sample,
    Score,
    ScorerProtocol,
    ToolCall,
)

__all__ = [
    "MCPAgent",
    "MCPServerConfig",
    "tool",
    "Task",
    "evaluate",
    "evaluate_async",
    "SafetyScorer",
    "Scorecard",
    "LocalSandbox",
    "DockerSandbox",
    "resolve_provider",
    "list_providers",
    "AdaptiveGenerator",
    "GapAnalysis",
    "CategoryGap",
    "AdaptiveEvalLoop",
    "AdaptiveResult",
    "AttackAgent",
    "Detectors",
    "HeuristicPlanner",
    "HttpTargetAdapter",
    "MockScriptedTargetAdapter",
    "RuleBasedJudge",
    "SafeTemplateGenerator",
    "SandboxToolProxy",
    "SandboxWrapper",
    "ToolPolicy",
    "build_default_tool_policy",
    "build_simulated_tools",
    "run_attack",
    "AgentProtocol",
    "AgentState",
    "Dataset",
    "SafetyLevel",
    "Sample",
    "ScorerProtocol",
    "Score",
    "ToolCall",
]
