"""Safety Kit -- Lightweight safety evaluation framework for AI agents."""

from .agent import MCPAgent, MCPServerConfig, tool
from .eval import Task, evaluate, evaluate_async
from .sandbox import DockerSandbox, LocalSandbox
from .scorer import SafetyScorer
from .scorecard import Scorecard
from .types import (
    AgentProtocol,
    AgentState,
    Dataset,
    SafetyLevel,
    Sample,
    ScorerProtocol,
    Score,
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
    "AgentProtocol",
    "AgentState",
    "Dataset",
    "SafetyLevel",
    "Sample",
    "ScorerProtocol",
    "Score",
    "ToolCall",
]
