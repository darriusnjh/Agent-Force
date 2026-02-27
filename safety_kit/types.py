from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


class SafetyLevel(str, Enum):
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"


@dataclass
class Sample:
    """A single safety test scenario."""

    input: str
    target: str
    category: str = "general"
    severity: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Adaptive generation metadata
    generated: bool = False  # True if created by AdaptiveGenerator
    generation_round: int = 0  # Which adaptive round produced this sample


@dataclass
class Dataset:
    """Collection of safety test scenarios."""

    name: str
    samples: list[Sample]
    description: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


@dataclass
class ToolCall:
    """Record of a single tool invocation during an agent run."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str


@dataclass
class AgentState:
    """Mutable state threaded through the agent evaluation pipeline."""

    input: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ToolCall] = field(default_factory=list)
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    """Safety evaluation result for a single sample."""

    value: float
    level: SafetyLevel
    explanation: str
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@runtime_checkable
class AgentProtocol(Protocol):
    """Any async callable that takes AgentState and returns AgentState."""

    async def __call__(self, state: AgentState) -> AgentState: ...


@runtime_checkable
class ScorerProtocol(Protocol):
    """Any async callable that scores an agent's behavior on a sample."""

    async def __call__(self, state: AgentState, sample: Sample) -> Score: ...
