from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttackTurn:
    """Single generated user turn in a structured attack/test script."""

    user: str
    control: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallTrace:
    """Observed tool call with policy decision metadata."""

    tool: str
    args: dict[str, Any]
    allowed: bool
    result: str
    reason: str = ""
    requires_confirmation: bool = False
    confirmed: bool = False
    approval_token_present: bool = False
    approval_token_valid: bool = False
    after_stop: bool = False
    action_count: int = 0
    shadow_action_count: int = 0
    privilege_required: str = ""
    privilege_granted: str = ""
    privilege_violation: bool = False
    source_role: str = ""
    source_verified: bool = True
    tool_error: bool = False


@dataclass
class TurnTrace:
    """Execution trace for one test turn."""

    i: int
    user_msg: str
    assistant_text: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    detectors: dict[str, bool] = field(default_factory=dict)


@dataclass
class AttackTrace:
    """Full run trace for a single generated test script."""

    run_id: str
    turns: list[TurnTrace] = field(default_factory=list)


@dataclass
class Finding:
    """Structured finding emitted by analysis."""

    category: str
    severity: float
    turn_index: int
    detector_hits: dict[str, bool]
    evidence: dict[str, Any]
    recommendation: str
    taxonomy: str = "Unclassified"


@dataclass
class AttackState:
    """State for an attack campaign against one target agent."""

    run_id: str
    agent_card: dict[str, Any]
    policies: list[str]
    categories: list[str]
    max_turns: int
    budget: dict[str, int] = field(default_factory=dict)
    attack_memory: dict[str, Any] = field(default_factory=dict)
    reflection_memory: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
