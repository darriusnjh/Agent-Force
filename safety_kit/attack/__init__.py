"""Defensive red-team attack/test agent components."""

from .adapters import HttpTargetAdapter, MockScriptedTargetAdapter, StatefulSandboxTargetAdapter
from .agent import AttackAgent, run_attack
from .artifacts import SyntheticInboxConfig, build_default_attack_artifacts, build_synthetic_inboxes
from .detectors import Detectors, RuleBasedJudge
from .erl import ReflectionEngine, ReflectionMemory, RewardConfig, RewardModel
from .generator import SafeTemplateGenerator
from .planner import HeuristicPlanner
from .sandbox import (
    SandboxToolProxy,
    SandboxWrapper,
    StopController,
    ToolPolicy,
    build_default_tool_policy,
    build_simulated_tools,
)

__all__ = [
    "AttackAgent",
    "Detectors",
    "HeuristicPlanner",
    "HttpTargetAdapter",
    "MockScriptedTargetAdapter",
    "StatefulSandboxTargetAdapter",
    "RuleBasedJudge",
    "RewardConfig",
    "RewardModel",
    "ReflectionEngine",
    "ReflectionMemory",
    "SafeTemplateGenerator",
    "SandboxToolProxy",
    "SandboxWrapper",
    "StopController",
    "SyntheticInboxConfig",
    "ToolPolicy",
    "build_default_attack_artifacts",
    "build_default_tool_policy",
    "build_synthetic_inboxes",
    "build_simulated_tools",
    "run_attack",
]
