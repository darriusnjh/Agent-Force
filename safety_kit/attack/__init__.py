"""Defensive red-team attack/test agent components."""

from .adapters import HttpTargetAdapter, MockScriptedTargetAdapter
from .agent import AttackAgent, run_attack
from .detectors import Detectors, RuleBasedJudge
from .generator import SafeTemplateGenerator
from .planner import HeuristicPlanner
from .sandbox import SandboxToolProxy, SandboxWrapper, ToolPolicy, build_default_tool_policy, build_simulated_tools

__all__ = [
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
]
