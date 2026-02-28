from __future__ import annotations

from typing import Any


class HeuristicPlanner:
    """Coverage-first planner with light memory of prior failure rates."""

    _default_templates = {
        "data_boundary": "data_boundary_probe_v1",
        "tool_safety": "tool_scope_ambiguity_v1",
        "memory_integrity": "memory_write_policy_v1",
        "goal_drift": "goal_drift_pivot_v1",
        "confirmation_gate_failure": "confirmation_gate_failure_v1",
        "stop_failsafe_failure": "stop_failsafe_failure_v1",
        "context_compaction_failure": "context_compaction_failure_v1",
        "batching_runaway": "batching_runaway_v1",
        "oversight_failure": "oversight_loss_control_v1",
        "privilege_escalation": "privilege_ladder_probe_v1",
        "deceptive_compliance": "deceptive_compliance_probe_v1",
        "role_confusion": "role_confusion_probe_v1",
        "state_corruption": "state_corruption_probe_v1",
        "autonomy_escalation_boundary": "autonomy_escalation_boundary_v1",
        "safety_decay_pressure": "safety_decay_pressure_v1",
    }

    def __call__(
        self,
        agent_card: dict[str, Any],
        policies: list[str],
        categories: list[str],
        attack_memory: dict[str, Any],
    ) -> dict[str, Any]:
        del agent_card, policies

        if not categories:
            categories = ["tool_safety"]

        scored_categories: list[tuple[str, float, int]] = []
        for category in categories:
            templates = attack_memory.get(category, {})
            runs = sum(int(v.get("runs", 0)) for v in templates.values())
            fails = sum(int(v.get("fails", 0)) for v in templates.values())
            fail_rate = (fails / runs) if runs else 0.0
            scored_categories.append((category, fail_rate, runs))

        scored_categories.sort(key=lambda item: (item[2], -item[1]))
        category = scored_categories[0][0]

        template_id = self._choose_template_for_category(category, attack_memory)
        scenario_pack = (
            "resilience_stress"
            if category
            in {
                "confirmation_gate_failure",
                "stop_failsafe_failure",
                "context_compaction_failure",
                "batching_runaway",
                "oversight_failure",
                "safety_decay_pressure",
            }
            else "baseline_coverage"
        )
        return {
            "category": category,
            "template_id": template_id,
            "strategy": "coverage_first_with_failure_bias",
            "scenario_pack": scenario_pack,
        }

    def _choose_template_for_category(self, category: str, attack_memory: dict[str, Any]) -> str:
        existing = attack_memory.get(category, {})
        if not existing:
            return self._default_templates.get(category, f"{category}_probe_v1")

        best_tid = None
        best_score = -1.0
        for tid, stats in existing.items():
            runs = max(1, int(stats.get("runs", 0)))
            fails = int(stats.get("fails", 0))
            fail_rate = fails / runs
            if fail_rate > best_score:
                best_score = fail_rate
                best_tid = tid

        return best_tid or self._default_templates.get(category, f"{category}_probe_v1")
