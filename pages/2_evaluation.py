# â”€â”€ pages/2_evaluation.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import json
import os
import sys
import shlex

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_client import (
    get_scenarios,
    is_backend_alive,
    start_run,
    stream_run,
)
from components.eval_runner import (
    render_agent_info,
    render_live_log,
    render_scenario_queue,
    run_mock_evaluation,
)
from components.metric_cards import pulsing_dot_html
from components.sidebar import render_sidebar
from components.topnav import render_page_header, render_topnav
from config import COLORS

load_dotenv()

ATTACK_CATEGORIES = [
    "data_boundary",
    "tool_safety",
    "memory_integrity",
    "goal_drift",
    "oversight_failure",
    "privilege_escalation",
    "deceptive_compliance",
    "role_confusion",
    "state_corruption",
    "autonomy_escalation_boundary",
    "safety_decay_pressure",
    "context_compaction_failure",
    "confirmation_gate_failure",
    "stop_failsafe_failure",
    "batching_runaway",
]

STANDARD_AGENT_OPTIONS = ["email", "web_search", "code_exec", "jira", "custom_http"]
STANDARD_CUSTOM_DATASET_OPTIONS = ["email", "web_search", "code_exec", "jira"]
STANDARD_AGENT_MODEL_OPTIONS = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]
STANDARD_JUDGE_MODEL_OPTIONS = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]


def _parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _parse_json_list(raw: str) -> list[dict]:
    text = (raw or "").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Mock script must be a JSON array.")
    return [item for item in data if isinstance(item, dict)]


def _parse_mcp_arg_string(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        return [item for item in shlex.split(text) if item.strip()]
    except ValueError:
        # Fallback to simple JSON array input if shell parsing fails.
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        raise ValueError("MCP args must be a shell-like command line or JSON array.")


def _build_tool_specs(tools: list[str]) -> list[dict]:
    specs: list[dict] = []
    for tool in tools:
        low = tool.lower()
        side_effect = any(
            token in low for token in ("send", "create", "delete", "archive", "write", "post", "update")
        )
        required_privilege = "user"
        if any(token in low for token in ("system", "root")):
            required_privilege = "system"
        elif any(token in low for token in ("admin", "privilege", "secret", "credential")):
            required_privilege = "admin"
        elif side_effect:
            required_privilege = "elevated"

        specs.append(
            {
                "name": tool,
                "description": f"Tool `{tool}` declared by user input",
                "side_effect": side_effect,
                "required_privilege": required_privilege,
            }
        )
    return specs


def _render_panel(label: str, border_color: str):
    st.markdown(
        f"""
<div style="background:{COLORS['panel']};border:1px solid {border_color};
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px;">{label}</div>
""",
        unsafe_allow_html=True,
    )


def _render_attack_report(report: dict):
    summary = report.get("summary", {})
    taxonomy = report.get("failure_taxonomy", {})
    guardrail = report.get("guardrail_checks", {})
    stress = report.get("autonomy_stress", {})
    decay = report.get("safety_decay", {})
    findings = report.get("findings", [])
    risk_points = report.get("risk_evolution", [])

    st.markdown(
        f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin:6px 0 14px 0;">
  ATTACK CAMPAIGN REPORT
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tests", summary.get("total_tests", 0))
    with c2:
        st.metric("Findings", summary.get("total_findings", 0))
    with c3:
        st.metric("Max Severity", f"{summary.get('max_severity', 0.0):.1f}")
    with c4:
        st.metric("Autonomy Stress Index", f"{summary.get('autonomy_stress_index', 0.0):.1f}")

    c5, c6, c7 = st.columns(3)
    shadow = summary.get("shadow_impact", {})
    with c5:
        st.metric("Would Delete", shadow.get("would_delete", 0))
    with c6:
        st.metric("Would Archive", shadow.get("would_archive", 0))
    with c7:
        st.metric("Risk Evolution Points", len(risk_points))

    _render_panel("FAILURE TAXONOMY", "rgba(255,107,53,0.25)")
    taxonomy_rows = []
    for name, count in (taxonomy.get("counts") or {}).items():
        taxonomy_rows.append(
            {
                "taxonomy": name,
                "count": count,
                "max_severity": (taxonomy.get("max_severity") or {}).get(name, 0.0),
            }
        )
    if taxonomy_rows:
        st.dataframe(taxonomy_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No taxonomy findings available for this run.")
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("DETERMINISTIC GUARDRAIL CHECKS", "rgba(0,212,255,0.25)")
    checks = [
        ("Stop Channel Enforced", guardrail.get("stop_channel_enforced", False)),
        ("Approval Gate Enforced", guardrail.get("approval_gate_enforced", False)),
        ("Privilege Boundaries Enforced", guardrail.get("privilege_boundaries_enforced", False)),
    ]
    for label, passed in checks:
        color = COLORS["safe"] if passed else COLORS["danger"]
        mark = "PASS" if passed else "FAIL"
        st.markdown(
            f"<div style='font-size:13px;margin:4px 0;color:{color};'><b>{mark}</b> â€” {label}</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Blocked tool calls: {guardrail.get('blocked_tool_calls', 0)}")
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("AUTONOMY STRESS & SAFETY DECAY", "rgba(123,97,255,0.25)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Avg Tool Calls / Turn", f"{stress.get('avg_tool_calls_per_turn', 0.0):.2f}")
    with s2:
        st.metric("Max Tool Calls / Turn", stress.get("max_tool_calls_in_turn", 0))
    with s3:
        st.metric("Lowâ†’High Decay", str(decay.get("decay_delta_low_to_high")))

    pressure_scores = decay.get("security_score_by_pressure", {})
    pressure_rows = []
    for level in ("low", "medium", "high"):
        pressure_rows.append({"pressure": level, "security_score": pressure_scores.get(level)})
    st.dataframe(pressure_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("RISK EVOLUTION", "rgba(0,255,178,0.25)")
    if risk_points:
        chart_rows = []
        for row in risk_points:
            chart_rows.append(
                {
                    "point": row.get("point"),
                    "reward": row.get("reward"),
                    "security_score": row.get("security_score"),
                }
            )
        st.line_chart(chart_rows, x="point", y=["reward", "security_score"], height=260)
        st.dataframe(risk_points, use_container_width=True, hide_index=True, height=220)
    else:
        st.info("No risk evolution points for this run.")
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("FINDINGS", "rgba(255,59,92,0.25)")
    finding_rows = []
    for idx, item in enumerate(findings, start=1):
        detector_hits = item.get("detector_hits", {})
        active_hits = [name for name, hit in detector_hits.items() if hit]
        finding_rows.append(
            {
                "id": f"F-{idx:03d}",
                "category": item.get("category"),
                "taxonomy": item.get("taxonomy", "Unclassified"),
                "severity": item.get("severity"),
                "detectors": ", ".join(active_hits[:4]),
                "recommendation": item.get("recommendation", "")[:140],
            }
        )
    if finding_rows:
        st.dataframe(finding_rows, use_container_width=True, hide_index=True, height=300)
    else:
        st.success("No findings detected.")
    st.markdown("</div>", unsafe_allow_html=True)


alive = is_backend_alive()
render_topnav("evaluation", backend_alive=alive)
render_sidebar(backend_alive=alive)
render_page_header("Run Evaluation", "Evaluation", (COLORS["accent2"], COLORS["accent3"]))

scenarios = get_scenarios()
tab_standard = st.container()

with tab_standard:
    _render_panel("STANDARD EVALUATION CONFIGURATION", "rgba(0,212,255,0.2)")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        std_agent = st.selectbox(
            "Agent Under Test",
            STANDARD_AGENT_OPTIONS,
            key="std_agent",
            help="Built-in profile or `custom_http` for a plug-and-play endpoint.",
        )
    with cfg2:
        std_agent_model = st.selectbox(
            "Agent Model",
            STANDARD_AGENT_MODEL_OPTIONS,
            key="std_agent_model",
            help="Model used by the tested agent.",
        )
    with cfg3:
        std_scorer_model = st.selectbox(
            "Judge Model",
            STANDARD_JUDGE_MODEL_OPTIONS,
            key="std_scorer_model",
            help="Model used by the safety judge/scorer.",
        )

    std_custom_http_endpoint = ""
    std_custom_http_auth = ""
    std_custom_http_dataset = "email"
    if std_agent == "custom_http":
        custom1, custom2, custom3 = st.columns(3)
        with custom1:
            std_custom_http_endpoint = st.text_input(
                "Custom HTTP Endpoint",
                key="std_custom_http_endpoint",
                help="Agent endpoint to call for each scenario (e.g., http://127.0.0.1:9000/invoke).",
            )
        with custom2:
            std_custom_http_auth = st.text_input(
                "Authorization Header (optional)",
                key="std_custom_http_auth",
                help="Sent as the HTTP Authorization header (e.g., Bearer <token>).",
            )
        with custom3:
            std_custom_http_dataset = st.selectbox(
                "Scenario Profile",
                STANDARD_CUSTOM_DATASET_OPTIONS,
                key="std_custom_http_dataset",
                help="Which built-in safety scenario set to run against your custom agent endpoint.",
            )

    cfg4, cfg5, cfg6 = st.columns(3)
    with cfg4:
        std_adaptive = st.toggle(
            "Adaptive Generation",
            key="std_adaptive",
            help="Generate harder follow-up scenarios that target weak categories.",
        )
    with cfg5:
        if std_adaptive:
            std_max_rounds = st.slider("Max Adaptive Rounds", 1, 5, 3, key="std_max_rounds")
        else:
            std_max_rounds = 3
            st.session_state["std_max_rounds"] = 3
            st.caption("Fixed: 3")
    with cfg6:
        if std_adaptive:
            std_samples_per_round = st.slider("Samples per Round", 2, 8, 4, key="std_samples_per_round")
        else:
            std_samples_per_round = 4
            st.session_state["std_samples_per_round"] = 4
            st.caption("Fixed: 4")

    std_epochs = st.slider(
        "Epochs per Scenario",
        1,
        5,
        1,
        key="std_epochs",
        help="Displayed for run planning. Current backend uses one execution pass.",
    )

    cfg9, cfg10, cfg11 = st.columns(3)
    with cfg9:
        std_adversarial_adaptive = st.toggle(
            "Adversarial Adaptive Mode",
            key="std_adversarial_adaptive",
            help=(
                "Runs an adversarial attacker loop against each sandbox sample and "
                "stops early when a safety violation is detected."
            ),
        )
    with cfg10:
        if std_adversarial_adaptive:
            std_adversarial_max_turns = st.slider(
                "Adversarial Max Turns",
                1,
                10,
                4,
                key="std_adversarial_max_turns",
            )
        else:
            std_adversarial_max_turns = 4
            st.session_state["std_adversarial_max_turns"] = 4
            st.caption("Fixed: 4")
    with cfg11:
        std_adversarial_stop_on_violation = st.toggle(
            "Stop On First Violation",
            value=True,
            key="std_adversarial_stop_on_violation",
            help="Stop the attacker-target interaction as soon as a violation is detected.",
        )

    st.markdown("</div>", unsafe_allow_html=True)
    std_config = {
        "agent": std_agent,
        "agent_model": std_agent_model,
        "scorer_model": std_scorer_model,
        "custom_http_endpoint": std_custom_http_endpoint,
        "custom_http_auth": std_custom_http_auth,
        "custom_http_dataset": std_custom_http_dataset,
        "adaptive": std_adaptive,
        "epochs": std_epochs,
        "max_rounds": std_max_rounds,
        "samples_per_round": std_samples_per_round,
        "adversarial_adaptive": std_adversarial_adaptive,
        "adversarial_max_turns": std_adversarial_max_turns,
        "adversarial_stop_on_violation": std_adversarial_stop_on_violation,
    }

    col_main, col_info = st.columns([2, 1])

    with col_info:
        render_agent_info(std_config)

    with col_main:
        status_color = COLORS["safe"] if alive else COLORS["warn"]
        status_msg = (
            "Backend connected â€” live evaluation available"
            if alive
            else "Demo mode â€” mock evaluation with animated progress"
        )

        st.markdown(
            f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(0,212,255,0.2);
            border-radius:12px;padding:14px 22px;margin-bottom:16px;
            display:flex;align-items:center;gap:10px;">
  {pulsing_dot_html(status_color, status_msg)}
</div>""",
            unsafe_allow_html=True,
        )

        b1, b2, b3 = st.columns([2, 1, 1])
        with b1:
            run_clicked = st.button("RUN EVALUATION", use_container_width=True, key="std_run")
        with b2:
            adaptive_run = st.button(
                "ADAPTIVE RUN",
                use_container_width=True,
                key="std_adaptive_run_button",
            )
        with b3:
            st.button("ABORT", use_container_width=True, key="std_abort")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        std_mcp_server_command = st.text_input(
            "MCP Server Command (optional)",
            key="std_mcp_server_command",
            help=(
                "Executable used to start an MCP server. "
                "Example: `uvx`, `python`, or `node`."
            ),
        )
        std_mcp_server_args_raw = st.text_area(
            "MCP Server Arguments (optional)",
            key="std_mcp_server_args",
            height=70,
            help=(
                "Space-separated arguments (recommended) or JSON array. "
                "Example: `--jira-url https://... --jira-username user --jira-token token`."
            ),
        )
        try:
            std_mcp_server_args = _parse_mcp_arg_string(std_mcp_server_args_raw)
        except Exception as exc:
            st.error(f"Invalid MCP arg format: {exc}")
            std_mcp_server_args = None
        std_mcp_urls_raw = st.text_area(
            "MCP Server URLs (optional)",
            key="std_mcp_server_urls",
            height=80,
            help=(
                "One MCP endpoint per line. Examples: "
                "`https://your-host/mcp` (streamable-http default) or "
                "`sse|https://your-host/sse`."
            ),
        )
        std_mcp_server_urls = [
            line.strip() for line in (std_mcp_urls_raw or "").splitlines() if line.strip()
        ]
        with st.expander("Legacy MCP Registry Links (optional)", expanded=False):
            std_mcp_links_raw = st.text_area(
                "MCP Registry Links (one URL per line)",
                key="std_mcp_registry_links",
                height=70,
                help="Optional manifest URLs for command-based MCP launch.",
            )
        std_mcp_registry_links = [
            line.strip() for line in (std_mcp_links_raw or "").splitlines() if line.strip()
        ]

        if not run_clicked and not adaptive_run:
            _render_panel(f"SCENARIO QUEUE â€” {len(scenarios)} TESTS READY", "rgba(0,212,255,0.18)")
            render_scenario_queue(scenarios, running=False)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_clicked or adaptive_run:
            _render_panel("EVALUATION IN PROGRESS", "rgba(0,212,255,0.25)")

            if std_mcp_server_args is None:
                st.error("Fix MCP argument format before starting the run.")
            elif std_config.get("agent") == "custom_http" and not str(
                std_config.get("custom_http_endpoint", "")
            ).strip():
                st.error("Custom HTTP Endpoint is required when Agent Under Test is `custom_http`.")
            elif alive:
                with st.status("Initializing Agent-Force Engine...", expanded=True) as status_box:
                    run_id = start_run(
                        agents=[std_config.get("agent", "email")],
                        adaptive=std_config.get("adaptive", False) or adaptive_run,
                        agent_model=std_config.get("agent_model"),
                        scorer_model=std_config.get("scorer_model"),
                        max_rounds=std_config.get("max_rounds", 3),
                        samples_per_round=std_config.get("samples_per_round", 4),
                        adversarial_adaptive=bool(std_config.get("adversarial_adaptive", False)),
                        adversarial_max_turns=int(std_config.get("adversarial_max_turns", 4)),
                        adversarial_stop_on_violation=bool(
                            std_config.get("adversarial_stop_on_violation", True)
                        ),
                        mcp_registry_links=std_mcp_registry_links,
                        mcp_server_urls=std_mcp_server_urls,
                        mcp_server_command=(std_mcp_server_command or "").strip() or None,
                        mcp_server_args=std_mcp_server_args or [],
                        custom_http_endpoint=(
                            str(std_config.get("custom_http_endpoint", "")).strip() or None
                            if std_config.get("agent") == "custom_http"
                            else None
                        ),
                        custom_http_auth=(
                            str(std_config.get("custom_http_auth", "")).strip() or None
                            if std_config.get("agent") == "custom_http"
                            else None
                        ),
                        custom_http_dataset=(
                            str(std_config.get("custom_http_dataset", "email")).strip() or "email"
                            if std_config.get("agent") == "custom_http"
                            else None
                        ),
                    )

                    if run_id:
                        st.session_state["last_run_id"] = run_id
                        prog = st.progress(0)
                        log_lines, log_ph = [], st.empty()
                        interaction_ph = st.empty()
                        tool_history_ph = st.empty()
                        interaction_rows: list[dict] = []
                        tool_history_rows: list[dict] = []
                        max_scenarios = max(len(scenarios), 1)
                        completed_samples = 0
                        observed_total = max_scenarios

                        for i, event in enumerate(stream_run(run_id)):
                            event_type = str(event.get("type", "event"))
                            agent_name = str(event.get("agent", std_config.get("agent", "agent")))
                            sample_index = int(event.get("sample_index") or 0)
                            total_samples = int(event.get("total_samples") or observed_total or max_scenarios)
                            observed_total = max(observed_total, total_samples)

                            if event_type == "sample_start":
                                prompt = str(event.get("input", "")).strip()
                                log_lines.append(f"[{agent_name}] sample {sample_index}/{observed_total} started")
                                if prompt:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": "user",
                                            "text": prompt[:260],
                                        }
                                    )
                            elif event_type == "tool_call":
                                tool_name = str(event.get("tool_name", "unknown_tool"))
                                tool_input = json.dumps(event.get("tool_input", {}), ensure_ascii=True)
                                tool_output = str(event.get("tool_output", ""))
                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index} tool {tool_name}"
                                )
                                tool_history_rows.append(
                                    {
                                        "agent": agent_name,
                                        "sample": sample_index,
                                        "tool": tool_name,
                                        "input": tool_input[:160],
                                        "output": tool_output[:200],
                                    }
                                )
                                interaction_rows.append(
                                    {
                                        "sample": sample_index,
                                        "role": f"tool:{tool_name}",
                                        "text": f"in={tool_input[:120]} | out={tool_output[:120]}",
                                    }
                                )
                            elif event_type == "sample_result":
                                completed_samples = max(completed_samples, sample_index)
                                score = float(event.get("score", 0.0))
                                level = str(event.get("level", "unknown")).upper()
                                fallback_used = bool(event.get("fallback_used", False))
                                fallback_reason = str(event.get("fallback_reason", "")).strip()
                                agent_output = str(event.get("agent_output", "")).strip()
                                judge_explanation = str(event.get("judge_explanation", "")).strip()
                                sample_tool_calls = event.get("tool_calls", [])

                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index}/{observed_total} "
                                    f"{level} score={score:.2f}"
                                )
                                if fallback_used:
                                    note = fallback_reason[:160] if fallback_reason else "live run failed"
                                    log_lines.append(f"[{agent_name}] fallback to deterministic: {note}")

                                if agent_output:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": "agent",
                                            "text": agent_output[:260],
                                        }
                                    )
                                if judge_explanation:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": "judge",
                                            "text": judge_explanation[:260],
                                        }
                                    )
                                if isinstance(sample_tool_calls, list):
                                    for call in sample_tool_calls[:5]:
                                        if not isinstance(call, dict):
                                            continue
                                        tool_name = str(call.get("tool", "unknown_tool"))
                                        tool_input = json.dumps(call.get("input", {}), ensure_ascii=True)
                                        tool_output = str(call.get("output", ""))
                                        tool_history_rows.append(
                                            {
                                                "agent": agent_name,
                                                "sample": sample_index,
                                                "tool": tool_name,
                                                "input": tool_input[:160],
                                                "output": tool_output[:200],
                                            }
                                        )
                            elif event_type == "adversarial_turn":
                                turn = int(event.get("turn") or 0)
                                prompt = str(event.get("attacker_prompt", "")).strip()
                                out = str(event.get("agent_output", "")).strip()
                                detected = bool(event.get("detected_violation", False))
                                vi_types = event.get("violation_types", []) or []
                                status = "violation" if detected else "no_violation"
                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index} adversarial turn {turn} ({status})"
                                )
                                if prompt:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": f"attacker(t{turn})",
                                            "text": prompt[:260],
                                        }
                                    )
                                if out:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": f"agent(t{turn})",
                                            "text": out[:260],
                                        }
                                    )
                                if detected:
                                    interaction_rows.append(
                                        {
                                            "sample": sample_index,
                                            "role": "vulnerability",
                                            "text": f"Detected: {', '.join(str(v) for v in vi_types[:4])}",
                                        }
                                    )
                            elif event_type == "sample_vulnerability":
                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index} vulnerability flagged "
                                    f"(turn={event.get('first_violation_turn')}, "
                                    f"reason={str(event.get('stop_reason', 'detected'))[:80]})"
                                )
                            elif event_type == "sample_agent_error":
                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index} agent error: "
                                    f"{str(event.get('error', 'unknown'))[:180]}"
                                )
                            elif event_type == "sample_scorer_error":
                                log_lines.append(
                                    f"[{agent_name}] sample {sample_index} scorer error: "
                                    f"{str(event.get('error', 'unknown'))[:180]}"
                                )
                            elif event_type == "agent_complete":
                                level = str(event.get("level", "unknown")).upper()
                                score = float(event.get("score", 0.0))
                                log_lines.append(
                                    f"[{agent_name}] complete {level} score={score:.2f}"
                                )
                            elif event_type == "agent_error":
                                log_lines.append(
                                    f"[{agent_name}] error: {str(event.get('error', 'unknown'))[:180]}"
                                )
                            elif event_type == "done":
                                log_lines.append("[system] run complete")
                            else:
                                msg = event.get("message", event_type)
                                log_lines.append(f"[{agent_name}] {msg}")

                            if observed_total > 0:
                                progress_pct = min(int((completed_samples / observed_total) * 100), 99)
                            else:
                                progress_pct = min(int((i / max_scenarios) * 100), 99)
                            prog.progress(progress_pct)

                            with log_ph.container():
                                render_live_log(log_lines)
                            with interaction_ph.container():
                                st.markdown("**Live Interaction**")
                                if interaction_rows:
                                    st.dataframe(
                                        interaction_rows[-24:],
                                        use_container_width=True,
                                        hide_index=True,
                                        height=240,
                                    )
                                else:
                                    st.caption("Waiting for interaction events...")
                            with tool_history_ph.container():
                                st.markdown("**Tool Call History**")
                                if tool_history_rows:
                                    st.dataframe(
                                        tool_history_rows[-24:],
                                        use_container_width=True,
                                        hide_index=True,
                                        height=240,
                                    )
                                else:
                                    st.caption("No tool calls yet.")

                        prog.progress(100)
                        status_box.update(label="Evaluation Complete!", state="complete", expanded=False)
                        st.session_state["eval_done"] = True
                        st.switch_page("pages/3_results.py")
                    else:
                        status_box.update(label="Failed to start run.", state="error")
                        st.error("Failed to start run. Check API connection.")
            else:
                with st.status("Running Demo Evaluation...", expanded=True) as status_box:
                    completed = run_mock_evaluation(scenarios)
                    if completed:
                        status_box.update(label="Demo Evaluation Complete!", state="complete", expanded=False)
                        st.session_state["eval_done"] = True
                        st.switch_page("pages/3_results.py")

                        st.markdown("</div>", unsafe_allow_html=True)

