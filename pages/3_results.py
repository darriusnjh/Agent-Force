# ── pages/3_results.py ────────────────────────────────────────────────────────
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_client import get_attack_report, get_radar_data, get_results, is_backend_alive
from components.charts import donut_chart, scores_bar_chart, waterfall_chart
from components.results_table import render_result_cards, render_results_table, render_violations_summary
from components.sidebar import render_sidebar
from components.topnav import render_page_header, render_topnav
from config import COLORS


def _render_panel(label: str, border_color: str):
    st.markdown(
        f"""
<div style="background:{COLORS['panel']};border:1px solid {border_color};
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">{label}</div>
""",
        unsafe_allow_html=True,
    )


def _render_attack_results(report: dict):
    summary = report.get("summary", {})
    campaign = report.get("campaign", {})
    taxonomy = report.get("failure_taxonomy", {})
    guardrail = report.get("guardrail_checks", {})
    stress = report.get("autonomy_stress", {})
    decay = report.get("safety_decay", {})
    findings = report.get("findings", [])
    control_mapping = report.get("control_mapping", {})
    risk_points = report.get("risk_evolution", [])

    st.markdown(
        f"""
<div style="font-family:'Space Mono',monospace;font-size:11px;color:{COLORS['accent']};
            letter-spacing:0.09em;text-transform:uppercase;margin:0 0 14px 0;">
  Attack Kit Results
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Tests", summary.get("total_tests", 0))
    with m2:
        st.metric("Total Findings", summary.get("total_findings", 0))
    with m3:
        st.metric("Max Severity", f"{summary.get('max_severity', 0.0):.1f}")
    with m4:
        st.metric("Autonomy Stress", f"{summary.get('autonomy_stress_index', 0.0):.1f}")

    shadow = summary.get("shadow_impact", {})
    m5, m6, m7 = st.columns(3)
    with m5:
        st.metric("Would Delete", shadow.get("would_delete", 0))
    with m6:
        st.metric("Would Archive", shadow.get("would_archive", 0))
    with m7:
        st.metric("Avg Reward", summary.get("avg_reward", 0.0))
    profile_label = str(summary.get("sandbox_profile", campaign.get("sandbox_profile", "unknown")))
    target_type = str(campaign.get("target_type", "unknown"))
    st.caption(
        f"Sandbox profile: {profile_label.replace('_', ' ').title()} · "
        f"Target type: {target_type.replace('_', ' ').title()}"
    )

    if control_mapping:
        _render_panel("CONTROL MAPPING COVERAGE", "rgba(255,160,28,0.24)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Control Library Size", control_mapping.get("library_size", 0))
        with c2:
            st.metric("Findings Mapped", control_mapping.get("findings_mapped", 0))
        with c3:
            st.metric("Standards Impacted", len(control_mapping.get("standard_counts", {})))

        finding_type_counts = control_mapping.get("finding_type_counts", {}) or {}
        if finding_type_counts:
            st.dataframe(
                [{"finding_type": k, "count": v} for k, v in finding_type_counts.items()],
                hide_index=True,
                use_container_width=True,
            )

        top_controls = control_mapping.get("top_controls", []) or []
        if top_controls:
            st.dataframe(top_controls, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    _render_panel("FAILURE TAXONOMY", "rgba(255,107,53,0.24)")
    taxonomy_rows = []
    for name, count in (taxonomy.get("counts") or {}).items():
        taxonomy_rows.append(
            {
                "taxonomy": name,
                "count": count,
                "max_severity": (taxonomy.get("max_severity") or {}).get(name, 0.0),
            }
        )
    st.dataframe(taxonomy_rows or [{"taxonomy": "none", "count": 0, "max_severity": 0.0}], hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("GUARDRAIL CHECKS", "rgba(0,212,255,0.24)")
    checks = [
        ("Stop Channel", guardrail.get("stop_channel_enforced", False)),
        ("Approval Gate", guardrail.get("approval_gate_enforced", False)),
        ("Privilege Boundaries", guardrail.get("privilege_boundaries_enforced", False)),
    ]
    for label, passed in checks:
        color = COLORS["safe"] if passed else COLORS["danger"]
        state = "PASS" if passed else "FAIL"
        st.markdown(
            f"<div style='color:{color};font-size:13px;margin:4px 0;'><b>{state}</b> — {label}</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Blocked tool calls: {guardrail.get('blocked_tool_calls', 0)}")
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("AUTONOMY STRESS / SAFETY DECAY", "rgba(123,97,255,0.24)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Avg Tool Calls / Turn", f"{stress.get('avg_tool_calls_per_turn', 0.0):.2f}")
    with s2:
        st.metric("Max Tool Calls / Turn", stress.get("max_tool_calls_in_turn", 0))
    with s3:
        st.metric("Low→High Decay", str(decay.get("decay_delta_low_to_high")))
    st.dataframe(
        [
            {"pressure": "low", "security_score": (decay.get("security_score_by_pressure") or {}).get("low")},
            {
                "pressure": "medium",
                "security_score": (decay.get("security_score_by_pressure") or {}).get("medium"),
            },
            {"pressure": "high", "security_score": (decay.get("security_score_by_pressure") or {}).get("high")},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("RISK EVOLUTION", "rgba(0,255,178,0.24)")
    if risk_points:
        chart_rows = [
            {
                "point": row.get("point"),
                "reward": row.get("reward"),
                "security_score": row.get("security_score"),
            }
            for row in risk_points
        ]
        st.line_chart(chart_rows, x="point", y=["reward", "security_score"], height=250)
    else:
        st.info("No risk evolution points available.")
    st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("ATTACK FINDINGS", "rgba(255,59,92,0.25)")
    finding_rows = []
    for idx, item in enumerate(findings, start=1):
        hits = [name for name, hit in (item.get("detector_hits") or {}).items() if hit]
        control_ids = item.get("mapped_control_ids", []) or []
        finding_types = item.get("finding_types", []) or []
        finding_rows.append(
            {
                "id": f"F-{idx:03d}",
                "category": item.get("category"),
                "taxonomy": item.get("taxonomy", "Unclassified"),
                "finding_types": ", ".join(finding_types[:3]),
                "severity": item.get("severity"),
                "detectors": ", ".join(hits[:4]),
                "controls": ", ".join(control_ids[:4]),
                "recommendation": item.get("recommendation", "")[:140],
            }
        )
    st.dataframe(
        finding_rows
        or [
            {
                "id": "-",
                "category": "-",
                "taxonomy": "-",
                "finding_types": "-",
                "severity": 0,
                "detectors": "-",
                "controls": "-",
                "recommendation": "No findings.",
            }
        ],
        hide_index=True,
        use_container_width=True,
    )

    for idx, item in enumerate(findings, start=1):
        relevance = item.get("control_relevance") or {}
        controls = relevance.get("relevant_standards_controls", []) or []
        if not controls:
            continue

        finding_title = f"F-{idx:03d} | {item.get('taxonomy', 'Unclassified')} | {item.get('category', 'general')}"
        with st.expander(finding_title, expanded=idx == 1):
            st.dataframe(
                [
                    {
                        "standard": c.get("standard", ""),
                        "control_id": c.get("control_id", ""),
                        "control_intent": c.get("control_intent", ""),
                    }
                    for c in controls
                ],
                hide_index=True,
                use_container_width=True,
            )

            why = str(relevance.get("why_it_matters", "")).strip()
            if why:
                st.markdown(f"**Why it matters:** {why}")

            evidence = relevance.get("evidence_from_run_logs", []) or []
            if evidence:
                st.markdown("**Evidence from run logs:**")
                for line in evidence:
                    st.markdown(f"- {line}")

            checklist = relevance.get("suggested_remediation_checklist", []) or []
            if checklist:
                st.markdown("**Suggested remediation checklist:**")
                for row in checklist:
                    st.markdown(f"- [ ] {row}")
    st.markdown("</div>", unsafe_allow_html=True)


alive = is_backend_alive()
render_topnav("results", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Evaluation Results", "Results", (COLORS["accent3"], COLORS["accent"]))

run_id = st.session_state.get("last_run_id")
attack_report = get_attack_report(run_id)

if attack_report:
    _render_attack_results(attack_report)
else:
    results = get_results(run_id)
    radar = get_radar_data()

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        filter_status = st.selectbox("Status", ["All", "Violations Only", "Compliant Only"])
    with f2:
        filter_fw = st.selectbox("Framework", ["All"] + list({r["framework"] for r in results}))
    with f3:
        filter_sev = st.multiselect(
            "Severity",
            ["critical", "high", "medium", "low"],
            default=["critical", "high", "medium", "low"],
        )

    filtered = results
    if filter_status == "Violations Only":
        filtered = [r for r in filtered if not r["compliant"]]
    elif filter_status == "Compliant Only":
        filtered = [r for r in filtered if r["compliant"]]
    if filter_fw != "All":
        filtered = [r for r in filtered if r["framework"] == filter_fw]
    if filter_sev:
        filtered = [r for r in filtered if r["severity"] in filter_sev]

    st.markdown(
        f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
            color:{COLORS['text_dim']};margin:4px 0 18px 0;">
  Showing <span style="color:{COLORS['accent']};">{len(filtered)}</span> of {len(results)} scenarios
</div>""",
        unsafe_allow_html=True,
    )

    _render_panel("SCENARIO SCORE CARDS", "rgba(0,212,255,0.18)")
    render_result_cards(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1])
    with c1:
        _render_panel("SCORES BY SCENARIO", "rgba(0,212,255,0.18)")
        scores_bar_chart(filtered, key="res_bar")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        _render_panel("PASS / FAIL BREAKDOWN", "rgba(0,255,178,0.18)")
        donut_chart(filtered, key="res_donut")
        st.markdown("</div>", unsafe_allow_html=True)

    _render_panel("FRAMEWORK DELTA VS 75% BASELINE", "rgba(123,97,255,0.18)")
    waterfall_chart(radar, key="res_waterfall")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""<div style="background:{COLORS['panel']};border:1px solid rgba(255,59,92,0.22);
                border-radius:12px;padding:20px 22px;margin-bottom:16px;">""",
        unsafe_allow_html=True,
    )
    render_violations_summary(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""<div style="background:{COLORS['panel']};border:1px solid rgba(123,97,255,0.2);
                border-radius:12px;padding:20px 22px;margin-bottom:16px;">""",
        unsafe_allow_html=True,
    )
    render_results_table(filtered)
    st.markdown("</div>", unsafe_allow_html=True)
