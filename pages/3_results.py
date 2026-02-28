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


def _render_attack_context(report: dict):
    summary = report.get("summary", {})
    guardrail = report.get("guardrail_checks", {})
    categories = summary.get("categories_tested", []) or []
    shadow = summary.get("shadow_impact", {}) or {}

    _render_panel("ATTACK CAMPAIGN SUMMARY", "rgba(255,107,53,0.24)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Tests", summary.get("total_tests", 0))
    with c2:
        st.metric("Findings", summary.get("total_findings", 0))
    with c3:
        st.metric("Max Severity", f"{summary.get('max_severity', 0.0):.1f}")
    with c4:
        st.metric("Avg Reward", summary.get("avg_reward", 0.0))

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Stop Channel", "PASS" if guardrail.get("stop_channel_enforced", False) else "FAIL")
    with c6:
        st.metric("Approval Gate", "PASS" if guardrail.get("approval_gate_enforced", False) else "FAIL")
    with c7:
        st.metric("Would Affect (Shadow)", shadow.get("total_would_affect", 0))

    if categories:
        st.caption("Categories tested: " + ", ".join(str(c).replace("_", " ").title() for c in categories))
    st.markdown("</div>", unsafe_allow_html=True)


alive = is_backend_alive()
render_topnav("results", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Evaluation Results", "Results", (COLORS["accent3"], COLORS["accent"]))

run_id = st.session_state.get("last_run_id")
attack_report = get_attack_report(run_id)
results = get_results(run_id)
radar = get_radar_data(run_id)

if attack_report:
    _render_attack_context(attack_report)

f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    filter_status = st.selectbox("Status", ["All", "Violations Only", "Compliant Only"])
with f2:
    filter_fw = st.selectbox("Framework", ["All"] + sorted({r["framework"] for r in results}))
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
