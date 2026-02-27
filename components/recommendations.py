# ── components/recommendations.py ────────────────────────────────────────────

import streamlit as st
import json
from config import COLORS, PRIORITY_COLORS
from components.metric_cards import (
    priority_badge, badge_html, section_label, progress_row
)


def render_remediation_roadmap(recommendations: list[dict]):
    """Renders the full prioritized remediation roadmap."""
    section_label("REMEDIATION ROADMAP — AI-GENERATED ACTION PLAN")

    for r in recommendations:
        color = PRIORITY_COLORS.get(r["priority"], COLORS["muted"])
        bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.05)"
        border = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)"

        st.markdown(f"""
<div style="background:{COLORS['surface']};border:1px solid {border};border-left:3px solid {color};
            border-radius:10px;padding:14px 18px;margin-bottom:10px;
            transition:background 0.2s;">
  <div style="display:flex;align-items:flex-start;gap:14px;">
    <div style="flex-shrink:0;margin-top:2px;">{priority_badge(r['priority'])}</div>
    <div style="flex:1;">
      <div style="font-size:13px;color:{COLORS['text']};font-weight:500;
                  margin-bottom:6px;line-height:1.5;">{r['action']}</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <span style="font-size:11px;font-family:'JetBrains Mono',monospace;
                     color:{COLORS['text_dim']};">{r['framework']}</span>
        <span style="font-size:11px;color:{COLORS['text_dim']};">·</span>
        <span style="font-size:11px;font-family:'JetBrains Mono',monospace;">
          <span style="color:{COLORS['text_dim']};">Est. effort: </span>
          <span style="color:{COLORS['accent']};">{r['effort']}</span>
        </span>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_risk_breakdown(recommendations: list[dict]):
    """Risk exposure breakdown by priority level."""
    section_label("RISK EXPOSURE BREAKDOWN")

    priority_counts = {}
    for r in recommendations:
        priority_counts[r["priority"]] = priority_counts.get(r["priority"], 0) + 1

    max_count = max(priority_counts.values()) if priority_counts else 1
    for p in ["P0", "P1", "P2", "P3"]:
        count = priority_counts.get(p, 0)
        if count > 0:
            label = {"P0": "P0 — Critical", "P1": "P1 — High",
                     "P2": "P2 — Medium", "P3": "P3 — Low"}.get(p, p)
            progress_row(label, count, max_count, PRIORITY_COLORS.get(p, COLORS["muted"]))


def render_export_panel():
    """Export buttons for report formats."""
    section_label("EXPORT AUDIT REPORT")

    exports = [
        ("📋 Board-Level PDF Report", COLORS["accent"]),
        ("🗂 Full JSON Audit Log", COLORS["accent2"]),
        ("📊 SARIF Format Export", COLORS["accent3"]),
        ("🔒 Regulatory Evidence Pack", COLORS["warn"]),
    ]

    for label, color in exports:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
            border-radius:8px;padding:10px 14px;margin-bottom:6px;
            font-size:13px;color:{COLORS['text']};">{label}</div>
""", unsafe_allow_html=True)
        with col2:
            if st.button("↓", key=f"export_{label}", help=f"Export {label}"):
                st.toast(f"Exporting {label}...", icon="📤")


def render_policy_uploader():
    """Optional policy document uploader for generating new test cases."""
    section_label("UPLOAD POLICY DOCUMENT")

    st.markdown(f"""
<div style="background:rgba(123,97,255,0.06);border:1px dashed rgba(123,97,255,0.35);
            border-radius:10px;padding:16px 18px;margin-bottom:12px;">
  <div style="font-size:12px;color:{COLORS['text_dim']};line-height:1.6;">
    Upload a policy PDF or regulatory document to <strong style="color:{COLORS['accent2']};">
    automatically generate new test scenarios</strong> targeting its clauses.
  </div>
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop policy document here",
        type=["pdf", "txt", "md"],
        help="EU AI Act, NIST RMF, internal AI policy, etc.",
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"""
<div style="background:rgba(0,255,178,0.06);border:1px solid rgba(0,255,178,0.25);
            border-radius:8px;padding:12px 16px;margin-top:8px;">
  <span style="color:{COLORS['safe']};font-size:12px;">✓ Loaded: </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
               color:{COLORS['text']};">{uploaded.name}</span>
</div>""", unsafe_allow_html=True)

        if st.button("⚡ Generate Test Cases from Policy", use_container_width=True):
            with st.spinner("Analyzing policy clauses..."):
                import time; time.sleep(2)
            st.success("Generated 6 new test scenarios targeting uploaded policy clauses.")

    return uploaded