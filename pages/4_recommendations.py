# ── pages/4_recommendations.py ────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import COLORS, GLOBAL_CSS, LOGO_SVG
from api_client import get_recommendations, get_radar_data, is_backend_alive
from components.sidebar import render_sidebar
from components.topnav import render_topnav, render_page_header
from components.recommendations import (
    render_remediation_roadmap, render_risk_breakdown,
    render_export_panel, render_policy_uploader
)
from components.charts import waterfall_chart, radar_chart

alive = is_backend_alive()
render_topnav("recommendations", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Remediation Roadmap", "Recommendations",
                   (COLORS["warn"], COLORS["accent2"]))

recs  = get_recommendations(st.session_state.get("last_run_id"))
radar = get_radar_data()

col_main, col_side = st.columns([2, 1])

def _panel(border, label, fn):
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {border};
            border-radius:12px;padding:22px 24px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px;">{label}</div>
""", unsafe_allow_html=True)
    fn()
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    _panel("rgba(255,107,53,0.22)", "AI-GENERATED REMEDIATION ROADMAP",
           lambda: render_remediation_roadmap(recs))
    _panel("rgba(123,97,255,0.2)", "COMPLIANCE GAP ANALYSIS &mdash; DELTA VS 75% BASELINE",
           lambda: waterfall_chart(radar, key="recs_waterfall"))
    _panel("rgba(0,212,255,0.18)", "FRAMEWORK RADAR &mdash; CURRENT POSTURE",
           lambda: radar_chart(radar, key="recs_radar"))

with col_side:
    _panel("rgba(255,107,53,0.22)", "RISK EXPOSURE BREAKDOWN",
           lambda: render_risk_breakdown(recs))
    _panel("rgba(0,255,178,0.2)", "EXPORT AUDIT REPORT",
           lambda: render_export_panel())
    _panel("rgba(123,97,255,0.2)", "UPLOAD POLICY DOCUMENT",
           lambda: render_policy_uploader())