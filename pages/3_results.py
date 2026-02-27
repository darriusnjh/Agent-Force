# ── pages/3_results.py ────────────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import COLORS, GLOBAL_CSS, LOGO_SVG
from api_client import get_results, get_radar_data, is_backend_alive
from components.sidebar import render_sidebar
from components.topnav import render_topnav, render_page_header
from components.results_table import (
    render_result_cards, render_results_table, render_violations_summary
)
from components.charts import scores_bar_chart, donut_chart, waterfall_chart

alive = is_backend_alive()
render_topnav("results", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Evaluation Results", "Results",
                   (COLORS["accent3"], COLORS["accent"]))

results = get_results(st.session_state.get("last_run_id"))
radar   = get_radar_data()

# Filters
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    filter_status = st.selectbox("Status", ["All", "Violations Only", "Compliant Only"])
with f2:
    filter_fw = st.selectbox("Framework", ["All"] + list({r["framework"] for r in results}))
with f3:
    filter_sev = st.multiselect("Severity", ["critical","high","medium","low"],
                                 default=["critical","high","medium","low"])

filtered = results
if filter_status == "Violations Only":  filtered = [r for r in filtered if not r["compliant"]]
elif filter_status == "Compliant Only": filtered = [r for r in filtered if r["compliant"]]
if filter_fw != "All":  filtered = [r for r in filtered if r["framework"] == filter_fw]
if filter_sev:          filtered = [r for r in filtered if r["severity"] in filter_sev]

st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
            color:{COLORS['text_dim']};margin:4px 0 18px 0;">
  Showing <span style="color:{COLORS['accent']};">{len(filtered)}</span> of {len(results)} scenarios
</div>""", unsafe_allow_html=True)

def _panel(border, label, fn):
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {border};
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">{label}</div>
""", unsafe_allow_html=True)
    fn()
    st.markdown("</div>", unsafe_allow_html=True)

_panel("rgba(0,212,255,0.18)", "SCENARIO SCORE CARDS",
       lambda: render_result_cards(filtered))

c1, c2 = st.columns([1.6, 1])
with c1:
    _panel("rgba(0,212,255,0.18)", "SCORES BY SCENARIO",
           lambda: scores_bar_chart(filtered, key="res_bar"))
with c2:
    _panel("rgba(0,255,178,0.18)", "PASS / FAIL BREAKDOWN",
           lambda: donut_chart(filtered, key="res_donut"))

_panel("rgba(123,97,255,0.18)", "FRAMEWORK DELTA VS 75% BASELINE",
       lambda: waterfall_chart(radar, key="res_waterfall"))

st.markdown(f"""<div style="background:{COLORS['panel']};border:1px solid rgba(255,59,92,0.22);
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">""", unsafe_allow_html=True)
render_violations_summary(filtered)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""<div style="background:{COLORS['panel']};border:1px solid rgba(123,97,255,0.2);
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">""", unsafe_allow_html=True)
render_results_table(filtered)
st.markdown("</div>", unsafe_allow_html=True)