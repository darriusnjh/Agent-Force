# ── pages/1_overview.py ───────────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import COLORS, GLOBAL_CSS, LOGO_SVG, score_color
from api_client import get_results, get_trend_data, get_radar_data, is_backend_alive
from components.metric_cards import stat_card
from components.charts import radar_chart, trend_chart, donut_chart, gauge_chart, heatmap_chart
from components.results_table import render_violations_summary
from components.topnav import render_topnav, render_page_header
from components.sidebar import render_sidebar

alive = is_backend_alive()
render_topnav("overview", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Compliance Overview", "Overview",
                   (COLORS["accent"], COLORS["accent2"]))

results = get_results(st.session_state.get("last_run_id"))
trend   = get_trend_data()
radar   = get_radar_data()

overall    = round(sum(r["score"] for r in results) / len(results))
violations = sum(1 for r in results if not r["compliant"])
compliant  = len(results) - violations
risk       = "HIGH" if violations >= 2 else "MEDIUM" if violations == 1 else "LOW"
risk_color = COLORS["danger"] if risk=="HIGH" else COLORS["warn"] if risk=="MEDIUM" else COLORS["safe"]
crit_count = sum(1 for r in results if not r["compliant"] and r["severity"]=="critical")

# KPI cards
c1, c2, c3, c4 = st.columns(4)
with c1: stat_card("OVERALL COMPLIANCE", f"{overall}%", "Across all frameworks", score_color(overall))
with c2: stat_card("VIOLATIONS FOUND",   str(violations), "Require immediate action", COLORS["danger"])
with c3: stat_card("SCENARIOS PASSED",   str(compliant),  f"Out of {len(results)} evaluated", COLORS["safe"])
with c4: stat_card("RISK EXPOSURE",      risk, f"{crit_count} critical P0 items open", risk_color)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

def _card(border_color, label, content_fn):
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {border_color};
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">{label}</div>
""", unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    _card("rgba(123,97,255,0.25)", "COMPLIANCE BY FRAMEWORK",
          lambda: radar_chart(radar, key="ov_radar"))
with col_r:
    _card("rgba(0,212,255,0.25)", "COMPLIANCE SCORE TREND",
          lambda: trend_chart(trend, key="ov_trend"))

c_a, c_b, c_c = st.columns(3)
with c_a:
    _card("rgba(0,212,255,0.2)", "OVERALL GAUGE",
          lambda: gauge_chart(overall, key="ov_gauge"))
with c_b:
    _card("rgba(0,255,178,0.2)", "PASS / FAIL RATIO",
          lambda: donut_chart(results, key="ov_donut"))
with c_c:
    _card("rgba(255,107,53,0.2)", "SEVERITY x FRAMEWORK HEATMAP",
          lambda: heatmap_chart(results, key="ov_heat"))


render_violations_summary(results)
st.markdown("</div>", unsafe_allow_html=True)