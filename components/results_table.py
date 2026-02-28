# ── components/results_table.py ───────────────────────────────────────────────

import streamlit as st
from config import COLORS, score_color
from components.metric_cards import (
    badge_html, score_ring_html, compliant_badge,
    framework_badge, severity_badge, section_label
)


def render_result_cards(results: list[dict]):
    """Score ring cards in a grid — one per scenario."""
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            color = COLORS["safe"] if r["compliant"] else COLORS["danger"]
            st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {'rgba(0,255,178,0.2)' if r['compliant'] else 'rgba(255,59,92,0.2)'};
            border-radius:12px;padding:16px 18px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:{COLORS['text_dim']};margin-bottom:4px;">{r['id']}</div>
      <div style="font-size:12px;font-weight:600;color:{COLORS['text']};
                  line-height:1.3;">{r['name'][:30]}{'…' if len(r['name'])>30 else ''}</div>
    </div>
    {score_ring_html(r['score'], size=50, stroke=5)}
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    {framework_badge(r['framework'])}
    {compliant_badge(r['compliant'])}
  </div>
</div>""", unsafe_allow_html=True)


def render_results_table(results: list[dict]):
    """Detailed expandable results table with judge reasoning."""
    section_label("EVALUATION RESULTS — LLM-AS-JUDGE AUDIT LOG")

    for r in results:
        color = COLORS["safe"] if r["compliant"] else COLORS["danger"]
        bg = "rgba(0,255,178,0.04)" if r["compliant"] else "rgba(255,59,92,0.04)"
        border = "rgba(0,255,178,0.2)" if r["compliant"] else "rgba(255,59,92,0.2)"

        with st.expander(
            f"{r['id']} — {r['name']}  |  Score: {r['score']}%  |  {'✓ COMPLIANT' if r['compliant'] else '✗ VIOLATION'}",
            expanded=not r["compliant"],
        ):
            col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.2, 1.2, 1])
            with col1:
                st.markdown(score_ring_html(r["score"], size=64, stroke=6),
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
<div style="font-size:13px;color:{COLORS['text']};font-weight:600;margin-bottom:8px;">{r['name']}</div>
{framework_badge(r['framework'])}
""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.1em;margin-bottom:6px;">ARTICLE</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:14px;
            color:{COLORS['accent2']};font-weight:700;">{r['article']}</div>
""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"{severity_badge(r['severity'])}", unsafe_allow_html=True)
            with col5:
                st.markdown(compliant_badge(r["compliant"]), unsafe_allow_html=True)

            st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
            padding:14px 18px;margin-top:12px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
               color:{COLORS['accent']};letter-spacing:0.08em;">JUDGE REASONING: </span>
  <span style="font-size:13px;color:{COLORS['text_dim']};line-height:1.7;">{r['reason']}</span>
</div>""", unsafe_allow_html=True)

            agent_output = (r.get("agent_output") or "").strip()
            if agent_output:
                st.markdown(
                    f"""
<div style="margin-top:12px;margin-bottom:6px;font-family:'JetBrains Mono',monospace;
            font-size:10px;color:{COLORS['accent3']};letter-spacing:0.08em;">
  AGENT RESPONSE
</div>
""",
                    unsafe_allow_html=True,
                )
                st.code(agent_output, language="text")

            tool_calls = r.get("tool_calls") or []
            if tool_calls:
                st.markdown(
                    f"""
<div style="margin-top:10px;margin-bottom:6px;font-family:'JetBrains Mono',monospace;
            font-size:10px;color:{COLORS['text_dim']};letter-spacing:0.08em;">
  TOOL CALLS
</div>
""",
                    unsafe_allow_html=True,
                )
                st.json(tool_calls)

def render_violations_summary(results):
    """Renders a self-contained box of failed scenarios."""
    violations = [r for r in results if not r.get('compliant', True)]
    
    if not violations:
        return

    C = COLORS
    
    # We open a styled outer container div here
    html = f"<div style='background-color: {C.get('panel', '#131722')}; border: 1px solid {C.get('border', '#1E2532')}; border-radius: 8px; padding: 20px; margin-top: 16px;'>\n"
    
    # Title
    html += f"<div style='color: {C.get('danger', '#ef4444')}; font-size: 11px; font-weight: 800; letter-spacing: 0.1em; margin-bottom: 16px; text-transform: uppercase;'>CRITICAL VIOLATIONS — {len(violations)} ITEMS REQUIRE ATTENTION</div>\n"
    
    # List items
    for r in violations:
        reason = r.get('reason', 'Behavior flagged by safety judge.')
        trunc_reason = reason[:110] + ('…' if len(reason) > 110 else '')
        id_str = r.get('id', '')
        
        html += f"<div style='margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid {C.get('border', '#1E2532')};'>\n"
        html += f"<div style='color: {C.get('text', '#E2E8F0')}; font-size: 13px; font-weight: 700; margin-bottom: 4px;'>{id_str} {r['name']}</div>\n"
        html += f"<div style='color: {C.get('text_dim', '#718096')}; font-size: 12px; line-height: 1.5;'>{trunc_reason}</div>\n"
        html += f"</div>\n"
        
    # Close the outer container div
    html += "</div>\n"
    
    st.markdown(html, unsafe_allow_html=True)
