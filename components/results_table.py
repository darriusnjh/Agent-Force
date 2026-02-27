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


def render_violations_summary(results: list[dict]):
    """Board-level violations summary cards."""
    violations = [r for r in results if not r["compliant"]]
    if not violations:
        st.markdown(f"""
<div style="background:rgba(0,255,178,0.06);border:1px solid rgba(0,255,178,0.25);
            border-radius:12px;padding:24px;text-align:center;">
  <div style="font-size:28px;margin-bottom:8px;">✅</div>
  <div style="font-size:15px;font-weight:700;color:{COLORS['safe']};">All Scenarios Compliant</div>
  <div style="font-size:12px;color:{COLORS['text_dim']};margin-top:4px;">No violations detected in this evaluation run.</div>
</div>""", unsafe_allow_html=True)
        return

    section_label(f"CRITICAL VIOLATIONS — {len(violations)} ITEMS REQUIRE ATTENTION")
    cols = st.columns(min(len(violations), 3))
    for i, r in enumerate(violations):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:rgba(255,59,92,0.06);border:1px solid rgba(255,59,92,0.25);
            border-radius:12px;padding:18px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    {framework_badge(r['framework'])}
    {score_ring_html(r['score'], size=44, stroke=4)}
  </div>
  <div style="font-size:13px;font-weight:600;color:{COLORS['text']};margin-bottom:8px;">{r['name']}</div>
  <div style="font-size:11px;color:{COLORS['text_dim']};line-height:1.6;">
    {r['reason'][:110]}{'…' if len(r['reason'])>110 else ''}
  </div>
</div>""", unsafe_allow_html=True)