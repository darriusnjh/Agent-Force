# components/results_table.py

import json
import streamlit as st
from config import COLORS
from components.metric_cards import (
    compliant_badge,
    score_ring_html,
    severity_badge,
    section_label,
)


def _truncate(value: object, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _tool_call_rows(tool_calls: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        rows.append(
            {
                "tool": str(call.get("tool", "unknown")),
                "input": _truncate(call.get("input", ""), 320),
                "output": _truncate(call.get("output", ""), 320),
            }
        )
    return rows


def _safe_json_dump(value: object) -> str:
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return str(value or "")


def render_result_cards(results: list[dict]):
    """Score ring cards in a grid — one per scenario."""
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.markdown(
                f"""
<div style="background:{COLORS['panel']};border:1px solid {'rgba(0,255,178,0.2)' if r['compliant'] else 'rgba(255,59,92,0.2)'};
            border-radius:12px;padding:16px 18px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:{COLORS['text_dim']};margin-bottom:4px;">{r['id']}</div>
      <div style="font-size:12px;font-weight:600;color:{COLORS['text']};
                  line-height:1.3;">{_truncate(r['name'], 30)}</div>
    </div>
    {score_ring_html(r["score"], size=50, stroke=5)}
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    {compliant_badge(r['compliant'])}
  </div>
</div>""",
                unsafe_allow_html=True,
            )


def render_results_table(results: list[dict]):
    """Detailed expandable results table with judge reasoning."""
    section_label("EVALUATION RESULTS — LLM-AS-JUDGE AUDIT LOG")

    for r in results:
        bg = "rgba(0,255,178,0.04)" if r["compliant"] else "rgba(255,59,92,0.04)"
        border = "rgba(0,255,178,0.2)" if r["compliant"] else "rgba(255,59,92,0.2)"
        judge_score = float(r.get("judge_score", r["score"] / 100))
        judge_level = str(r.get("judge_level", "unsafe")).upper() or ("SAFE" if r["compliant"] else "UNSAFE")
        starting_prompt = str(r.get("starting_prompt", r.get("name", "")))
        agent_output = str(r.get("agent_output", ""))
        judge_explanation = str(r.get("judge_explanation", r.get("reason", "")))
        recommendations = r.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        trace = r.get("adversarial_trace", [])
        if not isinstance(trace, list):
            trace = []
        tool_calls = r.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        with st.expander(
            f"{r['id']} — {r['name']}  |  Score: {r['score']}%  |  "
            f"{'✓ COMPLIANT' if r['compliant'] else '✗ VIOLATION'}",
            expanded=not r["compliant"],
        ):
            col1, col2, col3, col4 = st.columns([1, 2.2, 1.2, 1])
            with col1:
                st.markdown(score_ring_html(r["score"], size=64, stroke=6),
                            unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f"""
<div style="font-size:13px;color:{COLORS['text']};font-weight:600;margin-bottom:8px;">{r['name']}</div>
""",
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.1em;margin-bottom:6px;">ARTICLE</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:14px;
            color:{COLORS['accent2']};font-weight:700;">{r.get('article', '-')}</div>
""",
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(f"{severity_badge(r['severity'])}", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(compliant_badge(r["compliant"]), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:12px;color:{COLORS['text_dim']};margin-top:10px;'>"
                    f"Judge score: <b>{judge_score:.2f}</b><br/>"
                    f"Judge level: <b>{judge_level}</b>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
            padding:14px 18px;margin-top:12px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
               color:{COLORS['accent']};letter-spacing:0.08em;">JUDGE REASONING: </span>
  <span style="font-size:13px;color:{COLORS['text_dim']};line-height:1.7;">{_truncate(judge_explanation, 1400)}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
<div style="font-family:'Space Mono',monospace;font-size:11px;
            color:{COLORS['accent']};letter-spacing:0.08em;margin:10px 0 4px 0;">
  STARTING PROMPT (BENCHMARK INPUT)
</div>
""",
                unsafe_allow_html=True,
            )
            st.code(starting_prompt if starting_prompt else "(none)", language="markdown")

            if trace:
                st.markdown(
                    f"""
<div style="font-family:'Space Mono',monospace;font-size:11px;
            color:{COLORS['accent']};letter-spacing:0.08em;margin:12px 0 4px 0;">
  ADVERSARIAL TRACE (ATTACKER PROMPTS + AGENT RESPONSES)
</div>
""",
                    unsafe_allow_html=True,
                )
                for turn in trace:
                    if not isinstance(turn, dict):
                        continue
                    turn_no_raw = turn.get("turn", "?")
                    try:
                        turn_no = int(turn_no_raw)
                    except (TypeError, ValueError):
                        turn_no = str(turn_no_raw)
                    attacker_prompt = str(turn.get("attacker_prompt", "")).strip()
                    turn_output = str(turn.get("agent_output", "")).strip()
                    detected = bool(turn.get("detected_violation", False))
                    violation_types = turn.get("violation_types", [])
                    if not isinstance(violation_types, list):
                        violation_types = []
                    label = f"Turn {turn_no} — {'VIOLATION' if detected else 'NO VIOLATION'}"
                    if violation_types:
                        label += f" ({', '.join(str(v) for v in violation_types[:4])})"
                    with st.expander(label, expanded=detected):
                        if attacker_prompt:
                            st.markdown(
                                "<div style='font-size:11px;font-family:JetBrains Mono,monospace;"
                                "color:{0};margin-bottom:4px;'>Attacker prompt</div>".format(
                                    COLORS["text_dim"]
                                ),
                                unsafe_allow_html=True,
                            )
                            st.code(attacker_prompt, language="text")
                        if turn_output:
                            st.markdown(
                                "<div style='font-size:11px;font-family:JetBrains Mono,monospace;"
                                "color:{0};margin:10px 0 4px 0;'>Agent response</div>".format(
                                    COLORS["text_dim"]
                                ),
                                unsafe_allow_html=True,
                            )
                            st.code(turn_output, language="text")
                        call_rows = _tool_call_rows(turn.get("tool_calls", []))
                        if call_rows:
                            st.markdown(
                                "<div style='font-size:11px;font-family:JetBrains Mono,monospace;"
                                f"color:{COLORS['text_dim']};margin:10px 0 4px 0;'>Tool calls in this turn</div>",
                                unsafe_allow_html=True,
                            )
                            st.dataframe(call_rows, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    f"""
<div style="font-family:'Space Mono',monospace;font-size:11px;
            color:{COLORS['accent']};letter-spacing:0.08em;margin:12px 0 4px 0;">
  AGENT RESPONSE
</div>
""",
                    unsafe_allow_html=True,
                )
                if agent_output:
                    st.code(_truncate(agent_output, 1800), language="text")
                else:
                    st.caption("No agent output captured.")

            if tool_calls:
                with st.expander("Tool Call History", expanded=False):
                    st.dataframe(_tool_call_rows(tool_calls), use_container_width=True, hide_index=True)

            st.markdown(
                f"""
<div style="font-family:'Space Mono',monospace;font-size:11px;
            color:{COLORS['accent']};letter-spacing:0.08em;margin:12px 0 4px 0;">
  JUDGE RECOMMENDATIONS
</div>
""",
                unsafe_allow_html=True,
            )
            if recommendations:
                for idx, rec in enumerate(recommendations, start=1):
                    if isinstance(rec, str):
                        st.markdown(f"- **{idx}.** {rec}")
                    else:
                        st.markdown(f"- { _safe_json_dump(rec)}")
            else:
                st.caption("No recommendations returned for this sample.")


def render_violations_summary(results):
    """Renders a self-contained box of failed scenarios."""
    violations = [r for r in results if not r.get('compliant', True)]

    if not violations:
        return

    C = COLORS
    html = (
        f"<div style='background-color: {C.get('panel', '#131722')}; border: 1px solid {C.get('border', '#1E2532')}; "
        "border-radius: 8px; padding: 20px; margin-top: 16px;'>\n"
    )
    html += (
        f"<div style='color: {C.get('danger', '#ef4444')}; font-size: 11px; font-weight: 800; "
        f"letter-spacing: 0.1em; margin-bottom: 16px; text-transform: uppercase;'>"
        f"CRITICAL VIOLATIONS — {len(violations)} ITEMS REQUIRE ATTENTION</div>\n"
    )

    for r in violations:
        reason = r.get('reason', 'Behavior flagged by safety judge.')
        html += (
            f"<div style='margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid {C.get('border', '#1E2532')};'>\n"
            f"<div style='color: {C.get('text', '#E2E8F0')}; font-size: 13px; font-weight: 700; margin-bottom: 4px;'>{r.get('id', '')} {r['name']}</div>\n"
            f"<div style='color: {C.get('text_dim', '#718096')}; font-size: 12px; line-height: 1.5;'>{_truncate(reason, 110)}</div>\n"
            "</div>\n"
        )
    html += "</div>\n"
    st.markdown(html, unsafe_allow_html=True)
