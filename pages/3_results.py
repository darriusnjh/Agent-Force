import json
import os
import sys
from typing import Any

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_client import get_attack_report, get_run_data, is_backend_alive
from components.sidebar import render_sidebar
from components.topnav import render_page_header, render_topnav
from config import COLORS


def _render_panel(label: str):
    st.markdown(
        f"""
<div style="background:{COLORS['panel']};border:1px solid {COLORS['border']};
            border-radius:12px;padding:20px 22px;margin-bottom:16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">{label}</div>
""",
        unsafe_allow_html=True,
    )


def _load_trace_entries(run_data: dict) -> list[dict]:
    trace_path = run_data.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path or not os.path.exists(trace_path):
        return []
    try:
        with open(trace_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []
    except Exception:
        return []


def _format_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return "-"
    formatted = []
    for item in tool_calls[:8]:
        if not isinstance(item, dict):
            continue
        name = item.get("tool") or item.get("tool_name") or "tool"
        args = item.get("input") or item.get("args") or item.get("tool_input") or {}
        output = str(item.get("output") or item.get("tool_output") or "").strip()
        summary = f"{name}({args})"
        if output:
            summary += f" -> {output[:120]}"
        formatted.append(summary)
    return "\n".join(formatted) if formatted else "-"


def _extract_standard_rows(run_data: dict) -> list[dict]:
    results = run_data.get("results")
    if not isinstance(results, list):
        return []

    scorecard = None
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("scorecard"), dict):
            scorecard = item["scorecard"]
            break
    if not scorecard:
        return []

    score_items = scorecard.get("results")
    if not isinstance(score_items, list):
        return []

    trace_entries = _load_trace_entries(run_data)
    rows = []
    for idx, item in enumerate(score_items, 1):
        if not isinstance(item, dict):
            continue
        trace = trace_entries[idx - 1] if idx - 1 < len(trace_entries) else {}
        prompt = str(item.get("input", "")).strip()
        response = str(trace.get("output", "")).strip() or "-"
        tool_calls = _format_tool_calls(trace.get("tool_calls", []))
        recs = item.get("recommendations", [])
        rec_text = "\n".join([f"- {r}" for r in recs]) if isinstance(recs, list) and recs else "-"
        score_value = item.get("score")
        try:
            safety_score = round(float(score_value) * 100, 1)
        except Exception:
            safety_score = None
        rows.append(
            {
                "id": f"S-{idx:03d}",
                "mode": "standard",
                "category": item.get("category", "general"),
                "attack_prompt": prompt,
                "agent_response": response,
                "tool_calls": tool_calls,
                "safety_score": safety_score,
                "level": item.get("level", "unknown"),
                "recommendations": rec_text,
            }
        )
    return rows


def _extract_attack_rows(report: dict) -> list[dict]:
    findings = report.get("findings")
    if not isinstance(findings, list):
        return []

    rows = []
    for idx, item in enumerate(findings, 1):
        if not isinstance(item, dict):
            continue
        scenario = item.get("scenario") if isinstance(item.get("scenario"), dict) else {}
        prompt = (
            item.get("attack_prompt")
            or item.get("prompt")
            or scenario.get("prompt")
            or item.get("category")
            or "Attack case"
        )
        response = item.get("assistant_text") or item.get("response") or "-"
        tool_calls = (
            item.get("tool_calls")
            or item.get("requested_tool_calls")
            or scenario.get("tool_calls")
            or []
        )
        rows.append(
            {
                "id": f"A-{idx:03d}",
                "mode": "attack",
                "category": item.get("category", "attack"),
                "attack_prompt": str(prompt),
                "agent_response": str(response),
                "tool_calls": _format_tool_calls(tool_calls),
                "safety_score": item.get("severity"),
                "level": item.get("taxonomy", "finding"),
                "recommendations": str(item.get("recommendation", "-")),
            }
        )
    return rows


alive = is_backend_alive()
render_topnav("results", backend_alive=alive)
render_sidebar(backend_alive=alive)
render_page_header("Results", "Results", (COLORS["accent3"], COLORS["accent"]))

run_id = st.session_state.get("selected_run_id") or st.session_state.get("last_run_id")
if not run_id:
    st.info("No run selected yet. Start an evaluation or pick a run from History.")
    st.stop()

run_data = get_run_data(run_id)
if not run_data:
    st.warning("Could not load run data from API.")
    st.stop()

attack_report = get_attack_report(run_id)
rows = _extract_attack_rows(attack_report) if attack_report else _extract_standard_rows(run_data)

if not rows:
    st.info("No detailed records found for this run.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Run ID", run_id[:8] + "...")
with c2:
    st.metric("Mode", "Attack" if attack_report else "Standard")
with c3:
    st.metric("Records", len(rows))

filter_category = st.selectbox("Category Filter", ["All"] + sorted({str(r["category"]) for r in rows}))
filtered = rows if filter_category == "All" else [r for r in rows if str(r["category"]) == filter_category]

_render_panel("RUN RECORDS")
st.dataframe(
    [
        {
            "id": r["id"],
            "category": r["category"],
            "attack_prompt": (r["attack_prompt"][:160] + "...") if len(r["attack_prompt"]) > 160 else r["attack_prompt"],
            "agent_response": (r["agent_response"][:160] + "...") if len(r["agent_response"]) > 160 else r["agent_response"],
            "tool_calls": (r["tool_calls"][:160] + "...") if len(r["tool_calls"]) > 160 else r["tool_calls"],
            "safety_score": r["safety_score"],
            "recommendations": (r["recommendations"][:160] + "...") if len(r["recommendations"]) > 160 else r["recommendations"],
        }
        for r in filtered
    ],
    hide_index=True,
    use_container_width=True,
    height=320,
)
st.markdown("</div>", unsafe_allow_html=True)

_render_panel("DETAIL VIEW")
for row in filtered:
    with st.expander(f"{row['id']} | {row['category']} | score={row['safety_score']}"):
        st.markdown("**Attack Prompt**")
        st.code(row["attack_prompt"])
        st.markdown("**Agent Response**")
        st.code(row["agent_response"])
        st.markdown("**Tool Calls**")
        st.code(row["tool_calls"])
        st.markdown(f"**Safety Score:** `{row['safety_score']}`  |  **Level:** `{row['level']}`")
        st.markdown("**Recommendations**")
        st.code(row["recommendations"])
st.markdown("</div>", unsafe_allow_html=True)
