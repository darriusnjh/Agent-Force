import os
import sys
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_client import get_run_data, is_backend_alive, list_runs
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


def _format_ts(value: str | None) -> str:
    if not value:
        return "-"
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


alive = is_backend_alive()
render_topnav("history", backend_alive=alive)
render_sidebar(backend_alive=alive)
render_page_header("Run History", "History", (COLORS["accent"], COLORS["accent2"]))

if not alive:
    st.warning("Backend API is offline. Start the server to load run history.")
    st.stop()

runs = list_runs()
if not runs:
    st.info("No past runs found yet.")
    st.stop()

run_options = {f"{r.get('run_id')}  |  {_format_ts(r.get('created_at'))}": r.get("run_id") for r in runs}
current = st.session_state.get("selected_run_id") or st.session_state.get("last_run_id")
selected_label = st.selectbox("Select Run", list(run_options.keys()))
selected_run_id = run_options[selected_label]

if selected_run_id != current:
    st.session_state["selected_run_id"] = selected_run_id
    st.session_state["last_run_id"] = selected_run_id

rows = []
for run in runs:
    cfg = run.get("config", {})
    rows.append(
        {
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "created_at": _format_ts(run.get("created_at")),
            "finished_at": _format_ts(run.get("finished_at")),
            "agents": ", ".join(cfg.get("agents", []) if isinstance(cfg.get("agents"), list) else []),
            "adaptive": cfg.get("adaptive"),
            "overall_score": run.get("overall_score"),
        }
    )

_render_panel("PAST RUNS")
st.dataframe(rows, hide_index=True, use_container_width=True, height=320)
st.markdown("</div>", unsafe_allow_html=True)

run_data = get_run_data(selected_run_id)
if run_data:
    c1, c2 = st.columns(2)
    with c1:
        _render_panel("SELECTED RUN CONFIG")
        st.json(run_data.get("config", {}))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        _render_panel("SELECTED RUN METADATA")
        meta = {
            "run_id": run_data.get("run_id"),
            "status": run_data.get("status"),
            "created_at": run_data.get("created_at"),
            "finished_at": run_data.get("finished_at"),
            "artifact_dir": run_data.get("artifact_dir"),
            "trace_path": run_data.get("trace_path"),
            "summary_path": run_data.get("summary_path"),
            "overall_score": run_data.get("overall_score"),
        }
        st.json(meta)
        st.markdown("</div>", unsafe_allow_html=True)

