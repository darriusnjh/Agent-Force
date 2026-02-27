# ── pages/2_evaluation.py ────────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import COLORS, GLOBAL_CSS, LOGO_SVG
from api_client import get_scenarios, start_run, is_backend_alive
from components.sidebar import render_sidebar
from components.topnav import render_topnav, render_page_header
from components.eval_runner import (
    render_agent_info, render_scenario_queue,
    render_live_log, run_mock_evaluation
)
from components.metric_cards import pulsing_dot_html

alive = is_backend_alive()
render_topnav("evaluation", backend_alive=alive)
config = render_sidebar(backend_alive=alive)
render_page_header("Run Evaluation", "Evaluation",
                   (COLORS["accent2"], COLORS["accent3"]))

scenarios = get_scenarios()
col_main, col_info = st.columns([2, 1])

with col_info:
    render_agent_info(config)
    fws = config.get("frameworks") or ["EU AI Act", "GDPR", "NIST RMF 2.0"]
    # Build framework list HTML safely
    _fw_items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;">'
        f'<svg width="10" height="10" viewBox="0 0 10 10">'
        f'<polygon points="5,0 10,5 5,10 0,5" fill="{COLORS["safe"]}"/></svg>'
        f'<span style="font-size:13px;color:{COLORS["text"]};">{fw}</span>'
        f'</div>'
        for fw in fws
    )
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(123,97,255,0.2);
            border-radius:12px;padding:20px 22px;margin-bottom:14px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px;">
    FRAMEWORKS UNDER TEST
  </div>
  {_fw_items}
</div>""", unsafe_allow_html=True)

with col_main:
    status_color = COLORS["safe"] if alive else COLORS["warn"]
    status_msg   = "Backend connected — live evaluation available" if alive else "Demo mode — mock evaluation with animated progress"

    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(0,212,255,0.2);
            border-radius:12px;padding:14px 22px;margin-bottom:16px;
            display:flex;align-items:center;gap:10px;">
  {pulsing_dot_html(status_color, status_msg)}
</div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([2, 1, 1])
    with b1: run_clicked    = st.button("RUN EVALUATION",  use_container_width=True)
    with b2: adaptive_run   = st.button("ADAPTIVE RUN",    use_container_width=True)
    with b3: st.button("ABORT",                            use_container_width=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if not run_clicked and not adaptive_run:
        st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(0,212,255,0.18);
            border-radius:12px;padding:20px 22px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">
    SCENARIO QUEUE &mdash; {len(scenarios)} TESTS READY
  </div>""", unsafe_allow_html=True)
        render_scenario_queue(scenarios, running=False)
        st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked or adaptive_run:
        st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(0,212,255,0.25);
            border-radius:12px;padding:20px 22px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">
    EVALUATION IN PROGRESS
  </div>""", unsafe_allow_html=True)

        if alive:
            run_id = start_run(
                agents=[config["agent"]],
                adaptive=config["adaptive"] or adaptive_run,
                agent_model=config["agent_model"],
                scorer_model=config["scorer_model"],
            )
            if run_id:
                st.session_state["last_run_id"] = run_id
                from api_client import stream_run
                prog = st.progress(0)
                log_lines, log_ph = [], st.empty()
                for i, event in enumerate(stream_run(run_id)):
                    log_lines.append(str(event))
                    prog.progress(min(int((i / len(scenarios)) * 100), 99))
                    with log_ph.container(): render_live_log(log_lines)
                prog.progress(100)
                st.success("Evaluation complete — navigate to Results to view scorecard.")
            else:
                st.error("Failed to start run. Check API connection.")
        else:
            completed = run_mock_evaluation(scenarios)
            if completed:
                st.markdown("</div>", unsafe_allow_html=True)
                st.success("Demo evaluation complete — navigate to Results to view the scorecard.")
                st.session_state["eval_done"] = True