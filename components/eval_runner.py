# ── components/eval_runner.py ─────────────────────────────────────────────────
# Live evaluation progress panel + log stream

import streamlit as st
import time
from config import COLORS
from components.metric_cards import (
    badge_html, glow_card, section_label,
    pulsing_dot_html, severity_badge
)


def render_scenario_queue(scenarios: list[dict], current_idx: int = -1,
                          done_ids: set = None, running: bool = False):
    """Renders the animated scenario queue list."""
    done_ids = done_ids or set()

    for i, s in enumerate(scenarios):
        is_active = running and i == current_idx
        is_done = s["id"] in done_ids
        is_queued = not is_active and not is_done

        if is_active:
            bg = f"rgba(0,212,255,0.07)"
            border = f"rgba(0,212,255,0.3)"
            status_html = pulsing_dot_html(COLORS["accent"])
            name_color = COLORS["text"]
        elif is_done:
            bg = f"rgba(0,255,178,0.04)"
            border = f"rgba(0,255,178,0.15)"
            status_html = f'<span style="color:{COLORS["safe"]};font-size:14px;">✓</span>'
            name_color = COLORS["text_dim"]
        else:
            bg = COLORS["surface"]
            border = COLORS["border"]
            status_html = f'<span style="color:{COLORS["muted"]};font-size:10px;">○</span>'
            name_color = COLORS["text_dim"]

        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
            border-radius:8px;background:{bg};border:1px solid {border};
            margin-bottom:6px;transition:all 0.4s;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
               color:{COLORS['muted']};width:44px;flex-shrink:0;">{s['id']}</span>
  <span style="flex:1;font-size:13px;color:{name_color};">{s['name']}</span>
  <span style="margin-right:8px;">{severity_badge(s['severity'])}</span>
  <span>{status_html}</span>
</div>""", unsafe_allow_html=True)


def render_live_log(log_lines: list[str]):
    """Renders the live log panel."""
    lines_html = "".join([
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'color:{COLORS["accent"] if i == len(log_lines)-1 else COLORS["text_dim"]};'
        f'padding:3px 0;border-bottom:1px solid {COLORS["border"]}22;">'
        f'{"→ " if i < len(log_lines)-1 else "⚡ "}{line}'
        f'</div>'
        for i, line in enumerate(log_lines[-8:])
    ])

    st.markdown(f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:14px 16px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
    LIVE JUDGE LOG
  </div>
  {lines_html}
</div>""", unsafe_allow_html=True)


def render_agent_info(config: dict):
    """Renders the agent config info box."""
    rows = [
        ("Agent Type",    config.get("agent", "email")),
        ("Agent Model",   config.get("agent_model", "gpt-4o-mini")),
        ("Judge Model",   config.get("scorer_model", "gpt-4o")),
        ("Sandbox",       "LocalSandbox"),
        ("Mode",          "Adaptive" if config.get("adaptive") else "Standard"),
        ("Epochs",        str(config.get("epochs", 1))),
    ]
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<span style="font-size:12px;color:{COLORS["text_dim"]};">{k}</span>'
        f'<span style="font-size:12px;color:{COLORS["accent"]};'
        f'font-family:\'JetBrains Mono\',monospace;">{v}</span>'
        f'</div>'
        for k, v in rows
    ])

    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid rgba(0,255,178,0.2);
            border-radius:12px;padding:20px 22px;margin-bottom:14px;">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px;">
    AGENT UNDER TEST
  </div>
  {rows_html}
</div>""", unsafe_allow_html=True)


def run_mock_evaluation(scenarios: list[dict]):
    """
    Simulates a local evaluation with animated progress.
    In production: replace with stream_run() SSE consumer from api_client.
    """
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Progress bar placeholder
    prog_bar = st.progress(0)
    status_text = st.empty()
    queue_placeholder = st.empty()
    log_placeholder = st.empty()

    done_ids = set()
    log_lines = ["Initializing evaluation engine...", "Loading legal framework metadata..."]
    mock_logs = [
        "Injecting scenario prompt to agent...",
        "Awaiting agent tool calls...",
        "Agent response received, forwarding to judge...",
        "Judge model analyzing against legal clause...",
        "Parsing structured JSON output from judge...",
        "Score recorded. Moving to next scenario...",
    ]

    for i, scenario in enumerate(scenarios):
        pct = int(((i) / len(scenarios)) * 100)

        # Update progress
        prog_bar.progress(pct)
        status_text.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_dim"]};">Running: '
            f'<span style="color:{COLORS["accent"]};">{scenario["name"]}</span></div>',
            unsafe_allow_html=True,
        )

        # Animate through mock log lines for this scenario
        for j, log in enumerate(mock_logs[:4]):
            log_lines.append(log)
            with queue_placeholder.container():
                render_scenario_queue(scenarios, current_idx=i,
                                      done_ids=done_ids, running=True)
            with log_placeholder.container():
                render_live_log(log_lines)
            time.sleep(0.25)

        done_ids.add(scenario["id"])
        log_lines.append(f"✓ {scenario['id']} scored successfully")
        time.sleep(0.15)

    prog_bar.progress(100)
    status_text.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:11px;'
        f'color:{COLORS["safe"]};">✓ Evaluation complete — {len(scenarios)} scenarios processed</div>',
        unsafe_allow_html=True,
    )

    with queue_placeholder.container():
        render_scenario_queue(scenarios, current_idx=-1,
                              done_ids=done_ids, running=False)
    with log_placeholder.container():
        render_live_log(log_lines + ["Generating scorecard and recommendations..."])

    time.sleep(0.5)
    return True