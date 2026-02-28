import streamlit as st

from config import COLORS
from components.metric_cards import pulsing_dot_html


def render_sidebar(backend_alive: bool = False) -> dict:
    """Render a non-config sidebar (status + links only).

    Configuration is intentionally kept on main pages.
    """
    with st.sidebar:
        st.markdown(
            f"""
<div style="padding:8px 0 24px 0;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
    <div style="width:38px;height:38px;border-radius:9px;
                background:linear-gradient(135deg,{COLORS['accent2']},{COLORS['accent']});
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:16px;font-family:'JetBrains Mono',monospace;
                color:#000;flex-shrink:0;">AF</div>
    <div>
      <div style="font-size:15px;font-weight:700;color:{COLORS['text']};">Agent-Force</div>
      <div style="font-size:9px;color:{COLORS['text_dim']};font-family:'Space Mono',monospace;
                  letter-spacing:0.08em;">AI GOVERNANCE ENGINE</div>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        status_color = COLORS["safe"] if backend_alive else COLORS["danger"]
        status_label = "API ONLINE" if backend_alive else "DEMO MODE"
        st.markdown(pulsing_dot_html(status_color, status_label), unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">
  CONFIG LOCATION
</div>
<div style="font-size:12px;color:{COLORS['text_dim']};line-height:1.45;">
  All run configuration now lives on each page.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div style="font-size:11px;color:{COLORS['text_dim']};text-align:center;">
  <a href="http://localhost:8000/docs" target="_blank"
     style="color:{COLORS['accent']};text-decoration:none;">
    API Docs
  </a>
  &nbsp;·&nbsp;
  <a href="https://github.com/darriusnjh/Agent-Force" target="_blank"
     style="color:{COLORS['accent2']};text-decoration:none;">
    GitHub
  </a>
</div>""",
            unsafe_allow_html=True,
        )

    # Keep a compatibility return shape for pages/components that read this.
    return {
        "agent": st.session_state.get("std_agent", "email"),
        "agent_model": st.session_state.get("std_agent_model", "openai/gpt-4o-mini"),
        "scorer_model": st.session_state.get("std_scorer_model", "openai/gpt-4o"),
        "adaptive": bool(st.session_state.get("std_adaptive", False)),
        "frameworks": st.session_state.get(
            "std_frameworks", ["EU AI Act", "GDPR", "NIST RMF 2.0"]
        ),
        "epochs": int(st.session_state.get("std_epochs", 1)),
        "max_rounds": int(st.session_state.get("std_max_rounds", 3)),
        "samples_per_round": int(st.session_state.get("std_samples_per_round", 4)),
    }

