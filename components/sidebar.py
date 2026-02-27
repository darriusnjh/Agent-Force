# ── components/sidebar.py ─────────────────────────────────────────────────────

import streamlit as st
from config import COLORS
from components.metric_cards import pulsing_dot_html


def render_sidebar(backend_alive: bool = False) -> dict:
    """
    Renders the sidebar and returns config dict selected by user.
    Returns: {agent, agent_model, scorer_model, adaptive, frameworks}
    """
    with st.sidebar:
        # ── Logo / brand ──────────────────────────────────────────────────────
        st.markdown(f"""
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
</div>""", unsafe_allow_html=True)

        # ── Backend status ────────────────────────────────────────────────────
        status_color = COLORS["safe"] if backend_alive else COLORS["danger"]
        status_label = "API ONLINE" if backend_alive else "DEMO MODE"
        st.markdown(pulsing_dot_html(status_color, status_label), unsafe_allow_html=True)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
  AGENT CONFIGURATION
</div>""", unsafe_allow_html=True)

        # ── Agent selection ───────────────────────────────────────────────────
        agent = st.selectbox(
            "Agent Under Test",
            ["email", "web_search", "code_exec", "custom"],
            help="Select which pre-built agent to evaluate",
        )

        agent_model = st.selectbox(
            "Agent Model",
            ["openai/gpt-4o-mini", "openai/gpt-4o",
             "groq/llama-3.1-8b-instant", "ollama/llama3.2",
             "gemini/gemini-1.5-flash"],
            help="Model powering the agent under test",
        )

        scorer_model = st.selectbox(
            "Judge Model",
            ["openai/gpt-4o", "openai/gpt-4o-mini",
             "anthropic/claude-3-5-sonnet", "groq/llama-3.1-70b-versatile"],
            help="High-capability model acting as legal compliance judge",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
  FRAMEWORKS
</div>""", unsafe_allow_html=True)

        frameworks = []
        framework_opts = {
            "EU AI Act": COLORS["accent2"],
            "GDPR": COLORS["accent"],
            "NIST RMF 2.0": COLORS["accent3"],
            "ISO 27001": COLORS["warn"],
            "SOC 2": "#FFD166",
            "CCPA": COLORS["muted"],
        }
        for fw, color in framework_opts.items():
            checked = st.checkbox(fw, value=fw in ["EU AI Act", "GDPR", "NIST RMF 2.0"],
                                  key=f"fw_{fw}")
            if checked:
                frameworks.append(fw)

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
  EVAL OPTIONS
</div>""", unsafe_allow_html=True)

        adaptive = st.toggle("Adaptive Generation", value=False,
                             help="AI generates harder scenarios targeting weak categories")
        epochs = st.slider("Epochs per Scenario", 1, 5, 1,
                           help="Run each scenario N times and average scores")
        if adaptive:
            max_rounds = st.slider("Max Adaptive Rounds", 1, 5, 3)
            samples_per_round = st.slider("Samples per Round", 2, 8, 4)
        else:
            max_rounds = 3
            samples_per_round = 4

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Quick stats ───────────────────────────────────────────────────────
        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
  QUICK STATS
</div>
<div style="display:flex;flex-direction:column;gap:8px;">
  {''.join([
    f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
    f'border-bottom:1px solid {COLORS["border"]};">'
    f'<span style="font-size:12px;color:{COLORS["text_dim"]};">{k}</span>'
    f'<span style="font-size:12px;color:{COLORS["accent"]};font-family:JetBrains Mono,monospace;">{v}</span>'
    f'</div>'
    for k, v in [("Scenarios", "8"), ("Frameworks", str(len(frameworks))),
                 ("Epochs", str(epochs)), ("Mode", "Adaptive" if adaptive else "Standard")]
  ])}
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Docs link ─────────────────────────────────────────────────────────
        st.markdown(f"""
<div style="font-size:11px;color:{COLORS['text_dim']};text-align:center;">
  <a href="http://localhost:8000/docs" target="_blank"
     style="color:{COLORS['accent']};text-decoration:none;">
    ⚡ API Docs
  </a>
  &nbsp;·&nbsp;
  <a href="https://github.com/darriusnjh/Agent-Force" target="_blank"
     style="color:{COLORS['accent2']};text-decoration:none;">
    GitHub
  </a>
</div>""", unsafe_allow_html=True)

    return {
        "agent": agent,
        "agent_model": agent_model,
        "scorer_model": scorer_model,
        "adaptive": adaptive,
        "frameworks": frameworks,
        "epochs": epochs,
        "max_rounds": max_rounds,
        "samples_per_round": samples_per_round,
    }