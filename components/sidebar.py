# ── components/sidebar.py ─────────────────────────────────────────────────────

import json
import shlex

import streamlit as st
from config import COLORS
from components.metric_cards import pulsing_dot_html


def _parse_json_dict(raw: str, label: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"{label} must be valid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        st.error(f"{label} must be a JSON object.")
        return {}
    out: dict[str, str] = {}
    for key, value in data.items():
        out[str(key)] = str(value)
    return out


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
            ["email", "web_search", "code_exec", "jira", "custom"],
            help="Select a built-in agent or provide custom runtime configuration.",
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

        custom_agent = None
        custom_agent_error = None
        if agent == "custom":
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
  CUSTOM AGENT PLUG-IN
</div>""", unsafe_allow_html=True)

            custom_mode = st.selectbox(
                "Custom Agent Type",
                ["http", "mcp"],
                help="HTTP: call your running agent endpoint. MCP: launch an MCP server directly.",
            )
            custom_name = st.text_input("Custom Agent Name", value="Custom Agent")
            custom_dataset = st.selectbox(
                "Safety Dataset",
                ["email", "web_search", "code_exec", "jira"],
                help="Pick the scenario set used for evaluating your custom agent.",
            )
            fail_on_mutating = st.toggle(
                "Mark Mutating Tool Calls as Breach",
                value=True,
                help="When enabled, mutating tool calls are forced to UNSAFE in scoring.",
            )
            read_only = st.toggle(
                "Read-only Runtime Guard",
                value=True,
                help="When enabled, MCPAgent blocks mutating tools at runtime.",
            )

            if custom_mode == "http":
                endpoint_url = st.text_input("HTTP Endpoint URL", value="http://localhost:9000/invoke")
                timeout_seconds = st.slider("HTTP Timeout (seconds)", 5, 180, 30)
                headers_json = st.text_area(
                    "HTTP Headers (JSON object)",
                    value='{"Authorization":"Bearer <token>"}',
                    height=70,
                )
                headers = _parse_json_dict(headers_json, "HTTP Headers")
                if not endpoint_url.strip():
                    custom_agent_error = "Custom HTTP endpoint URL is required."

                custom_agent = {
                    "mode": "http",
                    "name": custom_name.strip() or "Custom Agent",
                    "dataset": custom_dataset,
                    "fail_on_mutating_tools": fail_on_mutating,
                    "read_only": read_only,
                    "endpoint_url": endpoint_url.strip(),
                    "timeout_seconds": float(timeout_seconds),
                    "headers": headers,
                }
            else:
                command = st.text_input("MCP Command", value="uvx")
                args_text = st.text_area(
                    "MCP Args (shell-style)",
                    value="mcp-atlassian --jira-url https://your-org.atlassian.net --jira-username you@example.com --jira-token <token>",
                    height=80,
                )
                env_json = st.text_area("MCP Env (JSON object)", value="{}", height=70)
                system_prompt = st.text_area("Custom System Prompt (optional)", value="", height=80)
                max_turns = st.slider("Max Agent Turns", 1, 20, 10)

                args = shlex.split(args_text.strip()) if args_text.strip() else []
                env = _parse_json_dict(env_json, "MCP Env")

                if not command.strip():
                    custom_agent_error = "MCP command is required."
                elif not args:
                    custom_agent_error = "MCP args are required."

                custom_agent = {
                    "mode": "mcp",
                    "name": custom_name.strip() or "Custom Agent",
                    "dataset": custom_dataset,
                    "fail_on_mutating_tools": fail_on_mutating,
                    "read_only": read_only,
                    "command": command.strip(),
                    "args": args,
                    "env": env,
                    "system_prompt": system_prompt.strip() or None,
                    "max_turns": int(max_turns),
                }

            if custom_agent_error:
                st.error(custom_agent_error)

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
        "custom_agent": custom_agent,
        "custom_agent_error": custom_agent_error,
    }
