# ── components/topnav.py ──────────────────────────────────────────────────────
import streamlit as st
from config import COLORS, NAV_ITEMS

def _svg_icon(path_d: str, size: int = 16, color: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{path_d}"/></svg>'
    )

def _page_href(page_id: str) -> str:
    base_url_path = str(st.get_option("server.baseUrlPath") or "").strip("/")
    base_prefix = f"/{base_url_path}" if base_url_path else ""

    # Overview is the default page and should resolve to app root.
    if page_id == "overview":
        return f"{base_prefix}/"
    return f"{base_prefix}/{page_id}"

def render_topnav(active_page: str, backend_alive: bool = False):
    links_html = ""
    for item in NAV_ITEMS:
        is_active = item["id"] == active_page
        active_class = "active" if is_active else ""
        icon_color = COLORS["accent"] if is_active else COLORS["text_dim"]
        icon = _svg_icon(item["icon"], 14, icon_color)

        href = _page_href(item["id"])
        links_html += f'<a href="{href}" target="_self" class="af-navbtn {active_class}">{icon} <span>{item["label"]}</span></a>'

    status_color = COLORS["safe"] if backend_alive else COLORS["danger"]
    status_label = "API CONNECTED" if backend_alive else "DEMO MODE"

    st.markdown(f"""
    <div class="af-nav">
      <div class="af-brand">
        <svg viewBox="0 0 40 40" width="24" height="24">
          <defs>
            <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#7B61FF"/>
              <stop offset="100%" stop-color="#00D4FF"/>
            </linearGradient>
          </defs>
          <path d="M20 2 L36 9 L36 22 C36 30 28 37 20 39 C12 37 4 30 4 22 L4 9 Z" fill="url(#g)" opacity="0.95"/>
        </svg>
        <div>
          <div style="font-size:15px;font-weight:700;color:{COLORS['text']};line-height:1.2;">Agent-Force</div>
          <div style="font-family:'Space Mono',monospace;font-size:8px;color:{COLORS['text_dim']};letter-spacing:0.1em;">
            AI GOVERNANCE ENGINE
          </div>
        </div>
      </div>

      <div class="af-nav-links">
        {links_html}
      </div>

      <div style="display:flex;align-items:center;gap:8px;">
        <span class="pulse-dot" style="background:{status_color};box-shadow:0 0 8px {status_color};"></span>
        <span style="font-family:'Space Mono',monospace;font-size:10px;color:{COLORS['text_dim']};letter-spacing:0.06em;">
          {status_label}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str, gradient_colors: tuple):
    c1, c2 = gradient_colors
    st.markdown(f"""
    <div style="margin-bottom:28px;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:{COLORS['text_dim']};letter-spacing:0.16em;text-transform:uppercase;margin-bottom:8px;">
        AGENT-FORCE &nbsp;/&nbsp; {subtitle.upper()}
      </div>
      <div style="font-size:28px;font-weight:800;letter-spacing:-0.02em;background:linear-gradient(90deg,{c1},{c2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        {title}
      </div>
    </div>
    """, unsafe_allow_html=True)
