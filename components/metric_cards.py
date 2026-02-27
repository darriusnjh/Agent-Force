# ── components/metric_cards.py ────────────────────────────────────────────────
# Reusable KPI stat cards rendered as styled HTML via st.markdown

import streamlit as st
from config import COLORS, score_color, SEVERITY_COLORS, PRIORITY_COLORS


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Badge ──────────────────────────────────────────────────────────────────────
def badge_html(text: str, color: str) -> str:
    bg = _hex_to_rgba(color, 0.15)
    border = _hex_to_rgba(color, 0.35)
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {border};'
        f'border-radius:4px;padding:2px 10px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.06em;text-transform:uppercase;'
        f'font-family:\'JetBrains Mono\',monospace;">{text}</span>'
    )


def severity_badge(severity: str) -> str:
    return badge_html(severity, SEVERITY_COLORS.get(severity, COLORS["muted"]))


def priority_badge(priority: str) -> str:
    return badge_html(priority, PRIORITY_COLORS.get(priority, COLORS["muted"]))


def framework_badge(framework: str) -> str:
    return badge_html(framework, COLORS["accent2"])


def compliant_badge(compliant: bool) -> str:
    return badge_html("COMPLIANT" if compliant else "VIOLATION",
                      COLORS["safe"] if compliant else COLORS["danger"])


# ── Score Ring (SVG) ──────────────────────────────────────────────────────────
def score_ring_html(score: int, size: int = 72, stroke: int = 6) -> str:
    import math
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    dash = (score / 100) * circ
    color = score_color(score)
    cx = cy = size / 2
    font_size = size * 0.22
    return f"""
<svg width="{size}" height="{size}" style="transform:rotate(-90deg)">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
    stroke="{COLORS['border']}" stroke-width="{stroke}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
    stroke="{color}" stroke-width="{stroke}"
    stroke-dasharray="{dash:.1f} {circ:.1f}"
    stroke-linecap="round"
    style="filter:drop-shadow(0 0 6px {color});transition:stroke-dasharray 1s ease"/>
  <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central"
    style="transform:rotate(90deg);transform-origin:50% 50%;
           font-size:{font_size}px;font-weight:800;fill:{color};
           font-family:'JetBrains Mono',monospace;">{score}</text>
</svg>"""


# ── Stat Card ─────────────────────────────────────────────────────────────────
def stat_card(label: str, value: str, sub: str, color: str, icon: str = ""):
    bg = _hex_to_rgba(color, 0.06)
    border = _hex_to_rgba(color, 0.25)
    glow = _hex_to_rgba(color, 0.08)
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {border};border-radius:12px;
            padding:20px 22px;box-shadow:0 0 24px {glow},inset 0 1px 0 {_hex_to_rgba(color,0.1)};
            transition:transform 0.2s;">
  <div style="font-family:'Space Mono',monospace;font-size:10px;color:{COLORS['text_dim']};
              letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">{label}</div>
  <div style="font-size:2.4rem;font-weight:900;color:{color};line-height:1;
              font-family:'JetBrains Mono',monospace;margin-bottom:6px;
              text-shadow:0 0 20px {_hex_to_rgba(color,0.4)};">{icon}{value}</div>
  <div style="font-size:12px;color:{COLORS['text_dim']};">{sub}</div>
</div>""", unsafe_allow_html=True)


# ── Glow Card wrapper ─────────────────────────────────────────────────────────
def glow_card(content_html: str, color: str = COLORS["accent"], padding: str = "22px 26px"):
    bg = _hex_to_rgba(color, 0.04)
    border = _hex_to_rgba(color, 0.2)
    glow = _hex_to_rgba(color, 0.07)
    st.markdown(f"""
<div style="background:{COLORS['panel']};border:1px solid {border};border-radius:12px;
            padding:{padding};box-shadow:0 0 28px {glow},inset 0 1px 0 {_hex_to_rgba(color,0.08)};
            margin-bottom:16px;">
  {content_html}
</div>""", unsafe_allow_html=True)


# ── Section label (mono uppercase) ───────────────────────────────────────────
def section_label(text: str):
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
        f'color:{COLORS["text_dim"]};letter-spacing:0.12em;text-transform:uppercase;'
        f'margin-bottom:14px;">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Pulsing dot ───────────────────────────────────────────────────────────────
def pulsing_dot_html(color: str, label: str = "") -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:8px;">'
        f'<span class="pulse-dot" style="background:{color};'
        f'box-shadow:0 0 8px {color};"></span>'
        f'<span style="font-family:\'Space Mono\',monospace;font-size:11px;'
        f'color:{COLORS["text_dim"]};">{label}</span></span>'
    )


# ── Progress row (for recommendations risk breakdown) ────────────────────────
def progress_row(label: str, value: int, max_val: int, color: str):
    pct = int((value / max_val) * 100)
    st.markdown(f"""
<div style="margin-bottom:14px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
    <span style="font-size:12px;color:{COLORS['text_dim']};">{label}</span>
    <span style="font-size:12px;color:{color};font-family:'JetBrains Mono',monospace;">
      {value} item{'s' if value != 1 else ''}
    </span>
  </div>
  <div style="height:4px;background:{COLORS['border']};border-radius:2px;overflow:hidden;">
    <div style="height:100%;border-radius:2px;background:{color};width:{pct}%;
                box-shadow:0 0 10px {color};transition:width 0.8s ease;"></div>
  </div>
</div>""", unsafe_allow_html=True)