# ── Agent-Force Dashboard Config ──────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"

# ── Color tokens ──────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#080B14",
    "surface":  "#0D1220",
    "panel":    "#111827",
    "border":   "#1E2D45",
    "accent":   "#00D4FF",
    "accent2":  "#7B61FF",
    "accent3":  "#00FFB2",
    "warn":     "#FF6B35",
    "danger":   "#FF3B5C",
    "safe":     "#00FFB2",
    "muted":    "#4A5568",
    "text":     "#E2E8F0",
    "text_dim": "#718096",
}

# ── Severity / priority mappings ───────────────────────────────────────────────
SEVERITY_COLORS = {
    "critical": COLORS["danger"],
    "high":     COLORS["warn"],
    "medium":   "#FFD166",
    "low":      COLORS["safe"],
}

PRIORITY_COLORS = {
    "P0": COLORS["danger"],
    "P1": COLORS["warn"],
    "P2": "#FFD166",
    "P3": COLORS["safe"],
}

def score_color(score: int) -> str:
    if score >= 80:
        return COLORS["safe"]
    elif score >= 50:
        return "#FFD166"
    return COLORS["danger"]

# ── SVG shield logo (used as favicon + brand mark) ────────────────────────────
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7B61FF"/>
      <stop offset="100%" stop-color="#00D4FF"/>
    </linearGradient>
  </defs>
  <path d="M20 2 L36 9 L36 22 C36 30 28 37 20 39 C12 37 4 30 4 22 L4 9 Z"
        fill="url(#g)" opacity="0.95"/>
  <path d="M20 7 L31 12 L31 22 C31 28 26 33 20 35 C14 33 9 28 9 22 L9 12 Z"
        fill="#080B14" opacity="0.5"/>
  <path d="M14 20 L18 24 L26 16" stroke="#00FFB2" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

# ── Navigation items (used in top nav bar) ────────────────────────────────────
NAV_ITEMS = [
    {"id": "overview",        "label": "Overview",        "icon": "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"},
    {"id": "evaluation",      "label": "Evaluation",      "icon": "M13 10V3L4 14h7v7l9-11h-7z"},
    {"id": "results",         "label": "Results",         "icon": "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"},
    {"id": "history",         "label": "History",         "icon": "M12 8v4l3 3M3.05 11a9 9 0 1 1 .5 4m-.5-4H7"},
]

# ── Global CSS injected once in app.py ────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

/* ── LAYOUT FIXES FOR WIDE SCREENS ── */
.block-container {{
    max-width: 1300px !important;     /* Stops it from stretching too wide */
    margin: 0 auto !important;        /* Keeps it perfectly centered */
    padding-top: 5rem !important;     /* Pushes content down so TopNav doesn't cover it */
    padding-bottom: 3rem !important;
}}

[data-testid="stHeader"] {{
    display: none !important;         /* Hides default Streamlit header */
}}
/* ─────────────────────────────────── */

html, body, [data-testid="stAppViewContainer"] {{
    background-color: {COLORS['bg']} !important;
    font-family: 'Inter', sans-serif;
    color: {COLORS['text']};
}}

[data-testid="stSidebar"] {{
    background-color: {COLORS['surface']} !important;
    border-right: 1px solid {COLORS['border']};
}}

[data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

/* Hide default streamlit header / footer / sidebar nav */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebarNav"] {{ display: none !important; }}

/* Top nav bar */
.af-topnav {{
    position: sticky;
    top: 0;
    z-index: 999;
    background: {COLORS['surface']}EE;
    border-bottom: 1px solid {COLORS['border']};
    backdrop-filter: blur(16px);
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    margin-bottom: 32px;
}}

.af-nav-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.af-nav-links {{
    display: flex;
    gap: 4px;
    align-items: center;
}}

.af-nav-link {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    text-decoration: none;
    color: {COLORS['text_dim']};
    border: 1px solid transparent;
    transition: all 0.18s;
}}

.af-nav-link:hover {{
    background: {COLORS['accent']}12;
    color: {COLORS['accent']};
    border-color: {COLORS['accent']}33;
}}

.af-nav-link.active {{
    background: {COLORS['accent']}18;
    color: {COLORS['accent']};
    border-color: {COLORS['accent']}44;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}}

/* Streamlit tabs — hidden (we use page nav) */
[data-baseweb="tab-list"] {{
    background: {COLORS['surface']} !important;
    border-radius: 10px;
    border: 1px solid {COLORS['border']};
    padding: 4px;
    gap: 4px;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    color: {COLORS['text_dim']} !important;
    text-transform: uppercase;
    border: none !important;
    padding: 8px 20px !important;
}}
[aria-selected="true"] {{
    background: {COLORS['accent']}18 !important;
    color: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']}44 !important;
}}

/* Buttons */
.stButton > button {{
    background: {COLORS['accent']}18 !important;
    color: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']}55 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: {COLORS['accent']}30 !important;
    border-color: {COLORS['accent']} !important;
}}

/* Progress bar */
.stProgress > div > div {{
    background: linear-gradient(90deg, {COLORS['accent2']}, {COLORS['accent']}) !important;
    box-shadow: 0 0 12px {COLORS['accent']} !important;
}}
[data-testid="stProgressBar"] > div {{
    background: {COLORS['border']} !important;
    border-radius: 4px !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {COLORS['surface']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
}}

/* Text / select inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div {{
    background: {COLORS['surface']} !important;
    border-color: {COLORS['border']} !important;
    color: {COLORS['text']} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-baseweb="select"] > div {{
    background: {COLORS['surface']} !important;
    border-color: {COLORS['border']} !important;
}}

hr {{ border-color: {COLORS['border']} !important; margin: 24px 0 !important; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 2px; }}

/* Pulsing dot */
@keyframes ping {{
    0%, 100% {{ transform: scale(1); opacity: 0.6; }}
    50%       {{ transform: scale(1.8); opacity: 0; }}
}}
.pulse-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: ping 1.4s ease-in-out infinite;
}}
</style>
"""
