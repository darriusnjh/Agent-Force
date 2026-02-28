# ── app.py ─────────────────────────────────────────────────────────────────────
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import COLORS, GLOBAL_CSS

# 1. Page Config MUST be here at the very top
st.set_page_config(
    page_title="Agent-Force · AI Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject the Top Navigation CSS
C = COLORS
st.markdown(f"""
<style>
/* Hide Streamlit chrome and default sidebar to use our custom nav */
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stSidebarNav"] { display:none !important; }

/* Layout Fixes for Gap */
.block-container {{ 
    padding-top: 4.5rem !important; 
    padding-bottom: 3rem !important; 
    padding-left: 3rem !important; 
    padding-right: 3rem !important; 
    max-width: 1300px !important; 
    margin: 0 auto !important; 
}}
.stMainBlockContainer {{ 
    padding-top: 4.5rem !important;
    padding-left: 3rem !important; 
    padding-right: 3rem !important; 
    max-width: 1300px !important; 
}}

/* TOP NAV BAR CSS */
.af-nav {{
    position: fixed !important;  
    top: 0; left: 0; width: 100vw;                
    z-index: 999999 !important; 
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; height: 62px;
    background: rgba(8,11,20,0.95);
    border-bottom: 1px solid #1E2D45;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}}
.af-brand {{ display: flex; align-items: center; gap: 11px; text-decoration: none; }}
.af-nav-links {{ display: flex; align-items: center; gap: 2px; background: rgba(30,45,69,0.5); border: 1px solid {C['border']}; border-radius: 10px; padding: 4px; }}
.af-navbtn {{
    display: flex; align-items: center; gap: 7px; padding: 7px 16px; border-radius: 7px;
    font-family: 'Space Mono', monospace; font-size: 10px; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase; color: {C['text_dim']};
    background: transparent; border: 1px solid transparent; cursor: pointer;
    transition: 0.18s; text-decoration: none; position: relative; overflow: hidden;
}}
.af-navbtn:hover {{ color: {C['accent']}; border-color: {C['accent']}44; box-shadow: 0 0 16px {C['accent']}18; }}
.af-navbtn.active {{ color: {C['accent']}; background: {C['accent']}16; border-color: {C['accent']}55; box-shadow: 0 0 20px {C['accent']}20, inset 0 1px 0 {C['accent']}30; }}
.af-navbtn svg {{ transition: transform 0.18s; position:relative; z-index:1; }}
.af-navbtn span {{ position:relative; z-index:1; }}
.af-navbtn:hover svg {{ transform: scale(1.12); }}

/* Pulse dot animation for API status */
@keyframes ping {{ 0% {{ transform: scale(1); opacity: 1; }} 75%, 100% {{ transform: scale(2.5); opacity: 0; }} }}
.pulse-dot {{ position: relative; width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.pulse-dot::after {{ content: ''; position: absolute; inset: 0; border-radius: 50%; background: inherit; animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite; }}
</style>
""", unsafe_allow_html=True)

# 3. Load the rest of your global CSS from config.py
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# 4. Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "eval_done" not in st.session_state:
    st.session_state.eval_done = False
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False

# 5. Define the pages and explicitly set their URL paths
pages = [
    st.Page("pages/1_overview.py", title="Overview", url_path="overview"),
    st.Page("pages/2_evaluation.py", title="Evaluation", url_path="evaluation"),
    st.Page("pages/3_results.py", title="Results", url_path="results"),
    st.Page("pages/4_recommendations.py", title="Recommendations", url_path="recommendations")
]

# 6. Run the router
pg = st.navigation(pages)
pg.run()
