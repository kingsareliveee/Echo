# ==============================================================================
# Echo Shield — AI/ML-Enabled Adaptive Noise Cancellation Dashboard
# Project: Echo Shield | PS ID: SIH26052 | DRDO | Theme: Smart Vehicles
# ==============================================================================

import os
import io
import glob
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Echo Shield — ANC Defence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(APP_DIR, "data")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
MODEL_PATH  = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

# --------------------------------------------------------------------------
# Audio Catalog Loaders (Cached for high performance & instant navigation)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_speech_catalog():
    """Load all CMU ARCTIC clean speech files grouped by speaker."""
    clean_dir = os.path.join(DATA_DIR, "clean_speech")
    catalog = {}
    if os.path.isdir(clean_dir):
        for spk in sorted(os.listdir(clean_dir)):
            spk_path = os.path.join(clean_dir, spk)
            if os.path.isdir(spk_path):
                files = sorted(glob.glob(os.path.join(spk_path, "*.wav")))
                if files:
                    catalog[spk] = files
    return catalog


@st.cache_data(show_spinner=False)
def get_step23_metadata():
    """Load Step-23 generated speech-in-noise dataset metadata."""
    meta_path = os.path.join(DATA_DIR, "speech_dataset", "metadata.csv")
    if os.path.isfile(meta_path):
        try:
            return pd.read_csv(meta_path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def get_esc50_metadata():
    """Load ESC-50 dataset metadata."""
    meta_path = os.path.join(DATA_DIR, "ESC-50-master", "meta", "esc50.csv")
    if os.path.isfile(meta_path):
        try:
            return pd.read_csv(meta_path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def get_all_audio_files():
    """Scan and index all WAV files in the data directory."""
    wavs = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
    return wavs


# Badge color taxonomy — precision scientific palette
BADGE_COLORS = {
    "STATIONARY":     ("#0284C7", "#F0F9FF"),
    "NON_STATIONARY": ("#D97706", "#FFFBEB"),
    "IMPULSIVE":      ("#DC2626", "#FEF2F2"),
    "NOISY_SPEECH":   ("#7C3AED", "#F5F3FF"),
    "CLEAN_SPEECH":   ("#16A34A", "#F0FDF4"),
    "TEST":           ("#475569", "#F8FAFC"),
    "ENVIRONMENTAL":  ("#0D9488", "#F0FDFA"),
}

# --------------------------------------------------------------------------
# Import utility module (wraps Steps 01-20)
# --------------------------------------------------------------------------
try:
    import utils
    _UTILS_OK = True
except ImportError as _e:
    _UTILS_OK = False
    _UTILS_ERR = str(_e)

# --------------------------------------------------------------------------
# Scientific White-First Design System — Calibrated Hierarchy
# --------------------------------------------------------------------------
PALETTE = {
    # Backgrounds
    "bg"          : "#FFFFFF",
    "bg_secondary": "#F8FAFC",
    "bg_tertiary" : "#F1F5F9",
    # Typography & Hierarchy
    "navy"        : "#0F172A",
    "navy_mid"    : "#1E3A8A",
    "text"        : "#0F172A",
    "text_dim"    : "#334155",
    "text_muted"  : "#64748B",
    # Semantic Accents
    "blue"        : "#0284C7",
    "blue_light"  : "#F0F9FF",
    "blue_border" : "#BAE6FD",
    "teal"        : "#0D9488",
    "teal_light"  : "#F0FDFA",
    "teal_border" : "#99F6E4",
    # Status & States
    "green"       : "#16A34A",
    "green_light" : "#F0FDF4",
    "green_border": "#BBF7D0",
    "amber"       : "#D97706",
    "amber_light" : "#FFFBEB",
    "amber_border": "#FDE68A",
    "red"         : "#DC2626",
    "red_light"   : "#FEF2F2",
    "red_border"  : "#FECACA",
    "purple"      : "#7C3AED",
    "purple_light": "#F5F3FF",
    "purple_border": "#DDD6FE",
    # Surfaces & Grid
    "border"      : "#E2E8F0",
    "border_mid"  : "#CBD5E1",
    # Chart Signal Colors
    "waveform"    : "#0284C7",
    "fft"         : "#DC2626",
    "anc"         : "#0D9488",
    "wavelet"     : "#7C3AED",
    "chart_grid"  : "#F1F5F9",
}

# --------------------------------------------------------------------------
# Global CSS — Scientific Instrumentation Theme
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;1,14..32,400&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

/* === RESET & BASE === */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    font-size: 14.5px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background-color: {PALETTE["bg"]}; }}
.main .block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* === SCROLLBAR === */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {PALETTE["bg_secondary"]}; }}
::-webkit-scrollbar-thumb {{ background: {PALETTE["border_mid"]}; border-radius: 3px; }}

/* === HEADER === */
.es-header {{
    background: {PALETTE["bg"]};
    border-bottom: 2px solid {PALETTE["border"]};
    padding: 16px 0 16px;
    margin-bottom: 0px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
}}
.es-header-mark {{
    width: 4px;
    background: {PALETTE["blue"]};
    border-radius: 2px;
    align-self: stretch;
    min-height: 56px;
}}
.es-title {{
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {PALETTE["navy"]};
    margin: 0;
    line-height: 1.15;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}}
.es-subtitle {{
    font-size: 0.76rem;
    color: {PALETTE["blue"]};
    font-weight: 700;
    margin-top: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}
.es-tagline {{
    font-size: 0.72rem;
    color: {PALETTE["text_muted"]};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
    letter-spacing: 0.03em;
}}
.es-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}}
.es-badge {{
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
}}
.es-badge-blue {{
    background: {PALETTE["blue_light"]};
    border: 1px solid {PALETTE["blue_border"]};
    color: {PALETTE["blue"]};
}}
.es-badge-teal {{
    background: {PALETTE["teal_light"]};
    border: 1px solid {PALETTE["teal_border"]};
    color: {PALETTE["teal"]};
}}
.es-badge-purple {{
    background: {PALETTE["purple_light"]};
    border: 1px solid {PALETTE["purple_border"]};
    color: {PALETTE["purple"]};
}}
.es-badge-slate {{
    background: {PALETTE["bg_secondary"]};
    border: 1px solid {PALETTE["border_mid"]};
    color: {PALETTE["text_dim"]};
}}

/* === NAVIGATION === */
.stRadio > div {{
    background: {PALETTE["bg_secondary"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 4px;
    gap: 4px;
}}
.stRadio [data-baseweb="radio"] label {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: {PALETTE["text_dim"]} !important;
    padding: 7px 16px !important;
    border-radius: 4px !important;
    transition: all 0.15s ease;
    letter-spacing: 0.01em;
    cursor: pointer;
    border: 1px solid transparent !important;
}}
.stRadio [data-baseweb="radio"] label:hover {{
    color: {PALETTE["navy"]} !important;
    background: {PALETTE["bg_tertiary"]} !important;
}}
.stRadio [data-baseweb="radio"] [aria-checked="true"] ~ label,
.stRadio [data-baseweb="radio"] label:has(+ div[data-checked="true"]) {{
    color: {PALETTE["blue"]} !important;
    background: {PALETTE["bg"]} !important;
    border: 1px solid {PALETTE["border_mid"]} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-weight: 700 !important;
}}
/* Hide radio circles */
.stRadio [data-baseweb="radio"] div:first-child {{ display: none !important; }}

/* === SIDEBAR === */
[data-testid="stSidebar"] {{
    background-color: {PALETTE["bg_secondary"]};
    border-right: 1px solid {PALETTE["border"]};
}}
[data-testid="stSidebar"] * {{ color: {PALETTE["text"]} !important; }}
[data-testid="stSidebar"] .stSelectbox label {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: {PALETTE["text_muted"]} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}}
[data-testid="stSidebar"] .stButton > button {{
    background: {PALETTE["blue"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em;
    padding: 10px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {PALETTE["navy_mid"]} !important;
}}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {PALETTE["bg_secondary"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 3px;
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 7px 16px !important;
    color: {PALETTE["text_dim"]} !important;
    border-radius: 4px !important;
    letter-spacing: 0.01em;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE["navy"]} !important;
    background: {PALETTE["bg"]} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    font-weight: 700 !important;
}}

/* === STATUS GRID === */
.status-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
    margin: 16px 0 24px;
}}
.status-card {{
    background: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 16px 18px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}}
.status-card:hover {{
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}}
.status-card-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {PALETTE["text_muted"]};
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.status-card-value {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {PALETTE["navy"]};
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    letter-spacing: -0.01em;
}}
.status-card-sub {{
    font-size: 0.72rem;
    color: {PALETTE["text_muted"]};
    margin-top: 5px;
    font-weight: 500;
}}

/* === SECTION HEADERS === */
.es-section {{
    border-top: 1px solid {PALETTE["border"]};
    margin-top: 28px;
    padding-top: 20px;
}}
.es-section-title {{
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {PALETTE["navy_mid"]};
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
}}
.es-section-title::before {{
    content: '';
    display: inline-block;
    width: 3px;
    height: 13px;
    background: {PALETTE["blue"]};
    border-radius: 2px;
}}
.es-section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {PALETTE["border"]};
}}

/* === METRIC ROWS === */
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 9px 0;
    border-bottom: 1px solid {PALETTE["border"]};
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-name {{
    font-size: 0.86rem;
    color: {PALETTE["text_dim"]};
    font-weight: 500;
}}
.metric-value {{
    font-size: 0.94rem;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    color: {PALETTE["navy"]};
    font-weight: 600;
    text-align: right;
}}

/* === TECHNICAL CARD === */
.tech-card {{
    background: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}}
.tech-card-title {{
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE["navy"]};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid {PALETTE["border"]};
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* === STATUS PILLS === */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.pill-measured {{
    background: {PALETTE["teal_light"]};
    color: {PALETTE["teal"]};
    border: 1px solid {PALETTE["teal_border"]};
}}
.pill-simulation {{
    background: {PALETTE["amber_light"]};
    color: {PALETTE["amber"]};
    border: 1px solid {PALETTE["amber_border"]};
}}
.pill-model {{
    background: {PALETTE["blue_light"]};
    color: {PALETTE["blue"]};
    border: 1px solid {PALETTE["blue_border"]};
}}
.pill-pending {{
    background: {PALETTE["bg_secondary"]};
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border_mid"]};
}}

/* === STATUS INDICATOR DOTS === */
.status-dot-active {{
    display: inline-block;
    width: 7px; height: 7px;
    background: {PALETTE["teal"]};
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}}
.status-dot-pending {{
    display: inline-block;
    width: 7px; height: 7px;
    background: {PALETTE["amber"]};
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}}
.status-dot-offline {{
    display: inline-block;
    width: 7px; height: 7px;
    background: {PALETTE["border_mid"]};
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}}

/* === STREAMLIT OVERRIDES === */
.stButton > button {{
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em;
    padding: 9px 20px !important;
    border: 1px solid {PALETTE["border"]} !important;
    background: {PALETTE["bg"]} !important;
    color: {PALETTE["navy"]} !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    background: {PALETTE["bg_secondary"]} !important;
    border-color: {PALETTE["border_mid"]} !important;
}}
.stButton [kind="primary"] > button,
.stButton > button[data-testid="baseButton-primary"] {{
    background: {PALETTE["blue"]} !important;
    color: white !important;
    border-color: {PALETTE["blue"]} !important;
    box-shadow: 0 1px 3px rgba(2,132,199,0.2) !important;
}}
.stButton > button[data-testid="baseButton-primary"]:hover {{
    background: {PALETTE["navy_mid"]} !important;
    border-color: {PALETTE["navy_mid"]} !important;
}}
.stSelectbox > div > div {{
    border: 1px solid {PALETTE["border"]} !important;
    border-radius: 4px !important;
    background: {PALETTE["bg"]} !important;
    font-size: 0.85rem !important;
    color: {PALETTE["text"]} !important;
    box-shadow: none !important;
}}
.stTextInput > div > div > input {{
    border: 1px solid {PALETTE["border"]} !important;
    border-radius: 4px !important;
    background: {PALETTE["bg"]} !important;
    font-size: 0.85rem !important;
    color: {PALETTE["text"]} !important;
    box-shadow: none !important;
    padding: 8px 12px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {PALETTE["blue"]} !important;
    box-shadow: 0 0 0 2px rgba(2,132,199,0.15) !important;
}}
/* Dataframe */
.stDataFrame {{
    border: 1px solid {PALETTE["border"]} !important;
    border-radius: 6px !important;
    overflow: hidden;
}}
.stDataFrame iframe {{
    border: none !important;
}}
/* Metric widget */
[data-testid="stMetric"] {{
    background: {PALETTE["bg_secondary"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {PALETTE["text_muted"]} !important;
    font-family: 'JetBrains Mono', monospace;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: {PALETTE["navy"]} !important;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
}}
/* Alerts */
[data-testid="stAlert"] {{
    border-radius: 6px !important;
    border: 1px solid !important;
    font-size: 0.85rem !important;
}}
/* Spinner */
.stSpinner > div {{ color: {PALETTE["blue"]} !important; }}
/* Caption */
.stCaption {{ color: {PALETTE["text_muted"]} !important; font-size: 0.78rem !important; }}
/* Audio player */
audio {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    width: 100%;
}}
/* Divider */
hr {{ border-color: {PALETTE["border"]} !important; margin: 20px 0 !important; }}

/* === PIPELINE DIAGRAM === */
.pipeline-container {{
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 16px 0;
}}
.pipeline-node {{
    background: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border_mid"]};
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: {PALETTE["navy"]};
    text-align: center;
    white-space: nowrap;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.pipeline-node.active {{
    background: {PALETTE["blue_light"]};
    border-color: {PALETTE["blue"]};
    color: {PALETTE["blue"]};
}}
.pipeline-arrow {{
    color: {PALETTE["border_mid"]};
    font-size: 1rem;
    padding: 0 6px;
    flex-shrink: 0;
}}

/* === PROGRESS BAR === */
.prog-bar-bg {{
    background: {PALETTE["bg_secondary"]};
    border-radius: 2px;
    height: 4px;
    overflow: hidden;
    margin-top: 6px;
}}
.prog-bar-fill {{
    height: 100%;
    border-radius: 2px;
    background: {PALETTE["blue"]};
}}

/* === AUDIO COMPARISON CARDS === */
.audio-compare-card {{
    background: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    overflow: hidden;
}}
.audio-compare-header {{
    padding: 10px 16px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid {PALETTE["border"]};
    background: {PALETTE["bg_secondary"]};
}}
.audio-compare-body {{
    padding: 14px 16px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Matplotlib Scientific Plot Theme (White Background)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor"  : PALETTE["bg"],
    "axes.facecolor"    : PALETTE["bg"],
    "axes.edgecolor"    : PALETTE["border_mid"],
    "axes.labelcolor"   : PALETTE["text_dim"],
    "xtick.color"       : PALETTE["text_muted"],
    "ytick.color"       : PALETTE["text_muted"],
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "text.color"        : PALETTE["text"],
    "grid.color"        : PALETTE["chart_grid"],
    "grid.linestyle"    : "-",
    "grid.alpha"        : 1.0,
    "grid.linewidth"    : 0.5,
    "font.family"       : "sans-serif",
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.titlecolor"   : PALETTE["navy"],
    "axes.labelsize"    : 9.5,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.8,
    "figure.dpi"        : 150,
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.9,
    "legend.edgecolor"  : PALETTE["border"],
    "legend.facecolor"  : PALETTE["bg"],
})

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=PALETTE["bg"], edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def _downsample_for_plot(arr: np.ndarray, max_pts: int = 10_000) -> np.ndarray:
    if len(arr) <= max_pts:
        return arr
    step = max(1, len(arr) // max_pts)
    return arr[::step]

# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "run_done" not in st.session_state:
    st.session_state["run_done"] = False
if "active_wav" not in st.session_state:
    st.session_state["active_wav"] = os.path.join(DATA_DIR, "tank.wav")
if "active_label" not in st.session_state:
    st.session_state["active_label"] = "tank.wav [STATIONARY]"
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Live ANC Studio"

# ==========================================================================
# § HEADER
# ==========================================================================
st.markdown("""
<div class="es-header">
  <div class="es-header-mark"></div>
  <div>
    <p class="es-title">Echo Shield</p>
    <p class="es-subtitle">AI/ML-Enabled Adaptive Noise Cancellation · Defence Vehicles</p>
    <p class="es-tagline">DRDO · SIH26052 · Smart Vehicles · SILENCE THE NOISE. AMPLIFY THE MISSION.</p>
    <div class="es-badge-row">
      <span class="es-badge es-badge-blue">SIH 2026</span>
      <span class="es-badge es-badge-blue">DRDO</span>
      <span class="es-badge es-badge-blue">PS ID: SIH26052</span>
      <span class="es-badge es-badge-slate">Software Prototype</span>
      <span class="es-badge es-badge-slate">15,391 Audio Files</span>
      <span class="es-badge es-badge-purple">DT Classifier · 98.97% Acc</span>
      <span class="es-badge es-badge-teal">NLMS · 31.58 dB</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# § TOP NAVIGATION BAR
# ==========================================================================
nav_options = [
    "Live ANC Studio",
    "Audio & Voice Library",
    "AI Classifier & Benchmarks",
    "DSP & Wavelet Lab",
    "Hardware & Roadmap",
]

# Map legacy session state values
_nav_map = {
    "🎛️ Live ANC Studio": "Live ANC Studio",
    "🎙️ Voice & Audio Explorer (15,000+ Library)": "Audio & Voice Library",
    "📊 AI Classifier & Benchmarks": "AI Classifier & Benchmarks",
    "🔬 DSP & Wavelet Lab": "DSP & Wavelet Lab",
    "📋 Hardware & Roadmap": "Hardware & Roadmap",
}
if st.session_state["nav_page"] in _nav_map:
    st.session_state["nav_page"] = _nav_map[st.session_state["nav_page"]]

st.markdown('<div style="margin-top: 12px; margin-bottom: 4px;">', unsafe_allow_html=True)
selected_nav = st.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["nav_page"] = selected_nav
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div style="border-top: 1px solid {PALETTE["border"]}; margin-bottom: 24px;"></div>', unsafe_allow_html=True)

# ==========================================================================
# § SIDEBAR — Comprehensive Audio Selector
# ==========================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 16px 0 4px;">
      <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
                  color: {PALETTE['text_muted']}; text-transform: uppercase;
                  font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">
        Input Audio Selector
      </div>
      <div style="font-size: 0.78rem; color: {PALETTE['text_dim']}; margin-top: 4px;">
        15,391-file acoustic database
      </div>
    </div>
    """, unsafe_allow_html=True)

    collection_mode = st.selectbox(
        "Audio Collection",
        [
            "Defence Vehicle & Threat Noise (Core)",
            "Clean Human Speech (CMU ARCTIC — 2,317 Voices)",
            "Speech + Defence Noise (Step-23 — 11,000+ Pairs)",
            "Environmental & Vehicle Audio (ESC-50 — 2,000 Sounds)",
            "Search & Explore All Audio Files",
            "Upload Custom WAV File",
        ],
        index=0,
    )

    wav_path_selected = None
    selected_display_label = None
    badge_label = "AUDIO"

    # 1. CORE DEFENCE NOISE
    if "Defence Vehicle" in collection_mode:
        core_files = [
            ("aircraft_carrier_deck.wav  [NON-STATIONARY]", os.path.join(DATA_DIR, "defence_noise", "aircraft_carrier_deck.wav"), "NON_STATIONARY"),
            ("jet_fighter_supersonic.wav  [NON-STATIONARY]", os.path.join(DATA_DIR, "defence_noise", "jet_fighter_supersonic.wav"), "NON_STATIONARY"),
            ("apc_armored_diesel.wav  [STATIONARY]", os.path.join(DATA_DIR, "defence_noise", "apc_armored_diesel.wav"), "STATIONARY"),
            ("naval_destroyer_turbine.wav  [STATIONARY]", os.path.join(DATA_DIR, "defence_noise", "naval_destroyer_turbine.wav"), "STATIONARY"),
            ("artillery_cannon_battery.wav  [IMPULSIVE]", os.path.join(DATA_DIR, "defence_noise", "artillery_cannon_battery.wav"), "IMPULSIVE"),
            ("tactical_cockpit_cabin.wav  [STATIONARY]", os.path.join(DATA_DIR, "defence_noise", "tactical_cockpit_cabin.wav"), "STATIONARY"),
            ("tank.wav  [STATIONARY]", os.path.join(DATA_DIR, "tank.wav"), "STATIONARY"),
            ("heli.wav  [NON-STATIONARY]", os.path.join(DATA_DIR, "heli.wav"), "NON_STATIONARY"),
            ("gun_fight.wav  [IMPULSIVE]", os.path.join(DATA_DIR, "gun_fight.wav"), "IMPULSIVE"),
            ("test.wav  [TEST]", os.path.join(DATA_DIR, "test.wav"), "TEST"),
        ]
        valid_core = [x for x in core_files if os.path.isfile(x[1])]
        choice = st.selectbox("Select Defence Audio", [lbl for lbl, _, _ in valid_core])
        matched = next(x for x in valid_core if x[0] == choice)
        wav_path_selected = matched[1]
        selected_display_label = matched[0]
        badge_label = matched[2]

    # 2. CLEAN HUMAN SPEECH (CMU ARCTIC)
    elif "Clean Human Speech" in collection_mode:
        clean_catalog = get_clean_speech_catalog()
        speaker_map = {
            "cmu_us_ahw_arctic": "ahw — Male · 593 Sentences",
            "cmu_us_fem_arctic": "fem — Female · 593 Sentences",
            "cmu_us_bdl_arctic": "bdl — Male · 1,131 Sentences",
        }
        spk_keys = list(clean_catalog.keys())
        spk_choice = st.selectbox(
            "Select Speaker",
            spk_keys,
            format_func=lambda k: speaker_map.get(k, k),
        )
        file_list = clean_catalog.get(spk_choice, [])
        if file_list:
            sentence_names = [os.path.basename(f) for f in file_list]
            selected_file_name = st.selectbox("Select Sentence", sentence_names, index=0)
            wav_path_selected = os.path.join(DATA_DIR, "clean_speech", spk_choice, selected_file_name)
            selected_display_label = f"{spk_choice} — {selected_file_name}"
            badge_label = "CLEAN_SPEECH"

    # 3. SPEECH + NOISE MIX (STEP-23)
    elif "Speech + Defence Noise" in collection_mode:
        df_s23 = get_step23_metadata()
        if df_s23 is not None and not df_s23.empty:
            c_split, c_kind = st.columns(2)
            with c_split:
                split_sel = st.selectbox("Split", ["test", "val", "train"], index=0)
            with c_kind:
                kind_sel = st.selectbox("Type", ["Noisy (Mixture)", "Clean (Reference)"], index=0)

            c_noise, c_snr = st.columns(2)
            with c_noise:
                noise_opts = ["ALL"] + sorted(df_s23["noise_type"].unique().tolist())
                noise_sel = st.selectbox("Noise Type", noise_opts)
            with c_snr:
                snr_opts = ["ALL"] + sorted([f"{s:+.0f} dB" for s in df_s23["snr_db_requested"].unique()])
                snr_sel = st.selectbox("SNR Level", snr_opts)

            sub_df = df_s23[df_s23["split"] == split_sel]
            if noise_sel != "ALL":
                sub_df = sub_df[sub_df["noise_type"] == noise_sel]
            if snr_sel != "ALL":
                snr_val = float(snr_sel.replace(" dB", ""))
                sub_df = sub_df[sub_df["snr_db_requested"] == snr_val]

            st.caption(f"{len(sub_df):,} matching pairs")
            if not sub_df.empty:
                pair_labels = [
                    f"{row['pair_id']} | {row['noise_label']} | SNR: {row['snr_db_requested']:+.0f}dB"
                    for _, row in sub_df.head(200).iterrows()
                ]
                pair_choice = st.selectbox("Select Pair", pair_labels, index=0)
                sel_row = sub_df.iloc[pair_labels.index(pair_choice)]
                if "Noisy" in kind_sel:
                    wav_path_selected = os.path.join(DATA_DIR, "speech_dataset", split_sel, "noisy", sel_row["noisy_file"])
                    badge_label = "NOISY_SPEECH"
                else:
                    wav_path_selected = os.path.join(DATA_DIR, "speech_dataset", split_sel, "clean", sel_row["clean_file"])
                    badge_label = "CLEAN_SPEECH"
                selected_display_label = f"{sel_row['pair_id']} ({kind_sel})"

    # 4. ESC-50 ENVIRONMENTAL & VEHICLES
    elif "Environmental" in collection_mode:
        df_esc = get_esc50_metadata()
        esc_dir = os.path.join(DATA_DIR, "ESC-50-master", "audio")
        if df_esc is not None and not df_esc.empty:
            cat_group = st.selectbox(
                "Category Group",
                [
                    "Defence & Vehicle Proxies (engine, heli, siren, train, plane...)",
                    "Impulsive & Transient (fireworks, gun/horn, glass, knock...)",
                    "Weather & Environment (wind, rain, thunderstorm...)",
                    "All 50 ESC-50 Classes",
                ],
                index=0,
            )

            if "Defence" in cat_group:
                cats = ["engine", "helicopter", "siren", "airplane", "train", "chainsaw", "vacuum_cleaner", "washing_machine"]
            elif "Impulsive" in cat_group:
                cats = ["fireworks", "glass_breaking", "car_horn", "door_wood_knock"]
            elif "Weather" in cat_group:
                cats = ["wind", "rain", "thunderstorm"]
            else:
                cats = sorted(df_esc["category"].unique().tolist())

            cat_sel = st.selectbox("Noise Category", cats)
            cat_files = df_esc[df_esc["category"] == cat_sel]["filename"].tolist()
            file_sel = st.selectbox("Select Audio Take", [f"Take #{i+1} ({fn})" for i, fn in enumerate(cat_files)])
            fn = cat_files[[f"Take #{i+1} ({f})" for i, f in enumerate(cat_files)].index(file_sel)]
            wav_path_selected = os.path.join(esc_dir, fn)
            selected_display_label = f"ESC-50: {cat_sel} ({fn})"
            badge_label = "ENVIRONMENTAL"

    # 5. ALL AUDIO SEARCH & EXPLORER
    elif "Search" in collection_mode:
        all_wavs = get_all_audio_files()
        st.caption(f"{len(all_wavs):,} total WAV files indexed")
        query = st.text_input("Search filename / keyword", value="tank").strip().lower()
        matched_wavs = [w for w in all_wavs if query in os.path.basename(w).lower() or query in w.lower()]
        st.caption(f"{len(matched_wavs):,} matches")
        if matched_wavs:
            disp_opts = [f"{os.path.basename(w)}" for w in matched_wavs[:300]]
            chosen_disp = st.selectbox("Select Matched File", disp_opts, index=0)
            wav_path_selected = matched_wavs[disp_opts.index(chosen_disp)]
            selected_display_label = os.path.basename(wav_path_selected)
            badge_label = "LIBRARY_FILE"

    # 6. UPLOAD CUSTOM WAV
    else:
        uploaded_file = st.file_uploader("Upload WAV File", type=["wav"], label_visibility="collapsed")
        if uploaded_file is not None:
            _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _tmp.write(uploaded_file.read())
            _tmp.flush()
            wav_path_selected = _tmp.name
            selected_display_label = uploaded_file.name
            badge_label = "UPLOADED"

    # Save active selection
    if wav_path_selected and os.path.isfile(wav_path_selected):
        st.session_state["active_wav"] = wav_path_selected
        st.session_state["active_label"] = selected_display_label or os.path.basename(wav_path_selected)

        # Show badge & metadata
        bc, bg = BADGE_COLORS.get(badge_label, (PALETTE["blue"], PALETTE["blue_light"]))
        st.markdown(
            f'<div style="margin: 12px 0 4px;">'
            f'<span style="background:{bg};border:1px solid {bc};border-radius:3px;'
            f'padding:3px 10px;font-size:0.68rem;font-family:\'JetBrains Mono\',monospace;'
            f'color:{bc};font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">'
            f'● {badge_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{os.path.basename(wav_path_selected)}`")

        # Preview Player
        st.markdown(f'<div style="margin: 10px 0 4px; font-size: 0.68rem; font-weight: 700; '
                    f'letter-spacing: 0.1em; color: {PALETTE["text_muted"]}; '
                    f'text-transform: uppercase; font-family: \'JetBrains Mono\', monospace;">Preview</div>',
                    unsafe_allow_html=True)
        st.audio(wav_path_selected, format="audio/wav")

    st.markdown(f'<hr style="border-color:{PALETTE["border"]}; margin: 16px 0;">', unsafe_allow_html=True)

    # Run Pipeline Button
    st.markdown(f'<div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; '
                f'color: {PALETTE["text_muted"]}; text-transform: uppercase; '
                f'font-family: \'JetBrains Mono\', monospace; margin-bottom: 8px;">Pipeline Control</div>',
                unsafe_allow_html=True)
    run_btn = st.button("▶  Run Full ANC Pipeline", use_container_width=True, type="primary")

    st.markdown(f'<hr style="border-color:{PALETTE["border"]}; margin: 16px 0;">', unsafe_allow_html=True)

    # System status indicators
    st.markdown(f"""
    <div style="margin-top: 4px; display: flex; flex-direction: column; gap: 8px;">
      <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
                  color: {PALETTE['text_muted']}; text-transform: uppercase;
                  font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">System Status</div>
      <div style="font-size: 0.8rem; color: {PALETTE['text_dim']}; display: flex; align-items: center; gap: 6px;">
        <span class="status-dot-active"></span>
        <span>Software Prototype · Active</span>
      </div>
      <div style="font-size: 0.8rem; color: {PALETTE['text_muted']}; display: flex; align-items: center; gap: 6px;">
        <span class="status-dot-offline"></span>
        <span>ARM Deployment · Pending</span>
      </div>
      <div style="font-size: 0.78rem; color: {PALETTE['text_muted']}; margin-top: 4px; font-style: italic;">
        Fully local · No external network calls
      </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load Model
# --------------------------------------------------------------------------
if not _UTILS_OK:
    st.error(f"Could not import utils.py: {_UTILS_ERR}")
    st.stop()

@st.cache_resource(show_spinner=False)
def _load_model():
    return utils.load_model(MODEL_PATH)

model, model_err = _load_model()
if model_err:
    st.error(f"Classifier model not found or failed to load at `{MODEL_PATH}`.")
    st.stop()

# --------------------------------------------------------------------------
# Pipeline Execution
# --------------------------------------------------------------------------
PIPELINE_STEPS = [
    "Audio Loaded",
    "Features Extracted",
    "Noise Classified",
    "Profile Selected",
    "DSP Configured",
    "ANC Simulation Completed",
    "Results Ready",
]

if run_btn:
    active_path = st.session_state.get("active_wav")
    if not active_path or not os.path.isfile(active_path):
        st.error("Audio file not found. Please select a valid file from the sidebar.")
    else:
        st.session_state["run_done"] = False
        st.session_state["results"] = None

        prog_box = st.empty()
        stat_box = st.empty()

        def _render_steps(done_up_to: int):
            nodes = []
            for i, s in enumerate(PIPELINE_STEPS):
                if i < done_up_to:
                    nodes.append(f'<div class="pipeline-node active">✓ {s}</div>')
                else:
                    nodes.append(f'<div class="pipeline-node">{s}</div>')
                if i < len(PIPELINE_STEPS) - 1:
                    nodes.append('<div class="pipeline-arrow">→</div>')
            prog_box.markdown(
                f'<div style="background:{PALETTE["bg_secondary"]};border:1px solid {PALETTE["border"]};'
                f'border-radius:6px;padding:12px 16px;margin-bottom:16px;">'
                f'<div class="pipeline-container">' + "".join(nodes) + '</div></div>',
                unsafe_allow_html=True,
            )

        _render_steps(0)
        with st.spinner("Running AI classification & adaptive DSP pipeline…"):
            try:
                res = utils.run_full_pipeline(active_path, model, max_anc_samples=60_000)
                for step in range(1, len(PIPELINE_STEPS) + 1):
                    _render_steps(step)
                st.session_state["results"] = res
                st.session_state["run_done"] = True
                stat_box.success(f"Pipeline complete — `{st.session_state['active_label']}`")
            except Exception as exc:
                st.session_state["run_done"] = False
                stat_box.error(f"Pipeline error: {exc}")
                st.exception(exc)

res = st.session_state.get("results")

# ==============================================================================
# PAGE 1: LIVE ANC STUDIO
# ==============================================================================
if st.session_state["nav_page"] == "Live ANC Studio":

    if res:
        noise_type    = res["majority_class"].upper().replace("_", " ")
        noise_profile = res["profile_info"]["processing_profile"]
        anc_mode      = res["dsp_cfg"]["processing_mode"]
        lms_db        = f"{res['lms_db']:.2f} dB"
        nlms_db       = f"{res['nlms_db']:.2f} dB"
        fx_db         = f"{res['fx_db']:.2f} dB"
        filter_len    = str(res["dsp_cfg"]["filter_length"])
        step_size     = str(res["dsp_cfg"]["step_size_mu"])
    else:
        noise_type    = "—"
        noise_profile = "—"
        anc_mode      = "—"
        lms_db        = "—"
        nlms_db       = "—"
        fx_db         = "—"
        filter_len    = "—"
        step_size     = "—"

    # ---- Status Cards ----
    st.markdown(f"""
    <div class="status-grid">
      <div class="status-card" style="border-top: 3px solid {PALETTE['blue']};">
        <div class="status-card-label" style="color:{PALETTE['blue']};">Predicted Noise Class</div>
        <div class="status-card-value" style="font-size:1.15rem; color:{PALETTE['navy']};">{noise_type}</div>
        <div class="status-card-sub">AI Decision Tree · Majority Vote</div>
      </div>
      <div class="status-card" style="border-top: 3px solid {PALETTE['amber']};">
        <div class="status-card-label" style="color:{PALETTE['amber']};">DSP Noise Profile</div>
        <div class="status-card-value" style="font-size:0.95rem; color:{PALETTE['navy']};">{noise_profile}</div>
        <div class="status-card-sub">{anc_mode}</div>
      </div>
      <div class="status-card" style="border-top: 3px solid {PALETTE['teal']};">
        <div class="status-card-label" style="color:{PALETTE['teal']};">NLMS Reduction</div>
        <div class="status-card-value" style="color:{PALETTE['teal']};">{nlms_db}</div>
        <div class="status-card-sub">Measured · Software Experiment</div>
      </div>
      <div class="status-card" style="border-top: 3px solid {PALETTE['purple']};">
        <div class="status-card-label" style="color:{PALETTE['purple']};">FxLMS Reduction</div>
        <div class="status-card-value" style="color:{PALETTE['purple']}; font-size:1.25rem;">{fx_db}</div>
        <div class="status-card-sub">Simulation · Synthetic S̃(z)</div>
      </div>
      <div class="status-card" style="border-top: 3px solid {PALETTE['navy_mid']};">
        <div class="status-card-label" style="color:{PALETTE['navy_mid']};">Filter Taps / Step Size</div>
        <div class="status-card-value" style="font-size:1.15rem; color:{PALETTE['navy']};">{filter_len} / {step_size}</div>
        <div class="status-card-sub">Adaptive FIR Configuration</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not res:
        st.markdown(f"""
        <div style="background:{PALETTE['bg_secondary']};border:1px solid {PALETTE['border']};
                    border-radius:6px;padding:32px;text-align:center;margin-top:8px;">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.14em;color:{PALETTE['text_muted']};
                      text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:12px;">
            Awaiting Analysis
          </div>
          <div style="font-size:1rem;color:{PALETTE['text_dim']};">
            Select an audio file from the sidebar and click
            <strong style="color:{PALETTE['blue']};">▶ Run Full ANC Pipeline</strong>
            to begin acoustic analysis.
          </div>
          <div style="margin-top:16px;font-size:0.78rem;color:{PALETTE['text_muted']};">
            Pipeline: Audio Load → FFT → STFT → AI Classify → LMS → NLMS → FxLMS → Wavelet → Output
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ---- Acoustic Characterization ----
        st.markdown(f'<div class="es-section"><div class="es-section-title">Acoustic Characterization & AI Decision</div></div>', unsafe_allow_html=True)

        gf = res["global_feats"]
        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            # AI Classification Panel
            vote_pcts = res.get("vote_pcts", {})
            classes = res.get("model_classes", [])
            avg_probas = res.get("avg_probas")

            st.markdown(f"""
            <div class="tech-card">
              <div class="tech-card-title">Acoustic Feature Vector</div>
              <div class="metric-row">
                <span class="metric-name">RMS Energy</span>
                <span class="metric-value">{gf['rms']:.5f}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Zero Crossing Rate</span>
                <span class="metric-value">{gf['zcr']:.6f}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Spectral Centroid</span>
                <span class="metric-value">{gf['spectral_centroid']:.1f} Hz</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Spectral Rolloff (85%)</span>
                <span class="metric-value">{gf['spectral_rolloff']:.1f} Hz</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Duration</span>
                <span class="metric-value">{gf['duration_sec']:.3f} s</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Sample Rate</span>
                <span class="metric-value">{res['sample_rate']:,} Hz</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Channels</span>
                <span class="metric-value">{res['num_channels']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_f2:
            # Sub-band energy chart (scientific style)
            be = gf["band_energies"]
            band_names  = ["0–500 Hz\n(Low)", "500–2k Hz\n(Mid-Low)", "2k–5k Hz\n(Mid-High)", "5k–10k Hz\n(High)"]
            band_values = [be[k] for k in list(be.keys())]
            band_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"], PALETTE["wavelet"]]

            fig_be, ax_be = plt.subplots(figsize=(5.8, 3.2))
            bars_be = ax_be.barh(band_names, band_values, color=band_colors, height=0.5, edgecolor="none")
            for bar, val in zip(bars_be, band_values):
                ax_be.text(val + 0.005 * max(band_values, default=1),
                           bar.get_y() + bar.get_height() / 2,
                           f"{val:.4f}", va="center", fontsize=8.5, fontweight="600",
                           color=PALETTE["text_dim"])
            ax_be.set_xlabel("Spectral Magnitude Sum", fontsize=9)
            ax_be.set_title("Sub-Band Energy Distribution", fontsize=11, fontweight="bold",
                            color=PALETTE["navy"], pad=10)
            ax_be.grid(axis="x", linewidth=0.5)
            ax_be.set_axisbelow(True)
            fig_be.tight_layout(pad=1.2)
            st.image(_fig_to_bytes(fig_be), use_container_width=True)

            # Per-class vote percentages
            if vote_pcts:
                st.markdown(f"""
                <div class="tech-card" style="margin-top: 12px; padding: 14px 18px;">
                  <div class="tech-card-title" style="margin-bottom: 10px;">AI Classification · Window Vote Distribution</div>
                """, unsafe_allow_html=True)
                for cls, pct in sorted(vote_pcts.items(), key=lambda x: -x[1]):
                    bar_color = PALETTE["blue"] if cls == res["majority_class"] else PALETTE["border_mid"]
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                      <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
                        <span style="font-size:0.8rem; font-family:'JetBrains Mono',monospace; font-weight:600;
                                     color:{''+PALETTE['navy'] if cls==res['majority_class'] else PALETTE['text_muted']+''}">{cls.upper()}</span>
                        <span style="font-size:0.8rem; font-family:'JetBrains Mono',monospace; color:{PALETTE['text_dim']};">{pct:.1f}%</span>
                      </div>
                      <div class="prog-bar-bg">
                        <div class="prog-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ---- Signal Analysis Tabs ----
        st.markdown(f'<div class="es-section"><div class="es-section-title">Multi-Domain Signal Analysis</div></div>', unsafe_allow_html=True)

        tab_wv, tab_fft, tab_spec, tab_anc = st.tabs([
            "Waveform",
            "Frequency Spectrum (FFT)",
            "Spectrogram (STFT)",
            "ANC Residual  e[n]",
        ])

        signal       = res["signal"]
        sample_rate  = res["sample_rate"]
        duration_sec = res["duration_sec"]
        t_full       = np.linspace(0.0, duration_sec, len(signal), endpoint=False)
        t_ds         = _downsample_for_plot(t_full)
        sig_ds       = _downsample_for_plot(signal)

        with tab_wv:
            fig_wv, ax_wv = plt.subplots(figsize=(12, 3.5))
            ax_wv.plot(t_ds, sig_ds, color=PALETTE["waveform"], linewidth=0.8, alpha=0.95)
            ax_wv.axhline(0, color=PALETTE["border_mid"], linewidth=0.5)
            ax_wv.set_title(f"Waveform — {os.path.basename(st.session_state['active_wav'])}", pad=10)
            ax_wv.set_xlabel("Time (seconds)")
            ax_wv.set_ylabel("Normalized Amplitude")
            ax_wv.grid(True, linewidth=0.5)
            ax_wv.set_xlim(t_ds[0], t_ds[-1])
            ax_wv.fill_between(t_ds, sig_ds, 0, color=PALETTE["waveform"], alpha=0.08)
            fig_wv.tight_layout(pad=1.2)
            st.image(_fig_to_bytes(fig_wv), use_container_width=True)
            st.caption(f"Input audio · {res['num_samples']:,} samples · {duration_sec:.3f} s · {sample_rate:,} Hz")
            st.audio(st.session_state["active_wav"], format="audio/wav")

        with tab_fft:
            freq_ax = res["fft_freq_axis"]
            fft_mag = res["fft_magnitude"]
            fft_ds  = _downsample_for_plot(fft_mag, max_pts=6000)
            freq_ds = _downsample_for_plot(freq_ax, max_pts=6000)

            fig_fft, ax_fft = plt.subplots(figsize=(12, 3.5))
            ax_fft.plot(freq_ds, fft_ds, color=PALETTE["fft"], linewidth=0.8)
            ax_fft.axvline(res["dominant_freq"], color=PALETTE["blue"], linestyle="--",
                           linewidth=1.2, label=f"Dominant Peak: {res['dominant_freq']:.1f} Hz")
            ax_fft.fill_between(freq_ds, fft_ds, 0, color=PALETTE["fft"], alpha=0.07)
            ax_fft.set_title(f"FFT Frequency Spectrum — {os.path.basename(st.session_state['active_wav'])}", pad=10)
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Magnitude")
            ax_fft.legend(framealpha=0.95)
            ax_fft.grid(True, linewidth=0.5)
            ax_fft.set_xlim(0, sample_rate / 2)
            fig_fft.tight_layout(pad=1.2)
            st.image(_fig_to_bytes(fig_fft), use_container_width=True)
            st.caption(f"Dominant frequency: {res['dominant_freq']:.1f} Hz · Magnitude: {res['dominant_mag']:.5f}")

        with tab_spec:
            fig_sp, ax_sp = plt.subplots(figsize=(12, 4))
            # Use 'viridis' on white background — reads well in scientific reports
            mesh = ax_sp.pcolormesh(res["spec_times"], res["spec_freqs"],
                                    20 * np.log10(res["spec_mag"] + 1e-10),
                                    shading="auto", cmap="viridis",
                                    vmin=-80, vmax=0)
            ax_sp.set_ylim(0, sample_rate / 2.0)
            ax_sp.set_xlabel("Time (s)")
            ax_sp.set_ylabel("Frequency (Hz)")
            ax_sp.set_title(f"STFT Spectrogram (dB) — {os.path.basename(st.session_state['active_wav'])}", pad=10)
            cbar = fig_sp.colorbar(mesh, ax=ax_sp, pad=0.01)
            cbar.set_label("Power (dB)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            fig_sp.tight_layout(pad=1.2)
            st.image(_fig_to_bytes(fig_sp), use_container_width=True)
            st.caption("STFT: Hann window · NFFT=1024 · Overlap=512")

        with tab_anc:
            e_fx     = res["e_fxlms"]
            n_anc    = res["anc_samples_used"]
            t_anc    = np.linspace(0.0, n_anc / sample_rate, n_anc, endpoint=False)
            e_fx_ds  = _downsample_for_plot(e_fx)
            t_anc_ds = _downsample_for_plot(t_anc)
            sig_anc  = _downsample_for_plot(signal[:n_anc])

            fig_anc, (ax_a1, ax_a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                                      gridspec_kw={"hspace": 0.35})
            ax_a1.plot(t_anc_ds, sig_anc, color=PALETTE["waveform"], linewidth=0.8)
            ax_a1.fill_between(t_anc_ds, sig_anc, 0, color=PALETTE["waveform"], alpha=0.07)
            ax_a1.set_title("Primary Noise Input  d[n]", pad=8)
            ax_a1.set_ylabel("Amplitude")
            ax_a1.grid(True, linewidth=0.5)
            ax_a1.axhline(0, color=PALETTE["border_mid"], linewidth=0.5)

            ax_a2.plot(t_anc_ds, e_fx_ds, color=PALETTE["anc"], linewidth=0.8)
            ax_a2.fill_between(t_anc_ds, e_fx_ds, 0, color=PALETTE["anc"], alpha=0.08)
            ax_a2.set_title(f"FxLMS Residual Error  e[n] — Simulated Reduction: {res['fx_db']:.2f} dB", pad=8)
            ax_a2.set_xlabel("Time (seconds)")
            ax_a2.set_ylabel("Amplitude")
            ax_a2.grid(True, linewidth=0.5)
            ax_a2.axhline(0, color=PALETTE["border_mid"], linewidth=0.5)

            fig_anc.tight_layout(pad=1.2)
            st.image(_fig_to_bytes(fig_anc), use_container_width=True)

            if res.get("anc_capped"):
                st.caption(f"⚠ ANC simulation capped at {n_anc:,} samples ({n_anc/sample_rate:.2f}s) for demo speed. Full signal: {res['num_samples']:,} samples.")

            # 3-Way Audio Comparison
            st.markdown(f'<div style="margin: 20px 0 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; color: {PALETTE["text_muted"]}; text-transform: uppercase; font-family: \'JetBrains Mono\', monospace;">3-Stage Pipeline Audio Comparison</div>', unsafe_allow_html=True)

            c_bef, c_anc_col, c_ai = st.columns(3)

            with c_bef:
                st.markdown(f"""
                <div class="audio-compare-card">
                  <div class="audio-compare-header" style="border-left:3px solid {PALETTE['red']}; color:{PALETTE['red']};">
                    STAGE 1 · NOISE INPUT
                  </div>
                  <div class="audio-compare-body">
                """, unsafe_allow_html=True)
                st.audio(st.session_state["active_wav"], format="audio/wav")
                st.caption(f"Raw Input: {os.path.basename(st.session_state['active_wav'])}")
                st.markdown("</div></div>", unsafe_allow_html=True)

            with c_anc_col:
                anc_buf = io.BytesIO()
                anc_clip = np.clip(e_fx, -1.0, 1.0).astype(np.float32)
                sf.write(anc_buf, anc_clip, sample_rate, format="WAV", subtype="PCM_16")
                anc_buf.seek(0)
                st.markdown(f"""
                <div class="audio-compare-card">
                  <div class="audio-compare-header" style="border-left:3px solid {PALETTE['teal']}; color:{PALETTE['teal']};">
                    STAGE 2 · FxLMS ANC  ({res['fx_db']:.1f} dB)
                  </div>
                  <div class="audio-compare-body">
                """, unsafe_allow_html=True)
                st.audio(anc_buf.read(), format="audio/wav")
                st.caption("Acoustic anti-noise residual e[n] — Simulation")
                st.markdown("</div></div>", unsafe_allow_html=True)

            with c_ai:
                enh_speech = res.get("enhanced_speech")
                if enh_speech is None:
                    enh_speech = res["denoised"]
                ai_buf = io.BytesIO()
                ai_clip = np.clip(enh_speech, -1.0, 1.0).astype(np.float32)
                sf.write(ai_buf, ai_clip, sample_rate, format="WAV", subtype="PCM_16")
                ai_buf.seek(0)
                st.markdown(f"""
                <div class="audio-compare-card">
                  <div class="audio-compare-header" style="border-left:3px solid {PALETTE['wavelet']}; color:{PALETTE['wavelet']};">
                    STAGE 3 · AI ENHANCED OUTPUT
                  </div>
                  <div class="audio-compare-body">
                """, unsafe_allow_html=True)
                st.audio(ai_buf.read(), format="audio/wav")
                st.caption("Step-24 Wiener + Wavelet Speech Enhancement")
                st.markdown("</div></div>", unsafe_allow_html=True)

        # ---- DSP Configuration Summary ----
        st.markdown(f'<div class="es-section"><div class="es-section-title">DSP Configuration · Active Pipeline Parameters</div></div>', unsafe_allow_html=True)

        c_dsp_l, c_dsp_r = st.columns(2)
        with c_dsp_l:
            dsp = res["dsp_cfg"]
            st.markdown(f"""
            <div class="tech-card">
              <div class="tech-card-title">Adaptive Filter Configuration</div>
              <div class="metric-row">
                <span class="metric-name">Algorithm</span>
                <span class="metric-value">LMS / NLMS / FxLMS</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Profile</span>
                <span class="metric-value">{dsp['profile']}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Filter Length (Taps)</span>
                <span class="metric-value">{dsp['filter_length']}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Step Size μ</span>
                <span class="metric-value">{dsp['step_size_mu']}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Processing Mode</span>
                <span class="metric-value">{dsp['processing_mode']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c_dsp_r:
            wr = res
            st.markdown(f"""
            <div class="tech-card">
              <div class="tech-card-title">Measured Performance · Software Experiment</div>
              <div class="metric-row">
                <span class="metric-name">LMS RMS Reduction</span>
                <span class="metric-value" style="color:{PALETTE['blue']};">{wr['lms_db']:.2f} dB</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">NLMS RMS Reduction</span>
                <span class="metric-value" style="color:{PALETTE['teal']};">{wr['nlms_db']:.2f} dB</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">FxLMS Sim. Reduction</span>
                <span class="metric-value" style="color:{PALETTE['wavelet']};">{wr['fx_db']:.2f} dB</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Wavelet Threshold σ</span>
                <span class="metric-value">{wr['wv_sigma']:.5f}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">DWT Decomp. Level</span>
                <span class="metric-value">{wr['wv_level']}</span>
              </div>
              <div class="metric-row">
                <span class="metric-name">Dominant Frequency</span>
                <span class="metric-value">{wr['dominant_freq']:.1f} Hz</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: AUDIO & VOICE LIBRARY
# ==============================================================================
elif st.session_state["nav_page"] == "Audio & Voice Library":

    st.markdown(f'<div class="es-section"><div class="es-section-title">Acoustic Database · 15,391 Audio Files</div></div>', unsafe_allow_html=True)

    tab_cmu, tab_s23, tab_esc, tab_all = st.tabs([
        "Clean Speech (CMU ARCTIC)",
        "Speech-in-Noise (Step-23)",
        "Environmental Audio (ESC-50)",
        "Full Directory Search",
    ])

    with tab_cmu:
        clean_cat = get_clean_speech_catalog()
        total_recs = sum(len(v) for v in clean_cat.values())

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Recordings", f"{total_recs:,}")
        with col_m2:
            st.metric("Speakers", str(len(clean_cat)))
        with col_m3:
            st.metric("Format", "16-bit PCM WAV · 16 kHz")

        st.markdown(f'<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; margin-bottom:16px; line-height:1.6;">
          CMU ARCTIC corpus — phonetically balanced sentence recordings used as clean speech reference for
          the Step-23 speech-in-noise dataset synthesis pipeline. All recordings are studio-quality
          mono WAV files sampled at 16 kHz.
        </div>
        """, unsafe_allow_html=True)

        spk_sel = st.selectbox("Select Speaker", list(clean_cat.keys()),
                               format_func=lambda k: f"{k}  ({'Male' if 'fem' not in k else 'Female'} · {len(clean_cat[k])} sentences)")
        spk_files = clean_cat.get(spk_sel, [])

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.metric("Sentences Available", f"{len(spk_files):,}")
            st.metric("Speaker ID", spk_sel)
            st.metric("Gender", "Female" if "fem" in spk_sel else "Male")
        with col_right:
            selected_speech = st.selectbox("Select Sentence File", spk_files, format_func=lambda p: os.path.basename(p))
            if selected_speech:
                st.caption(f"File: `{os.path.basename(selected_speech)}`")
                st.audio(selected_speech, format="audio/wav")
                if st.button("Set as Active Pipeline Input", key="set_clean_spk"):
                    st.session_state["active_wav"] = selected_speech
                    st.session_state["active_label"] = f"{spk_sel} — {os.path.basename(selected_speech)}"
                    st.session_state["nav_page"] = "Live ANC Studio"
                    st.rerun()

    with tab_s23:
        df_s23 = get_step23_metadata()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Pairs", f"{len(df_s23):,}" if df_s23 is not None else "N/A")
        with col_m2:
            st.metric("Audio Duration", "6.15 Hours")
        with col_m3:
            st.metric("SNR Accuracy", "0.00 dB Mean Error")
        with col_m4:
            st.metric("Noise Profiles", "3 Defence Types")

        st.markdown(f"""
        <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; margin: 12px 0 16px; line-height:1.6;">
          Step-23 synthesized dataset: 5,535 clean-noisy speech pairs calibrated across multiple SNR levels
          (−5 dB, 0 dB, +5 dB, +10 dB, +15 dB) using three defence vehicle noise profiles (tank, helicopter,
          gun_fight). Used as the primary evaluation corpus for speech enhancement algorithms.
        </div>
        """, unsafe_allow_html=True)

        if df_s23 is not None:
            st.dataframe(
                df_s23[["pair_id", "split", "speech_source", "noise_label",
                         "noise_type", "snr_db_requested", "duration_sec"]].head(50),
                use_container_width=True,
                hide_index=True,
            )

    with tab_esc:
        df_esc = get_esc50_metadata()

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Clips", "2,000")
        with col_m2:
            st.metric("Categories", "50")
        with col_m3:
            st.metric("Duration", "5 s each · 44.1 kHz")

        st.markdown(f"""
        <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; margin: 12px 0 16px; line-height:1.6;">
          ESC-50 environmental audio benchmark — used for auxiliary acoustic characterization and
          defence vehicle sound proxy evaluation (engine, helicopter, siren, airplane categories).
        </div>
        """, unsafe_allow_html=True)

        if df_esc is not None:
            st.dataframe(
                df_esc[["filename", "category", "fold", "target"]].head(50),
                use_container_width=True,
                hide_index=True,
            )

    with tab_all:
        all_wavs = get_all_audio_files()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total WAV Files", f"{len(all_wavs):,}")
        with col_m2:
            st.metric("Indexed Directories", str(len(set(os.path.dirname(w) for w in all_wavs))))

        st.markdown(f'<div style="height:8px;"></div>', unsafe_allow_html=True)
        q = st.text_input("Filter by filename / folder / keyword", value="heli", placeholder="e.g. tank, arctic_b0001, noise...").strip().lower()
        results_wav = [w for w in all_wavs if q in w.lower()]
        st.caption(f"{min(len(results_wav), 100)} of {len(results_wav):,} matching files shown")

        for r in results_wav[:10]:
            c_info, c_play = st.columns([3, 2])
            with c_info:
                st.markdown(f"""
                <div style="padding: 8px 0; border-bottom: 1px solid {PALETTE['border']};">
                  <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                               font-weight: 600; color: {PALETTE['navy']};">
                    {os.path.basename(r)}
                  </div>
                  <div style="font-size: 0.75rem; color: {PALETTE['text_muted']}; margin-top: 2px;">
                    {os.path.relpath(r, DATA_DIR)}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c_play:
                st.audio(r, format="audio/wav")

# ==============================================================================
# PAGE 3: AI CLASSIFIER & BENCHMARKS
# ==============================================================================
elif st.session_state["nav_page"] == "AI Classifier & Benchmarks":

    st.markdown(f'<div class="es-section"><div class="es-section-title">ML Classifier Evaluation & Performance Benchmarks</div></div>', unsafe_allow_html=True)

    # Top metrics
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Classification Accuracy", "98.97%", help="Held-out stratified test windows")
    with c_m2:
        st.metric("Macro F1-Score", "0.97", help="Balanced across all 3 noise classes")
    with c_m3:
        st.metric("Algorithm", "Decision Tree", help="Fast, deterministic embedded inference")
    with c_m4:
        st.metric("Test Windows", "290", help="Stratified held-out 100 ms windows")

    # Per-class detail
    st.markdown(f"""
    <div class="tech-card" style="margin: 20px 0;">
      <div class="tech-card-title">Per-Class Classification Performance</div>
      <div class="metric-row">
        <span class="metric-name">TANK (Stationary · Low-Freq Mechanical)</span>
        <span class="metric-value">F1 = 0.93</span>
      </div>
      <div class="metric-row">
        <span class="metric-name">HELICOPTER (Non-Stationary · Low-Freq Tonal)</span>
        <span class="metric-value">F1 = 1.00</span>
      </div>
      <div class="metric-row">
        <span class="metric-name">GUN FIGHT (Impulsive · Transient)</span>
        <span class="metric-value">F1 = 0.99</span>
      </div>
      <div class="metric-row">
        <span class="metric-name">Feature Vector Dimensions</span>
        <span class="metric-value">8 features · 100 ms window · 50 ms hop</span>
      </div>
      <div class="metric-row">
        <span class="metric-name">Feature Extraction</span>
        <span class="metric-value">RMS · ZCR · Centroid · Rolloff · 4× Band Energy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Full Benchmark Table
    st.markdown(f'<div class="es-section"><div class="es-section-title">Measured Experiment Results vs Hardware Validation Status</div></div>', unsafe_allow_html=True)

    for item in utils.METRICS_DATA:
        t = item["type"]
        if "MEASURED" in t:
            pill_cls = "pill-measured"
        elif "SIMULATION" in t:
            pill_cls = "pill-simulation"
        elif "TEST" in t:
            pill_cls = "pill-model"
        else:
            pill_cls = "pill-pending"

        val_color = PALETTE["navy"] if item["numeric_val"] is not None else PALETTE["text_muted"]

        st.markdown(f"""
        <div class="metric-row" style="align-items: flex-start; padding: 13px 0;">
          <div style="flex: 1;">
            <div style="font-size:0.88rem; font-weight:600; color:{PALETTE['navy']}; margin-bottom:3px;">
              {item['module']}
            </div>
            <div style="font-size:0.82rem; color:{PALETTE['text_dim']}; margin-bottom:3px;">
              {item['metric']}
            </div>
            <div style="font-size:0.75rem; color:{PALETTE['text_muted']}; font-style:italic;">
              {item.get('notes', '')}
            </div>
          </div>
          <div style="text-align:right; flex-shrink:0; margin-left:24px;">
            <div style="font-size:1.05rem; font-family:'JetBrains Mono',monospace; font-weight:700;
                         color:{val_color}; margin-bottom:5px;">
              {item['value']}
            </div>
            <span class="pill {pill_cls}">{t}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Visualisation — comparison chart
    st.markdown(f'<div class="es-section"><div class="es-section-title">DSP Algorithm Performance Comparison</div></div>', unsafe_allow_html=True)

    measured = [(m["module"].split("(")[1].rstrip(")") if "(" in m["module"] else m["module"],
                 m["numeric_val"], m["type"])
                for m in utils.METRICS_DATA
                if m["numeric_val"] is not None and "NOT MEASURED" not in m["value"]]

    if measured:
        labels, values, types = zip(*measured)
        colors_chart = []
        for t in types:
            if "MEASURED" in t:
                colors_chart.append(PALETTE["teal"])
            elif "SIMULATION" in t:
                colors_chart.append(PALETTE["amber"])
            else:
                colors_chart.append(PALETTE["blue"])

        fig_bench, ax_bench = plt.subplots(figsize=(10, 3.5))
        bars = ax_bench.bar(range(len(labels)), values, color=colors_chart,
                             width=0.55, edgecolor="none")
        for bar, val in zip(bars, values):
            ax_bench.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                          f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                          fontweight="600", color=PALETTE["text_dim"])
        ax_bench.set_xticks(range(len(labels)))
        ax_bench.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
        ax_bench.set_ylabel("Performance Value (%  or  dB)")
        ax_bench.set_title("Algorithm Performance Overview", pad=10)
        ax_bench.grid(axis="y", linewidth=0.5)
        ax_bench.set_axisbelow(True)

        # Legend
        patches = [
            mpatches.Patch(color=PALETTE["teal"], label="Measured Software Experiment"),
            mpatches.Patch(color=PALETTE["amber"], label="Simulation"),
            mpatches.Patch(color=PALETTE["blue"], label="Model Test"),
        ]
        ax_bench.legend(handles=patches, loc="upper right", fontsize=8.5)
        fig_bench.tight_layout(pad=1.2)
        st.image(_fig_to_bytes(fig_bench), use_container_width=True)

# ==============================================================================
# PAGE 4: DSP & WAVELET LAB
# ==============================================================================
elif st.session_state["nav_page"] == "DSP & Wavelet Lab":

    st.markdown(f'<div class="es-section"><div class="es-section-title">Adaptive DSP Algorithms & Wavelet Reconstruction</div></div>', unsafe_allow_html=True)

    # DSP Config Reference Table
    st.markdown(f"""
    <div class="tech-card" style="margin-bottom: 20px;">
      <div class="tech-card-title">Adaptive Filter Configuration Reference · All Noise Profiles</div>
      <div style="overflow-x: auto;">
        <table style="width:100%; border-collapse:collapse; font-size:0.83rem;">
          <thead>
            <tr style="border-bottom: 1px solid {PALETTE['border_mid']};">
              <th style="text-align:left; padding:8px 12px; color:{PALETTE['text_muted']}; font-size:0.7rem;
                          letter-spacing:0.1em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;
                          font-weight:700;">Noise Class</th>
              <th style="text-align:left; padding:8px 12px; color:{PALETTE['text_muted']}; font-size:0.7rem;
                          letter-spacing:0.1em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;
                          font-weight:700;">Profile</th>
              <th style="text-align:right; padding:8px 12px; color:{PALETTE['text_muted']}; font-size:0.7rem;
                          letter-spacing:0.1em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;
                          font-weight:700;">Taps</th>
              <th style="text-align:right; padding:8px 12px; color:{PALETTE['text_muted']}; font-size:0.7rem;
                          letter-spacing:0.1em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;
                          font-weight:700;">Step Size μ</th>
              <th style="text-align:left; padding:8px 12px; color:{PALETTE['text_muted']}; font-size:0.7rem;
                          letter-spacing:0.1em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;
                          font-weight:700;">Mode</th>
            </tr>
          </thead>
          <tbody>
    """, unsafe_allow_html=True)

    for cls, cfg in utils.DSP_CONFIG_LOOKUP.items():
        st.markdown(f"""
            <tr style="border-bottom: 1px solid {PALETTE['border']};">
              <td style="padding:9px 12px; font-family:'JetBrains Mono',monospace; font-weight:600;
                          color:{PALETTE['navy']};">{cls.upper()}</td>
              <td style="padding:9px 12px; color:{PALETTE['text_dim']};">{cfg['profile']}</td>
              <td style="padding:9px 12px; text-align:right; font-family:'JetBrains Mono',monospace;
                          color:{PALETTE['navy']};">{cfg['filter_length']}</td>
              <td style="padding:9px 12px; text-align:right; font-family:'JetBrains Mono',monospace;
                          color:{PALETTE['blue']};">{cfg['step_size_mu']}</td>
              <td style="padding:9px 12px; color:{PALETTE['text_dim']}; font-size:0.8rem;">{cfg['processing_mode']}</td>
            </tr>
        """, unsafe_allow_html=True)

    st.markdown("""
          </tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_dsp1, c_dsp2 = st.columns(2)

    with c_dsp1:
        st.markdown(f"""
        <div class="tech-card">
          <div class="tech-card-title">1. Least Mean Squares (LMS) & NLMS</div>

          <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; line-height:1.75; margin-bottom:14px;">
            Standard LMS applies stochastic gradient descent to minimise mean squared error
            between the desired signal and adaptive filter output.
          </div>

          <div style="background:{PALETTE['bg_secondary']}; border:1px solid {PALETTE['border']};
                      border-left:3px solid {PALETTE['blue']};
                      border-radius:4px; padding:12px 16px; margin-bottom:12px;">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; color:{PALETTE['text_muted']};
                         text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
              Standard LMS Update Rule
            </div>
            <code style="font-size:0.88rem; color:{PALETTE['navy']}; font-family:'JetBrains Mono',monospace;">
              w[n+1] = w[n] + 2·μ·e[n]·x[n]
            </code>
          </div>

          <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; line-height:1.75; margin-bottom:14px;">
            Normalized LMS (NLMS) adapts the step size dynamically based on instantaneous input
            signal power to prevent divergence during transient noise bursts.
          </div>

          <div style="background:{PALETTE['bg_secondary']}; border:1px solid {PALETTE['border']};
                      border-left:3px solid {PALETTE['teal']};
                      border-radius:4px; padding:12px 16px; margin-bottom:14px;">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; color:{PALETTE['text_muted']};
                         text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
              NLMS Update Rule
            </div>
            <code style="font-size:0.88rem; color:{PALETTE['navy']}; font-family:'JetBrains Mono',monospace;">
              w[n+1] = w[n] + (μ / (‖x[n]‖² + ε)) · e[n] · x[n]
            </code>
          </div>

          <div class="metric-row">
            <span class="metric-name">LMS Measured Reduction (tank.wav)</span>
            <span class="metric-value" style="color:{PALETTE['blue']};">20.65 dB</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">NLMS Measured Reduction (tank.wav)</span>
            <span class="metric-value" style="color:{PALETTE['teal']};">31.58 dB</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Configuration</span>
            <span class="metric-value">32 taps · μ=0.005 / μ_norm=0.5</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c_dsp2:
        st.markdown(f"""
        <div class="tech-card">
          <div class="tech-card-title">2. FxLMS Active Noise Cancellation</div>

          <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; line-height:1.75; margin-bottom:14px;">
            Filtered-X LMS (FxLMS) is the standard algorithm for feedforward active noise
            cancellation. The reference signal is filtered through the estimated secondary path
            S̃(z) before weight adaptation.
          </div>

          <div style="background:{PALETTE['bg_secondary']}; border:1px solid {PALETTE['border']};
                      border-left:3px solid {PALETTE['wavelet']};
                      border-radius:4px; padding:12px 16px; margin-bottom:12px;">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; color:{PALETTE['text_muted']};
                         text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
              FxLMS Update Rule
            </div>
            <code style="font-size:0.88rem; color:{PALETTE['navy']}; font-family:'JetBrains Mono',monospace;">
              w[n+1] = w[n] + μ·e[n]·x'[n]<br>
              x'[n] = S̃(z) * x[n]
            </code>
          </div>

          <div style="font-size:0.78rem; color:{PALETTE['amber']}; background:{PALETTE['amber_light']};
                      border:1px solid {PALETTE['amber']}; border-radius:4px; padding:8px 12px;
                      margin-bottom:14px; font-family:'JetBrains Mono',monospace;">
            SIMULATION: Secondary path S̃(z) is synthetic 12-tap FIR (not a measured acoustic path).
            Physical SPL measurements require hardware validation phase.
          </div>

          <div class="metric-row">
            <span class="metric-name">FxLMS Simulated Reduction (tank.wav)</span>
            <span class="metric-value" style="color:{PALETTE['wavelet']};">11.89 dB</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Secondary Path Length</span>
            <span class="metric-value">12 taps · Synthetic</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">ANC Demo Cap</span>
            <span class="metric-value">60,000 samples</span>
          </div>
        </div>

        <div class="tech-card" style="margin-top:12px;">
          <div class="tech-card-title">3. Wavelet Denoising — Daubechies-4 Level-5</div>

          <div style="font-size:0.85rem; color:{PALETTE['text_dim']}; line-height:1.75; margin-bottom:14px;">
            Donoho-Johnstone VisuShrink universal soft-thresholding applied to DWT detail
            sub-bands. Preserves speech transients while attenuating wideband vehicle noise.
          </div>

          <div style="background:{PALETTE['bg_secondary']}; border:1px solid {PALETTE['border']};
                      border-left:3px solid {PALETTE['blue']};
                      border-radius:4px; padding:12px 16px; margin-bottom:14px;">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; color:{PALETTE['text_muted']};
                         text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
              VisuShrink Threshold
            </div>
            <code style="font-size:0.88rem; color:{PALETTE['navy']}; font-family:'JetBrains Mono',monospace;">
              λ = σ · √(2 · ln N)
            </code>
          </div>

          <div class="metric-row">
            <span class="metric-name">Wavelet Basis</span>
            <span class="metric-value">Daubechies-4 (db4)</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Decomposition Level</span>
            <span class="metric-value">5</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Speech Distortion</span>
            <span class="metric-value">&lt; 0.02 dB RMS</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline Diagram
    st.markdown(f'<div class="es-section"><div class="es-section-title">Signal Processing Pipeline Architecture</div></div>', unsafe_allow_html=True)

    pipeline_nodes = [
        ("SENSE", "INPUT"),
        ("MIC ARRAY", "ACOUSTIC"),
        ("ADC", "HARDWARE"),
        ("PRE-PROC", "DSP"),
        ("STFT", "TRANSFORM"),
        ("AI/ML", "CLASSIFY"),
        ("LMS/NLMS", "ADAPT"),
        ("FxLMS ANC", "CANCEL"),
        ("WAVELET", "DENOISE"),
        ("OUTPUT", "VERIFY"),
    ]

    node_items = []
    for i, (label, sublabel) in enumerate(pipeline_nodes):
        is_ai = label in ("AI/ML",)
        is_dsp = label in ("LMS/NLMS", "FxLMS ANC", "WAVELET")
        is_hw = label in ("MIC ARRAY", "ADC")
        if is_ai:
            node_color = PALETTE["purple"]
            bg_color = PALETTE["purple_light"]
            border_color = PALETTE["purple_border"]
        elif is_dsp:
            node_color = PALETTE["teal"]
            bg_color = PALETTE["teal_light"]
            border_color = PALETTE["teal_border"]
        elif is_hw:
            node_color = PALETTE["blue"]
            bg_color = PALETTE["blue_light"]
            border_color = PALETTE["blue_border"]
        else:
            node_color = PALETTE["navy"]
            bg_color = PALETTE["bg"]
            border_color = PALETTE["border_mid"]

        txt_muted = PALETTE["text_muted"]
        node_items.append(
            f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
            f'background:{bg_color};border:1.5px solid {border_color};border-radius:6px;'
            f'padding:8px 12px;min-width:82px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.03);">'
            f'<span style="font-size:0.72rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;'
            f'color:{node_color};letter-spacing:0.06em;">{label}</span>'
            f'<span style="font-size:0.6rem;font-weight:600;color:{txt_muted};'
            f'margin-top:2px;letter-spacing:0.05em;text-transform:uppercase;">{sublabel}</span>'
            f'</div>'
        )

    b_mid = PALETTE["border_mid"]
    b_col = PALETTE["border"]
    bg_sec = PALETTE["bg_secondary"]
    n_mid = PALETTE["navy_mid"]
    blue_c = PALETTE["blue"]

    arrow_sep = f'<span style="color:{b_mid};font-size:1.1rem;padding:0 4px;align-self:center;font-weight:700;">→</span>'
    diagram_nodes_html = arrow_sep.join(node_items)

    st.markdown(
        f'<div style="background:{bg_sec};border:1px solid {b_col};'
        f'border-radius:6px;padding:18px 22px;overflow-x:auto;">'
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;color:{n_mid};'
        f'text-transform:uppercase;font-family:\'JetBrains Mono\',monospace;margin-bottom:14px;display:flex;align-items:center;gap:8px;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{blue_c};"></span>'
        f'Sense → Understand → Adapt → Cancel → Verify'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:3px;min-width:max-content;">'
        f'{diagram_nodes_html}'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ==============================================================================
# PAGE 5: HARDWARE & ROADMAP
# ==============================================================================
elif st.session_state["nav_page"] == "Hardware & Roadmap":

    st.markdown(f'<div class="es-section"><div class="es-section-title">Embedded Hardware Architecture & SIH 2026 Roadmap</div></div>', unsafe_allow_html=True)

    c_hw1, c_hw2 = st.columns(2)

    with c_hw1:
        _done_items = [
            "End-to-end DSP &amp; AI pipeline software validation (Steps 01&#8211;24)",
            "Decision Tree classifier training &amp; evaluation &middot; 98.97% accuracy &middot; F1=0.97",
            "Standard LMS (20.65 dB) &amp; Normalized LMS (31.58 dB) software validation",
            "FxLMS acoustic simulation with synthetic secondary path S&#771;(z) &middot; 11.89 dB",
            "Daubechies-4 Level-5 wavelet soft-thresholding denoising",
            "Step-24 AI speech enhancement pipeline (Wiener + Multi-Band Wavelet Masking)",
            "15,391 audio file repository with interactive acoustic dashboard",
            "Speech-in-noise dataset synthesis (5,535 pairs &middot; 6.15 hours &middot; 5 SNR levels)",
        ]
        _done_rows = ""
        for _item in _done_items:
            _done_rows += (
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid {PALETTE["border"]};">'
                f'<div style="color:{PALETTE["teal"]};font-weight:700;font-size:0.9rem;flex-shrink:0;">&#10003;</div>'
                f'<div style="font-size:0.85rem;color:{PALETTE["text_dim"]};line-height:1.5;">{_item}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="tech-card" style="border-top:3px solid {PALETTE["teal"]};">'
            f'<div class="tech-card-title">'
            f'<span class="status-dot-active"></span>'
            f'Completed &#8212; Software Prototype Phase'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;gap:0;">'
            f'{_done_rows}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c_hw2:
        _pending_items = [
            "Real-time audio I/O: Microphone &#8594; ADC &#8594; ARM SoC &#8594; DAC &#8594; Speaker chain",
            "Measured secondary-path S(z) acoustic system identification",
            "ARM SoC real-time C/C++ firmware port with NEON SIMD acceleration",
            "In-cabin defence vehicle acoustic field trial &amp; physical SPL attenuation measurement",
            "Physical power profiling &amp; hardware processing latency benchmark",
            "FPGA co-processor integration for accelerated FxLMS computation",
        ]
        _pending_rows = ""
        for _item in _pending_items:
            _pending_rows += (
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid {PALETTE["border"]};">'
                f'<div style="color:{PALETTE["text_muted"]};font-weight:700;font-size:0.9rem;flex-shrink:0;">&#9675;</div>'
                f'<div style="font-size:0.85rem;color:{PALETTE["text_muted"]};line-height:1.5;">{_item}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="tech-card" style="border-top:3px solid {PALETTE["amber"]};">'
            f'<div class="tech-card-title">'
            f'<span class="status-dot-pending"></span>'
            f'Pending &#8212; Hardware Validation Phase'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;gap:0;">'
            f'{_pending_rows}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Hardware Architecture Overview
    st.markdown(f'<div class="es-section"><div class="es-section-title">Target Hardware Architecture</div></div>', unsafe_allow_html=True)

    c_h1, c_h2, c_h3 = st.columns(3)

    hw_components = [
        ("Sensor Layer", [
            ("Primary Microphone", "Error signal d[n] — noise source reference", "PENDING"),
            ("Reference Microphone", "Reference input x[n] — upstream noise", "PENDING"),
            ("Error Microphone", "Residual e[n] — cancellation feedback", "PENDING"),
        ]),
        ("Processing Layer", [
            ("Audio Codec", "ADC/DAC · I2S/SPI interface", "PENDING"),
            ("ARM SoC", "Real-time DSP host (main processor)", "PENDING"),
            ("FPGA Co-processor", "FxLMS hardware acceleration (planned)", "PENDING"),
        ]),
        ("Output Layer", [
            ("Anti-noise Speaker", "ANC acoustic output y[n]", "PENDING"),
            ("AI Classification", "Software · Decision Tree · 98.97% Acc", "ACTIVE"),
            ("Speech Enhancement", "Software · Wavelet + Wiener", "ACTIVE"),
        ]),
    ]

    for col, (layer_title, components) in zip([c_h1, c_h2, c_h3], hw_components):
        with col:
            st.markdown(f"""
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;color:{PALETTE['text_muted']};
                         text-transform:uppercase;font-family:'JetBrains Mono',monospace;
                         margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid {PALETTE['border']};">
              {layer_title}
            </div>
            """, unsafe_allow_html=True)
            for name, desc, status in components:
                dot_class = "status-dot-active" if status == "ACTIVE" else "status-dot-offline"
                st.markdown(f"""
                <div style="background:{PALETTE['bg']};border:1px solid {PALETTE['border']};
                            border-radius:5px;padding:12px 14px;margin-bottom:8px;">
                  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span class="{dot_class}"></span>
                    <span style="font-size:0.82rem;font-weight:700;color:{PALETTE['navy']};
                                  font-family:'JetBrains Mono',monospace;">{name}</span>
                  </div>
                  <div style="font-size:0.76rem;color:{PALETTE['text_muted']};line-height:1.4;">
                    {desc}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Defence Noise Profiles
    st.markdown(f'<div class="es-section"><div class="es-section-title">Defence Vehicle Acoustic Noise Profiles</div></div>', unsafe_allow_html=True)

    for cls, profile in utils.PROFILE_LOOKUP.items():
        dsp = utils.DSP_CONFIG_LOOKUP.get(cls, {})
        st.markdown(f"""
        <div class="metric-row" style="padding: 12px 0;">
          <div>
            <div style="font-size:0.9rem;font-weight:700;color:{PALETTE['navy']};
                         font-family:'JetBrains Mono',monospace;margin-bottom:3px;">
              {cls.upper()}
            </div>
            <div style="font-size:0.8rem;color:{PALETTE['text_dim']};">{profile.get('reason','')}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.82rem;font-family:'JetBrains Mono',monospace;color:{PALETTE['blue']};
                         font-weight:600;">{profile.get('processing_profile','')}</div>
            <div style="font-size:0.76rem;color:{PALETTE['text_muted']};margin-top:2px;">
              {dsp.get('filter_length','?')} taps · μ={dsp.get('step_size_mu','?')} · {dsp.get('processing_mode','')}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(f"""
<div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid {PALETTE['border']};
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
  <div style="font-size: 0.75rem; color: {PALETTE['text_muted']}; font-family: 'JetBrains Mono', monospace;">
    Echo Shield · PS ID SIH26052 · DRDO · Smart Vehicles · SIH 2026
  </div>
  <div style="font-size: 0.75rem; color: {PALETTE['text_muted']}; font-family: 'JetBrains Mono', monospace;">
    <span style="color:{PALETTE['teal']}; font-weight:700;">● SOFTWARE PROTOTYPE</span>
    &nbsp;·&nbsp;
    No hardware connected · Fully local · Zero external network calls
  </div>
</div>
""", unsafe_allow_html=True)
