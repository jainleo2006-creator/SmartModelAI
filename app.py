"""
SmartModel AI — Your Data. Your Model.
Enterprise AutoML platform. Works exclusively with user-uploaded datasets.
No external data sources. No Kaggle. No internet fetching.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io, os, json, time, pickle, warnings, traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# ── Third-party core ──────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Scikit-Learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

# ── Optional boosting ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor; HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor; HAS_CAT = True
except ImportError:
    HAS_CAT = False

# ── PDF ───────────────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
MODELS   = ROOT / "models";   MODELS.mkdir(exist_ok=True)
EXPORTS  = ROOT / "exports";  EXPORTS.mkdir(exist_ok=True)
REPORTS  = ROOT / "reports";  REPORTS.mkdir(exist_ok=True)
HISTORY_FILE = ROOT / "history.json"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartModel AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Dark/Light themes, premium SaaS aesthetic
# ══════════════════════════════════════════════════════════════════════════════
DARK_CSS = """
:root {
  --bg-base:    #020b18;
  --bg-surface: #071428;
  --bg-card:    #0a1c38;
  --bg-hover:   #0f2648;
  --border:     #153060;
  --border-accent: #1e4a8a;
  --text-primary: #ddeeff;
  --text-secondary: #7aaad4;
  --text-muted: #3a6090;
  --accent-1: #2d7dd2;
  --accent-2: #38b6ff;
  --accent-3: #1a56a8;
  --accent-warn: #f59e0b;
  --accent-err: #ef4444;
  --accent-ok: #22c55e;
  --grad-primary: linear-gradient(135deg, #1a56a8 0%, #38b6ff 100%);
  --grad-card: linear-gradient(135deg, #0a1c38 0%, #071428 100%);
}
"""

LIGHT_CSS = """
:root {
  --bg-base:    #eaf2fc;
  --bg-surface: #ffffff;
  --bg-card:    #ffffff;
  --bg-hover:   #ddeeff;
  --border:     #b8d6f0;
  --border-accent: #7ab5e0;
  --text-primary: #051a38;
  --text-secondary: #1a508a;
  --text-muted: #4a82b8;
  --accent-1: #1a6bbf;
  --accent-2: #0590d0;
  --accent-3: #0a4a8a;
  --accent-warn: #d97706;
  --accent-err: #dc2626;
  --accent-ok: #16a34a;
  --grad-primary: linear-gradient(135deg, #0a4a8a 0%, #0590d0 100%);
  --grad-card: linear-gradient(135deg, #ffffff 0%, #eaf4ff 100%);
}
"""

SHARED_CSS = """
/* ── Reset & Base ── */
[data-testid="stAppViewContainer"] { background: var(--bg-base) !important; }
[data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--border) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; box-sizing: border-box; }
[data-testid="stSidebar"] * { font-family: 'Inter', -apple-system, sans-serif !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5 { color: var(--text-primary) !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
p, li, label { color: var(--text-secondary) !important; }
.stMarkdown p { color: var(--text-secondary) !important; }
code { background: var(--bg-hover) !important; color: var(--accent-1) !important; border-radius: 4px !important; padding: 2px 6px !important; font-size: .85em !important; }

/* ── Sidebar brand ── */
.brand-logo {
  font-size: 1.45rem; font-weight: 800; letter-spacing: -0.03em;
  background: var(--grad-primary);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.brand-tag {
  font-size: .6rem; font-weight: 700; letter-spacing: .2em;
  text-transform: uppercase; color: var(--text-muted); margin-top: .1rem;
}

/* ── Cards ── */
.sm-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem 1.5rem;
  transition: border-color .2s, box-shadow .2s;
}
.sm-card:hover { border-color: var(--border-accent); }
.sm-card-accent { border-color: var(--accent-1) !important; }

.card-eyebrow {
  font-size: .65rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: .4rem;
}
.card-metric { font-size: 2rem; font-weight: 800; color: var(--text-primary); letter-spacing: -.04em; line-height: 1; }
.card-sub { font-size: .78rem; color: var(--text-muted); margin-top: .3rem; }

/* ── Hero section ── */
.page-hero { padding: .5rem 0 1.5rem; }
.page-title {
  font-size: 1.65rem; font-weight: 800; color: var(--text-primary);
  letter-spacing: -.03em; line-height: 1.2;
}
.page-subtitle { font-size: .9rem; color: var(--text-muted); margin-top: .35rem; }

/* ── Active dataset chip ── */
.dataset-chip {
  background: var(--bg-hover); border: 1px solid var(--border-accent);
  border-radius: 10px; padding: .6rem 1rem; margin: .5rem 0;
}
.dataset-chip-label { font-size: .6rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent-2); }
.dataset-chip-name { font-size: .85rem; font-weight: 600; color: var(--text-primary); margin: .15rem 0 .1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dataset-chip-meta { font-size: .7rem; color: var(--text-muted); }

/* ── Nav items ── */
.nav-group-label {
  font-size: .6rem; font-weight: 700; letter-spacing: .16em;
  text-transform: uppercase; color: var(--text-muted);
  padding: .9rem 1rem .3rem; display: block;
}

/* ── Badges ── */
.badge {
  display: inline-flex; align-items: center; gap: .3rem;
  padding: .2rem .65rem; border-radius: 20px;
  font-size: .7rem; font-weight: 700; letter-spacing: .04em;
}
.badge-clf { background: rgba(91,141,238,.15); color: var(--accent-1); }
.badge-reg { background: rgba(157,122,255,.15); color: var(--accent-3); }
.badge-ok  { background: rgba(34,197,94,.12); color: var(--accent-ok); }
.badge-warn{ background: rgba(245,158,11,.12); color: var(--accent-warn); }
.badge-err { background: rgba(239,68,68,.12); color: var(--accent-err); }
.badge-best{ background: rgba(56,217,192,.12); color: var(--accent-2); }

/* ── Step pills ── */
.step-pill {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .35rem .85rem; border-radius: 30px; font-size: .76rem; font-weight: 600;
  background: var(--bg-hover); color: var(--accent-1);
  border: 1px solid var(--border-accent); margin: .15rem;
}

/* ── Workflow steps ── */
.workflow-step {
  display: flex; align-items: flex-start; gap: 1rem;
  padding: 1rem 1.25rem; border-radius: 12px;
  background: var(--bg-card); border: 1px solid var(--border);
  margin-bottom: .5rem; transition: border-color .2s;
}
.workflow-step:hover { border-color: var(--accent-1); }
.step-num {
  width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
  background: var(--grad-primary); display: flex; align-items: center; justify-content: center;
  font-size: .75rem; font-weight: 800; color: #fff;
}
.step-content-title { font-size: .9rem; font-weight: 700; color: var(--text-primary); margin-bottom: .15rem; }
.step-content-desc { font-size: .8rem; color: var(--text-muted); }

/* ── Section divider ── */
.sec-divider { height: 1px; background: var(--border); margin: 1.25rem 0; }
.sec-title {
  font-size: .8rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: .75rem;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--grad-primary) !important;
  color: #fff !important; border: none !important; border-radius: 9px !important;
  font-weight: 700 !important; font-size: .85rem !important;
  padding: .6rem 1.4rem !important; letter-spacing: .01em !important;
  transition: opacity .2s, transform .1s !important; box-shadow: none !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea,
.stNumberInput input {
  background: var(--bg-hover) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: 9px !important;
}
.stSelectbox > div > div { color: var(--text-primary) !important; }

/* ── Multiselect ── */
.stMultiSelect > div > div {
  background: var(--bg-hover) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
}

/* ── Checkbox ── */
.stCheckbox label { color: var(--text-secondary) !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent-1) !important; }
.stSlider div[data-testid="stSliderTrack"] > div:first-child { background: var(--bg-hover) !important; }

/* ── Progress ── */
.stProgress > div > div { background: var(--grad-primary) !important; border-radius: 4px !important; }
.stProgress > div { background: var(--bg-hover) !important; border-radius: 4px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: .72rem !important; font-weight: 700 !important; letter-spacing: .08em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: .78rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; padding: 4px !important; gap: 2px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 7px !important; color: var(--text-muted) !important; font-weight: 600 !important; font-size: .83rem !important; padding: .4rem .9rem !important; border: none !important; }
.stTabs [aria-selected="true"] { background: var(--bg-hover) !important; color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: var(--bg-card) !important; color: var(--text-secondary) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; }
.streamlit-expanderContent { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 10px 10px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; border: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] iframe { background: var(--bg-card) !important; border-radius: 10px !important; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; border-left-width: 3px !important; }
[data-testid="stWarning"] { background: rgba(245,158,11,.07) !important; border-color: var(--accent-warn) !important; }
[data-testid="stSuccess"] { background: rgba(34,197,94,.07) !important; border-color: var(--accent-ok) !important; }
[data-testid="stError"] { background: rgba(239,68,68,.07) !important; border-color: var(--accent-err) !important; }
[data-testid="stInfo"] { background: rgba(91,141,238,.07) !important; border-color: var(--accent-1) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { background: var(--bg-card) !important; border: 2px dashed var(--border-accent) !important; border-radius: 14px !important; padding: 1.5rem !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent-1) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 4px; }

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  color: var(--text-secondary) !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; font-size: .84rem !important;
  padding: .5rem .85rem !important; text-align: left !important;
  width: 100% !important; justify-content: flex-start !important;
  transition: background .15s, color .15s !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--bg-hover) !important;
  color: var(--text-primary) !important;
  transform: none !important;
  opacity: 1 !important;
}

/* ── Sidebar clear button override ── */
.sidebar-clear .stButton > button {
  background: rgba(239,68,68,.08) !important;
  color: var(--accent-err) !important;
  border: 1px solid rgba(239,68,68,.2) !important;
}
.sidebar-clear .stButton > button:hover {
  background: rgba(239,68,68,.15) !important;
  color: var(--accent-err) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
  background: var(--bg-hover) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-accent) !important;
  border-radius: 9px !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent-1) !important;
  color: var(--accent-1) !important;
  transform: none !important; opacity: 1 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-1) !important; }

/* ── Upload zone ── */
.upload-zone {
  border: 2px dashed var(--border-accent); border-radius: 16px;
  padding: 3rem 2rem; text-align: center; cursor: pointer;
  background: var(--bg-card); transition: border-color .2s, background .2s;
}
.upload-zone:hover { border-color: var(--accent-1); background: var(--bg-hover); }
.upload-icon { font-size: 2.5rem; margin-bottom: .75rem; }
.upload-title { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
.upload-desc { font-size: .82rem; color: var(--text-muted); margin-top: .35rem; }

/* ── Leaderboard rows ── */
.lb-row {
  display: flex; align-items: center; gap: 1rem;
  padding: .7rem 1rem; border-radius: 10px; margin-bottom: .35rem;
  background: var(--bg-card); border: 1px solid var(--border);
  transition: border-color .15s;
}
.lb-row:hover { border-color: var(--border-accent); }
.lb-rank { font-size: 1rem; font-weight: 800; color: var(--accent-3); min-width: 1.8rem; }
.lb-name { font-size: .88rem; font-weight: 600; color: var(--text-primary); flex: 1; }
.lb-score { font-size: .92rem; font-weight: 800; color: var(--accent-2); }
.lb-time { font-size: .75rem; color: var(--text-muted); min-width: 50px; text-align: right; }

/* ── Feature importance bar ── */
.fi-row { display: flex; align-items: center; gap: .75rem; margin-bottom: .4rem; }
.fi-name { font-size: .8rem; color: var(--text-secondary); min-width: 120px; text-align: right; flex-shrink: 0; }
.fi-bar-bg { flex: 1; height: 8px; background: var(--bg-hover); border-radius: 4px; overflow: hidden; }
.fi-bar-fill { height: 100%; border-radius: 4px; background: var(--grad-primary); }
.fi-val { font-size: .75rem; font-weight: 700; color: var(--text-muted); min-width: 45px; }

/* ── Form wrapper ── */
[data-testid="stForm"] {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; padding: 1.25rem 1.5rem !important;
}
[data-testid="stFormSubmitButton"] > button {
  background: var(--grad-primary) !important;
  color: #fff !important; font-weight: 700 !important;
  border-radius: 9px !important; border: none !important;
  padding: .65rem 1.5rem !important;
}
"""

def apply_theme():
    theme = st.session_state.get("theme", "Dark")
    theme_css = DARK_CSS if theme == "Dark" else LIGHT_CSS
    st.markdown(f"<style>{theme_css}{SHARED_CSS}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "df_raw": None, "df_clean": None, "df_source": None,
    "target_col": None, "problem_type": None, "feature_cols": [],
    "results": [], "best_model": None, "best_model_name": "",
    "X_test": None, "y_test": None, "le": None,
    "cleaning_log": [], "training_log": [],
    "page": "Dashboard",
    "theme": "Dark",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

apply_theme()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_num(n) -> str:
    try:
        n = float(n)
        if n >= 1e9: return f"{n/1e9:.1f}B"
        if n >= 1e6: return f"{n/1e6:.1f}M"
        if n >= 1e3: return f"{n/1e3:.1f}K"
        return str(int(n))
    except Exception:
        return str(n)

@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_bytes: bytes, ext: str) -> pd.DataFrame:
    """Parse uploaded CSV/Excel bytes into a DataFrame. Cached on file content so
    re-running the script (e.g. switching tabs) doesn't re-parse unchanged files."""
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if ext == ".csv" else pd.read_excel(buf)

def infer_problem(series: pd.Series) -> str:
    if series.dtype == object or series.nunique() <= 20:
        return "Classification"
    return "Regression"

@st.cache_data(show_spinner=False)
def compute_overview_stats(df: pd.DataFrame):
    """Expensive, pure dataframe summaries — cached so they don't recompute on every rerun."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.count().values,
        "Null": df.isnull().sum().values,
        "Unique": df.nunique().values,
    })
    describe = df.describe(include="all").T
    return info, describe

@st.cache_data(show_spinner=False)
def compute_missing_summary(df: pd.DataFrame):
    miss = df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    return miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)

@st.cache_data(show_spinner=False)
def compute_basic_stats(df: pd.DataFrame):
    """Rows/cols/missing/duplicates/memory — cached since they scan the full dataframe."""
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
    }

def badge(text: str, kind: str = "clf") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'

def plotly_dark(fig, title=""):
    """Apply dark theme to plotly chart."""
    is_dark = st.session_state.get("theme", "Dark") == "Dark"
    bg_base    = "#080c14" if is_dark else "#f0f4fc"
    bg_surface = "#0a1c38" if is_dark else "#ffffff"
    text_col   = "#e8eeff" if is_dark else "#0d1530"
    grid_col   = "#1e2845" if is_dark else "#dde4f5"
    sub_col    = "#8898c4" if is_dark else "#4a5c8a"

    fig.update_layout(
        paper_bgcolor=bg_base, plot_bgcolor=bg_surface,
        font_color=sub_col, title_font_color=text_col,
        title_text=title, title_x=0.5, title_font_size=14,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(bgcolor=bg_surface, bordercolor=grid_col, borderwidth=1,
                    font=dict(color=sub_col)),
        xaxis=dict(gridcolor=grid_col, zerolinecolor=grid_col, linecolor=grid_col,
                   tickfont=dict(color=sub_col)),
        yaxis=dict(gridcolor=grid_col, zerolinecolor=grid_col, linecolor=grid_col,
                   tickfont=dict(color=sub_col)),
    )
    return fig

def sm_card(title: str, value: str, sub: str = "", color: str = "var(--accent-1)"):
    st.markdown(f"""
    <div class="sm-card">
      <div class="card-eyebrow">{title}</div>
      <div class="card-metric" style="color:{color}">{value}</div>
      {'<div class="card-sub">'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def section_header(title: str, subtitle: str = ""):
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="page-hero">
      <div class="page-title">{title}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

def sec_title(text: str):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

PALETTE = ["#38b6ff","#2d7dd2","#1a56a8","#f59e0b","#ef4444",
           "#22c55e","#f472b6","#fb923c","#38bdf8","#a78bfa"]

# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def load_history() -> List[Dict]:
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text())
        except Exception: return []
    return []

def save_history(record: Dict):
    hist = load_history()
    hist.insert(0, record)
    HISTORY_FILE.write_text(json.dumps(hist[:200], indent=2, default=str))

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def clean_dataframe(
    df: pd.DataFrame,
    remove_dups: bool = True,
    fill_strategy: str = "median",
    encode_cats: bool = True,
    encoding_method: str = "label",
    scale: str = "standard",
    drop_outliers: bool = False,
    outlier_method: str = "zscore",
    outlier_z: float = 3.0,
    feature_engineering: str = "none",
) -> Tuple[pd.DataFrame, List[str]]:
    log = []
    df = df.copy()
    orig_rows = len(df)

    if remove_dups:
        df = df.drop_duplicates()
        removed = orig_rows - len(df)
        if removed: log.append(f"Removed {removed:,} duplicate rows")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=object).columns.tolist()

    for c in num_cols:
        if df[c].isna().any():
            if fill_strategy == "median":    fill_val = df[c].median()
            elif fill_strategy == "mean":    fill_val = df[c].mean()
            elif fill_strategy == "zero":    fill_val = 0
            elif fill_strategy == "drop":
                df = df.dropna(subset=[c])
                log.append(f"Dropped rows with NaN in '{c}'"); continue
            else:                            fill_val = df[c].median()
            df[c] = df[c].fillna(fill_val)
            log.append(f"Filled '{c}' NaN → {fill_strategy}")

    for c in cat_cols:
        if df[c].isna().any():
            fill_val = df[c].mode()[0] if len(df[c].mode()) else "Unknown"
            df[c] = df[c].fillna(fill_val)
            log.append(f"Filled '{c}' NaN → mode")

    if drop_outliers and num_cols:
        if outlier_method == "iqr":
            mask = np.ones(len(df), dtype=bool)
            for c in num_cols:
                q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask &= df[c].between(lo, hi) | df[c].isna()
            removed = (~mask).sum()
            df = df[mask]
            if removed: log.append(f"Removed {removed:,} outlier rows (IQR method, 1.5×IQR)")
        else:
            from scipy import stats as sp_stats
            mask = np.ones(len(df), dtype=bool)
            for c in num_cols:
                z = np.abs(sp_stats.zscore(df[c], nan_policy="omit"))
                mask &= (z <= outlier_z)
            removed = (~mask).sum()
            df = df[mask]
            if removed: log.append(f"Removed {removed:,} outlier rows (|Z|>{outlier_z})")

    if encode_cats:
        if encoding_method == "onehot":
            for c in list(cat_cols):
                if c not in df.columns: continue
                nun = df[c].nunique()
                if nun <= 50:
                    if nun <= 2:
                        # Binary categorical: keep as a single label-encoded column
                        le = LabelEncoder()
                        df[c] = le.fit_transform(df[c].astype(str))
                        log.append(f"Label-encoded binary column '{c}'")
                    else:
                        dummies = pd.get_dummies(df[c].astype(str), prefix=c, dtype=int)
                        df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
                        log.append(f"One-hot encoded '{c}' ({dummies.shape[1]} new columns)")
                else:
                    df = df.drop(columns=[c])
                    log.append(f"Dropped high-cardinality column '{c}'")
        else:
            for c in list(cat_cols):
                if c not in df.columns: continue
                if df[c].nunique() <= 50:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].astype(str))
                    log.append(f"Label-encoded '{c}' ({df[c].nunique()} classes)")
                else:
                    df = df.drop(columns=[c])
                    log.append(f"Dropped high-cardinality column '{c}'")

    cur_num = df.select_dtypes(include=np.number).columns.tolist()

    if feature_engineering != "none" and cur_num:
        # Cap the base column count so we don't explode dimensionality
        base_cols = cur_num[:8]
        if feature_engineering == "interactions":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            arr = poly.fit_transform(df[base_cols])
            new_names = poly.get_feature_names_out(base_cols)
            added = [n for n in new_names if n not in base_cols]
            poly_df = pd.DataFrame(arr, columns=new_names, index=df.index)[added]
            df = pd.concat([df, poly_df], axis=1)
            log.append(f"Added {len(added)} interaction features (pairwise products of {len(base_cols)} columns)")
        elif feature_engineering == "polynomial":
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            arr = poly.fit_transform(df[base_cols])
            new_names = poly.get_feature_names_out(base_cols)
            added = [n for n in new_names if n not in base_cols]
            poly_df = pd.DataFrame(arr, columns=new_names, index=df.index)[added]
            df = pd.concat([df, poly_df], axis=1)
            log.append(f"Added {len(added)} polynomial features (degree 2, incl. squares) from {len(base_cols)} columns")

    cur_num = df.select_dtypes(include=np.number).columns.tolist()
    if scale != "none" and cur_num:
        scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
                  "robust": RobustScaler()}.get(scale, StandardScaler())
        df[cur_num] = scaler.fit_transform(df[cur_num])
        log.append(f"Applied {scale} scaling to {len(cur_num)} numeric columns")

    if not log: log.append("Dataset is already clean — no changes needed.")
    return df, log

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════
def get_clf_models(class_weight=None) -> Dict[str, Any]:
    cw = class_weight  # None or "balanced"
    # NOTE: n_jobs=1 on individual estimators here — parallelism is applied at the
    # outer cross_val_score/GridSearchCV level instead. Setting n_jobs=-1 on both
    # layers oversubscribes CPU cores and slows things down rather than speeding
    # them up.
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42, class_weight=cw, solver="lbfgs"),
        "Decision Tree":        DecisionTreeClassifier(random_state=42, class_weight=cw, max_depth=10),
        "Random Forest":        RandomForestClassifier(n_estimators=150, random_state=42, class_weight=cw, n_jobs=1),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=150, random_state=42, class_weight=cw, n_jobs=1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42, subsample=0.8),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, random_state=42, class_weight=cw),
        "KNN":                  KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=1),
        "SVM":                  CalibratedClassifierCV(SVC(kernel="rbf", random_state=42, class_weight=cw), cv=3),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, random_state=42, eval_metric="logloss", verbosity=0,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.05, use_label_encoder=False,
            n_jobs=1,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150, random_state=42, verbose=-1,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
            class_weight=cw, n_jobs=1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(iterations=150, random_state=42, verbose=0, learning_rate=0.05)
    return models

def get_reg_models() -> Dict[str, Any]:
    # n_jobs=1 on individual estimators — parallelism applied at the outer
    # cross_val_score/GridSearchCV level to avoid oversubscribing CPU cores.
    models = {
        "Linear Regression":   LinearRegression(),
        "Ridge":               Ridge(alpha=1.0),
        "Lasso":               Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet":          ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "KNN Regressor":       KNeighborsRegressor(n_neighbors=7, metric="euclidean", n_jobs=1),
        "Decision Tree":       DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest":       RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
        "Extra Trees":         ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=1),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=150, random_state=42, subsample=0.8),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, random_state=42),
        "SVR":                 SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, random_state=42, verbosity=0,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
            n_jobs=1,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=150, random_state=42, verbose=-1,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
            n_jobs=1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(iterations=150, random_state=42, verbose=0, learning_rate=0.05)
    return models

def build_stacking_clf(class_weight=None):
    """Build a stacking ensemble from top base learners.
    Inner estimators use n_jobs=1 since StackingClassifier(n_jobs=-1) already
    parallelizes across base learners — stacking both layers oversubscribes cores."""
    estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight, n_jobs=1)),
        ("et",  ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight=class_weight, n_jobs=1)),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss", n_jobs=1)))
    meta = LogisticRegression(max_iter=500, random_state=42, class_weight=class_weight)
    return StackingClassifier(estimators=estimators, final_estimator=meta, cv=3, n_jobs=-1, passthrough=False)

def build_stacking_reg():
    """Build a stacking ensemble for regression. See build_stacking_clf for n_jobs rationale."""
    estimators = [
        ("rf",  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ("et",  ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ("gb",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=1)))
    meta = Ridge(alpha=1.0)
    return StackingRegressor(estimators=estimators, final_estimator=meta, cv=3, n_jobs=-1, passthrough=False)

CLF_SPACES = {
    "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Random Forest":       {"n_estimators":[100,200,300],"max_depth":[None,5,10,20],"min_samples_split":[2,5,10],"max_features":["sqrt","log2"]},
    "Extra Trees":         {"n_estimators":[100,200,300],"max_depth":[None,5,10,20],"min_samples_split":[2,5,10]},
    "Gradient Boosting":   {"n_estimators":[100,200,300],"max_depth":[3,5,7],"learning_rate":[.01,.05,.1],"subsample":[.7,.8,1.0]},
    "Hist Gradient Boosting": {"max_iter":[100,150,200,300],"max_depth":[None,5,10,20],"learning_rate":[.01,.05,.1,.2],"l2_regularization":[0,.1,1.0]},
    "XGBoost":             {"n_estimators":[100,200,300],"max_depth":[3,5,7],"learning_rate":[.01,.05,.1,.2],"subsample":[.7,.8,1.0],"colsample_bytree":[.6,.8,1.0],"min_child_weight":[1,3,5]},
    "LightGBM":            {"n_estimators":[100,200,300],"max_depth":[-1,5,10],"learning_rate":[.01,.05,.1,.2],"num_leaves":[31,63,127],"subsample":[.7,.8,1.0]},
    "KNN":                 {"n_neighbors":[3,5,7,11,15],"weights":["uniform","distance"],"metric":["euclidean","manhattan"]},
}
REG_SPACES = {
    "Ridge":               {"alpha":[0.001,0.01,0.1,1.0,10.0,100.0]},
    "Lasso":               {"alpha":[0.001,0.01,0.1,1.0,10.0]},
    "ElasticNet":          {"alpha":[0.01,0.1,1.0],"l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
    "Random Forest":       {"n_estimators":[100,200,300],"max_depth":[None,5,10,20],"min_samples_split":[2,5,10],"max_features":["sqrt","log2"]},
    "Extra Trees":         {"n_estimators":[100,200,300],"max_depth":[None,5,10,20],"min_samples_split":[2,5,10]},
    "Gradient Boosting":   {"n_estimators":[100,200,300],"max_depth":[3,5,7],"learning_rate":[.01,.05,.1],"subsample":[.7,.8,1.0]},
    "Hist Gradient Boosting": {"max_iter":[100,150,200,300],"max_depth":[None,5,10,20],"learning_rate":[.01,.05,.1,.2],"l2_regularization":[0,.1,1.0]},
    "XGBoost":             {"n_estimators":[100,200,300],"max_depth":[3,5,7],"learning_rate":[.01,.05,.1,.2],"subsample":[.7,.8,1.0],"colsample_bytree":[.6,.8,1.0]},
    "LightGBM":            {"n_estimators":[100,200,300],"max_depth":[-1,5,10],"learning_rate":[.01,.05,.1,.2],"num_leaves":[31,63,127]},
    "KNN Regressor":       {"n_neighbors":[3,5,7,11,15],"weights":["uniform","distance"]},
}

# ══════════════════════════════════════════════════════════════════════════════
# AUTOML ENGINE  (upgraded)
# ══════════════════════════════════════════════════════════════════════════════
def detect_class_imbalance(y) -> bool:
    """Return True if the minority class is < 20% of the dataset."""
    try:
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) < 2: return False
        ratio = counts.min() / counts.sum()
        return ratio < 0.20
    except Exception:
        return False

def run_automl(
    X_train, X_test, y_train, y_test,
    problem: str,
    tune_mode: str = "none",
    use_stacking: bool = True,
    cv_folds: int = 5,
    progress_bar=None,
    status_text=None,
) -> List[Dict]:

    # ── Imbalance detection ──────────────────────────────────────────────────
    imbalanced = False
    class_weight = None
    if problem == "Classification":
        imbalanced = detect_class_imbalance(y_train)
        if imbalanced:
            class_weight = "balanced"

    models = get_clf_models(class_weight=class_weight) if problem == "Classification" else get_reg_models()

    # ── Add stacking ensemble ────────────────────────────────────────────────
    if use_stacking and len(X_train) >= 100:
        if problem == "Classification":
            models["Stacking Ensemble"] = build_stacking_clf(class_weight=class_weight)
        else:
            models["Stacking Ensemble"] = build_stacking_reg()

    spaces = CLF_SPACES if problem == "Classification" else REG_SPACES
    results = []
    n = len(models)
    clf_scoring = "f1_weighted" if problem == "Classification" else "r2"

    for i, (name, model) in enumerate(models.items()):
        pct = i / n
        if progress_bar: progress_bar.progress(pct, text=f"Training {name}… ({i+1}/{n})")
        if status_text:  status_text.markdown(f'<div class="step-pill">⚙️ {name}</div>', unsafe_allow_html=True)

        t0 = time.time()
        try:
            # ── Hyperparameter tuning ────────────────────────────────────────
            search_cv = 3
            searcher = None
            if tune_mode != "none" and name in spaces:
                kwargs = dict(cv=search_cv, scoring=clf_scoring, n_jobs=-1, error_score=0)
                if tune_mode == "random":
                    searcher = RandomizedSearchCV(model, spaces[name], n_iter=15, random_state=42, **kwargs)
                else:
                    searcher = GridSearchCV(model, spaces[name], **kwargs)
                searcher.fit(X_train, y_train)
                model = searcher.best_estimator_
                best_params = searcher.best_params_
            else:
                model.fit(X_train, y_train)
                best_params = {}

            elapsed = round(time.time() - t0, 2)
            y_pred  = model.predict(X_test)

            # ── Cross-validation score ───────────────────────────────────────
            # If the search already ran CV with the same fold count the caller asked
            # for, reuse its results instead of retraining the model from scratch
            # again via a fresh cross_val_score pass — searcher.cv_results_ already
            # has the per-fold scores for the winning hyperparameter combination.
            if searcher is not None and search_cv == cv_folds:
                best_idx = searcher.best_index_
                cv_res = searcher.cv_results_
                split_scores = [cv_res[f"split{k}_test_score"][best_idx] for k in range(search_cv)]
                cv_mean = round(float(np.mean(split_scores)), 4)
                cv_std  = round(float(np.std(split_scores)), 4)
            else:
                try:
                    cv_sc = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                            scoring=clf_scoring, n_jobs=-1, error_score=0)
                    cv_mean = round(float(cv_sc.mean()), 4)
                    cv_std  = round(float(cv_sc.std()), 4)
                except Exception:
                    cv_mean, cv_std = None, None

            row = {"Model": name, "Time(s)": elapsed,
                   "CV Score": cv_mean, "CV Std": cv_std,
                   "_best_params": best_params}

            if problem == "Classification":
                n_classes = len(np.unique(y_test))
                row["Accuracy"]          = round(accuracy_score(y_test, y_pred), 4)
                row["Balanced Accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
                row["Precision"]         = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                row["Recall"]            = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                row["F1"]                = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                try:
                    row["MCC"] = round(matthews_corrcoef(y_test, y_pred), 4)
                except Exception:
                    row["MCC"] = "—"
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)
                        if n_classes == 2:
                            row["ROC-AUC"] = round(roc_auc_score(y_test, proba[:, 1]), 4)
                        else:
                            row["ROC-AUC"] = round(
                                roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 4
                            )
                    else:
                        row["ROC-AUC"] = "—"
                except Exception:
                    row["ROC-AUC"] = "—"
                row["Score"] = row["F1"]   # F1-weighted as primary score

            else:
                row["R²"]    = round(r2_score(y_test, y_pred), 4)
                row["RMSE"]  = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                row["MAE"]   = round(mean_absolute_error(y_test, y_pred), 4)
                try:
                    row["MAPE"] = round(
                        float(np.mean(np.abs((np.array(y_test) - np.array(y_pred)) /
                                             np.clip(np.abs(np.array(y_test)), 1e-8, None))) * 100), 2
                    )
                except Exception:
                    row["MAPE"] = "—"
                row["Score"] = row["R²"]

            row["_model"]    = model
            row["_y_pred"]   = y_pred
            row["_imbalanced"] = imbalanced
            results.append(row)

        except Exception as e:
            results.append({"Model": name, "Score": -9999, "Error": str(e),
                            "_model": None, "_y_pred": None})

    if progress_bar: progress_bar.progress(1.0, text="Training complete ✅")
    results.sort(key=lambda r: r.get("Score", -9999), reverse=True)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def chart_confusion(y_test, y_pred, labels=None):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0,"#0a1c38"],[1,"#2d7dd2"]],
                    labels=dict(x="Predicted", y="Actual"), x=labels, y=labels)
    return plotly_dark(fig, "Confusion Matrix")

def chart_roc(y_test, y_pred_proba, model_name=""):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.3f}",
                             line=dict(color="#2d7dd2", width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dot", color="#4a5680"), name="Random"))
    return plotly_dark(fig, f"ROC Curve — {model_name}")

def chart_prc(y_test, y_pred_proba):
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             line=dict(color="#38b6ff", width=2.5), name="P-R Curve"))
    return plotly_dark(fig, "Precision-Recall Curve")

def chart_feature_importance(model, feature_names: List[str], top_n: int = 20):
    """Extract feature importances from a wide variety of model types."""
    imp = None
    label = "Importance"

    # Unwrap wrappers
    inner = model
    if hasattr(model, "estimator"):       inner = model.estimator          # CalibratedClassifierCV
    if hasattr(inner, "best_estimator_"): inner = inner.best_estimator_   # GridSearch/RandomSearch

    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        imp = np.abs(inner.coef_).flatten()[:len(feature_names)]
        label = "|Coefficient|"
    elif hasattr(inner, "final_estimator_"):
        # Stacking — use final estimator if it has coef_
        fe = inner.final_estimator_
        if hasattr(fe, "coef_"):
            imp = np.abs(fe.coef_).flatten()
            # Pad/trim to match feature count
            if len(imp) != len(feature_names):
                imp = None

    if imp is None:
        return None

    imp = np.array(imp).flatten()
    names = list(feature_names[:len(imp)])
    df_imp = pd.DataFrame({"Feature": names, label: imp[:len(names)]})
    df_imp = df_imp.nlargest(top_n, label).sort_values(label)
    fig = px.bar(df_imp, x=label, y="Feature", orientation="h",
                 color=label,
                 color_continuous_scale=[[0, "#071428"], [0.5, "#2d7dd2"], [1, "#38b6ff"]])
    fig.update_coloraxes(showscale=False)
    return plotly_dark(fig, f"Feature {label} (Top {top_n})")

def chart_actual_vs_pred(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color="#2d7dd2", opacity=.5, size=5), name="Predictions"))
    mn, mx = min(float(min(y_test)), float(min(y_pred))), max(float(max(y_test)), float(max(y_pred)))
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                             line=dict(color="#38b6ff", dash="dash", width=2), name="Perfect fit"))
    return plotly_dark(fig, "Actual vs Predicted")

def chart_residuals(y_test, y_pred):
    res = np.array(y_test) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=res, mode="markers",
                             marker=dict(color="#1a56a8", opacity=.5, size=5), name="Residuals"))
    fig.add_hline(y=0, line_dash="dash", line_color="#38b6ff", line_width=1.5)
    return plotly_dark(fig, "Residual Plot")

def chart_error_dist(y_test, y_pred):
    res = np.array(y_test) - np.array(y_pred)
    fig = px.histogram(x=res, nbins=40, color_discrete_sequence=["#2d7dd2"])
    return plotly_dark(fig, "Error Distribution")

def chart_leaderboard_bar(results: List[Dict], problem: str):
    df = pd.DataFrame([{"Model": r["Model"], "Score": r.get("Score",0)}
                       for r in results if r.get("Score",-9999) > -9999])
    df = df.sort_values("Score")
    fig = px.bar(df, x="Score", y="Model", orientation="h",
                 color="Score", color_continuous_scale=[[0,"#071428"],[.5,"#2d7dd2"],[1,"#38b6ff"]])
    fig.update_coloraxes(showscale=False)
    label = "Accuracy" if problem == "Classification" else "R²"
    fig.update_xaxes(title=label)
    return plotly_dark(fig, "Model Leaderboard")

@st.cache_data(show_spinner=False)
def compute_correlation(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2: return None
    return num.corr()

def chart_correlation(df: pd.DataFrame):
    corr = compute_correlation(df)
    if corr is None: return None
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f")
    return plotly_dark(fig, "Correlation Matrix")

def chart_distribution(series: pd.Series, col_name: str):
    if series.dtype == object or series.nunique() <= 20:
        vc = series.value_counts().reset_index()
        vc.columns = [col_name, "count"]
        fig = px.bar(vc, x=col_name, y="count", color_discrete_sequence=["#2d7dd2"])
    else:
        fig = px.histogram(x=series, nbins=40, color_discrete_sequence=["#2d7dd2"])
    return plotly_dark(fig, f"Distribution — {col_name}")

# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(df, target, problem, results, best_name, source_name="") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    story = []

    def h1(t): story.append(Paragraph(t, ParagraphStyle("H1", fontSize=20, textColor=rl_colors.HexColor("#2d7dd2"), spaceAfter=8, fontName="Helvetica-Bold")))
    def h2(t): story.append(Paragraph(t, ParagraphStyle("H2", fontSize=14, textColor=rl_colors.HexColor("#38b6ff"), spaceAfter=6, fontName="Helvetica-Bold")))
    def body(t): story.append(Paragraph(t, ParagraphStyle("B", fontSize=10, textColor=rl_colors.HexColor("#8898c4"), spaceAfter=4)))
    def hr(): story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#1e2845"), spaceAfter=8))
    def sp(h=10): story.append(Spacer(1, h))
    def tbl(data, cw=None):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1e2845")),
            ("TEXTCOLOR",(0,0),(-1,0),rl_colors.HexColor("#2d7dd2")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("BACKGROUND",(0,1),(-1,-1),rl_colors.HexColor("#0a1c38")),
            ("TEXTCOLOR",(0,1),(-1,-1),rl_colors.HexColor("#8898c4")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#0a1c38"),rl_colors.HexColor("#0f1422")]),
            ("GRID",(0,0),(-1,-1),.4,rl_colors.HexColor("#2d3d6b")),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROWHEIGHT",(0,0),(-1,-1),18),
        ]))
        story.append(t); sp()

    h1("◈  SmartModel AI")
    body(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    body("Your Data. Your Model.  —  Results are based exclusively on your uploaded dataset.")
    if source_name: body(f"Dataset: {source_name}")
    hr(); sp()

    h2("Dataset Summary")
    tbl([["Rows","Columns","Target","Problem Type","Missing Values","Duplicates"],
         [str(len(df)),str(df.shape[1]),target,problem,
          str(df.isnull().sum().sum()),str(df.duplicated().sum())]],
        cw=[2.5*cm]*6)

    h2("Column Overview")
    type_data = [["Column","Type","Unique","Missing"]]
    for col in df.columns[:30]:
        type_data.append([col, str(df[col].dtype), str(df[col].nunique()), str(df[col].isnull().sum())])
    tbl(type_data, cw=[6*cm,3*cm,3*cm,3*cm])

    h2("Model Leaderboard")
    valid = [r for r in results if r.get("Score",-9999) > -9999]
    if problem == "Classification":
        hdrs = ["Model","Accuracy","F1","Precision","Recall","ROC-AUC","Time(s)"]
        rows = [[r["Model"],str(r.get("Accuracy","—")),str(r.get("F1","—")),
                 str(r.get("Precision","—")),str(r.get("Recall","—")),
                 str(r.get("ROC-AUC","—")),str(r.get("Time(s)","—"))] for r in valid]
    else:
        hdrs = ["Model","R²","RMSE","MAE","Time(s)"]
        rows = [[r["Model"],str(r.get("R²","—")),str(r.get("RMSE","—")),
                 str(r.get("MAE","—")),str(r.get("Time(s)","—"))] for r in valid]
    tbl([hdrs]+rows)

    h2("Best Model")
    best = next((r for r in valid if r["Model"]==best_name), valid[0] if valid else {})
    if best: body(f"<b>{best['Model']}</b> — Score: {best.get('Score','—')}")
    hr()
    body("Generated by SmartModel AI  •  Your Data. Your Model.")

    doc.build(story)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("Dashboard",      "📊"),
    ("Upload Dataset", "⬆️"),
    ("Train Model",    "⚡"),
    ("Predictions",    "🎯"),
    ("Reports",        "📄"),
    ("Settings",       "⚙️"),
]

with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:1.25rem 1rem 1rem">
      <div class="brand-logo">◈ SmartModel AI</div>
      <div class="brand-tag">Your Data · Your Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider" style="margin:.25rem 0 .5rem"></div>', unsafe_allow_html=True)

    # Active dataset chip
    if st.session_state.df_raw is not None:
        df_info = st.session_state.df_raw
        src = st.session_state.df_source or "dataset"
        st.markdown(f"""
        <div class="dataset-chip">
          <div class="dataset-chip-label">Active Dataset</div>
          <div class="dataset-chip-name">{src[:30]}</div>
          <div class="dataset-chip-meta">{fmt_num(df_info.shape[0])} rows × {df_info.shape[1]} cols</div>
        </div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<span class="nav-group-label">Navigation</span>', unsafe_allow_html=True)
    for label, icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        btn_label = f"{icon}  {label}"
        if is_active:
            st.markdown(f"""
            <div style="background:var(--bg-hover);border:1px solid var(--border-accent);
              border-radius:8px;padding:.5rem .85rem;margin:.1rem 0;
              font-size:.84rem;font-weight:700;color:var(--accent-1)">
              {icon}&nbsp;&nbsp;{label}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(btn_label, key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

    st.markdown('<div class="sec-divider" style="margin:.75rem 0 .5rem"></div>', unsafe_allow_html=True)

    # Best model quick info
    if st.session_state.best_model_name:
        prob = st.session_state.problem_type or ""
        results = st.session_state.results
        best_r = next((r for r in results if r["Model"]==st.session_state.best_model_name), None)
        score_val = f"{best_r.get('Score',0):.4f}" if best_r else "—"
        st.markdown(f"""
        <div class="dataset-chip">
          <div class="dataset-chip-label">Best Model</div>
          <div class="dataset-chip-name">{st.session_state.best_model_name}</div>
          <div class="dataset-chip-meta">{prob} · Score: {score_val}</div>
        </div>""", unsafe_allow_html=True)

    # Clear session
    st.markdown('<div class="sidebar-clear">', unsafe_allow_html=True)
    if st.button("↺  Clear Session", use_container_width=True, key="clear_session"):
        for k in list(_DEFAULTS.keys()):
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    history = load_history()
    total_runs = len(history)
    best_ever_score = max((r.get("score",0) for r in history), default=0)
    datasets_used = len(set(r.get("source","") for r in history))

    # Hero
    st.markdown("""
    <div style="padding:2rem 0 1rem">
      <div style="font-size:2.2rem;font-weight:900;letter-spacing:-.04em;
        background:var(--grad-primary);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;background-clip:text;line-height:1.1">
        SmartModel AI
      </div>
      <div style="font-size:1rem;color:var(--text-muted);margin-top:.5rem;font-weight:500">
        Enterprise AutoML. Your data stays yours.
      </div>
    </div>""", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1: sm_card("Training Runs", str(total_runs), "All time", "var(--accent-1)")
    with c2: sm_card("Best Score", f"{best_ever_score:.4f}" if best_ever_score else "—", "Across all runs", "var(--accent-2)")
    with c3: sm_card("Datasets Trained", str(datasets_used), "Unique uploads", "var(--accent-3)")
    with c4:
        caps = sum([HAS_XGB, HAS_LGB, HAS_CAT])
        sm_card("Boosting Models", f"+{caps}", "XGB/LGB/CAT available", "var(--accent-warn)")

    divider()

    # Workflow
    st.markdown('<div class="sec-title">How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Upload Dataset", "Drop a CSV or Excel file. Your data never leaves this server."),
        ("2", "Auto Analysis", "Instant profiling: types, missing values, distributions, correlations."),
        ("3", "Data Cleaning", "Impute, encode, scale, and remove outliers with one click."),
        ("4", "Feature Selection", "Choose which columns to use as features and define your target."),
        ("5", "Model Training", "AutoML trains up to 10 algorithms and ranks them by performance."),
        ("6", "Evaluate Results", "Leaderboard, confusion matrices, ROC curves, feature importances."),
        ("7", "Make Predictions", "Upload new data and download predictions as CSV."),
        ("8", "Export & Report", "Download your trained model (.pkl/.joblib) and a PDF report."),
    ]
    cols = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="workflow-step">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-content-title">{title}</div>
                <div class="step-content-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    divider()

    # Quick start CTA
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("⬆️  Upload Your Dataset", use_container_width=True):
            st.session_state.page = "Upload Dataset"; st.rerun()

    # Capabilities
    st.markdown('<div class="sec-title" style="margin-top:1.5rem">Available Libraries</div>', unsafe_allow_html=True)
    caps_info = [
        ("Scikit-Learn", True, "Core ML algorithms"),
        ("XGBoost", HAS_XGB, "Gradient boosting"),
        ("LightGBM", HAS_LGB, "Fast gradient boosting"),
        ("CatBoost", HAS_CAT, "Categorical boosting"),
        ("PDF Reports", HAS_PDF, "Export reports"),
    ]
    html = ""
    for name, ok, desc in caps_info:
        icon = "✓" if ok else "✗"
        color = "var(--accent-ok)" if ok else "var(--accent-err)"
        bg = "rgba(34,197,94,.08)" if ok else "rgba(239,68,68,.06)"
        html += f'<span class="step-pill" style="color:{color};background:{bg};border-color:{color}22">{icon} {name}</span>'
    st.markdown(html, unsafe_allow_html=True)

    # Recent history
    if history:
        divider()
        st.markdown('<div class="sec-title">Recent Training Runs</div>', unsafe_allow_html=True)
        recent = history[:5]
        for r in recent:
            score = r.get("score", 0)
            prob = r.get("problem","")
            kind = "clf" if prob=="Classification" else "reg"
            st.markdown(f"""
            <div class="lb-row">
              <div class="lb-name">{r.get("source","")[:30]}</div>
              <span class="badge badge-{kind}">{prob}</span>
              <div style="flex:1;font-size:.8rem;color:var(--text-muted)">{r.get("model","")}</div>
              <div class="lb-score">{score:.4f}</div>
              <div class="lb-time">{r.get("date","")[:10]}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Upload Dataset":
    section_header("Upload Dataset", "CSV or Excel files only · Your data never leaves this environment")

    # Upload widget
    uploaded = st.file_uploader(
        "Drop your dataset here or click to browse",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV (.csv) and Excel (.xlsx, .xls) formats. Max recommended: 500MB.",
    )

    if uploaded:
        try:
            # Only re-parse and reset session state if this is actually a new/changed file —
            # otherwise every rerun (e.g. clicking any button on this page) would re-read the
            # whole file from bytes and silently wipe any trained model already in session.
            is_new_file = st.session_state.df_source != uploaded.name
            if is_new_file:
                ext = Path(uploaded.name).suffix.lower()
                with st.spinner("Reading file…"):
                    df = parse_uploaded_file(uploaded.getvalue(), ext)

                st.session_state.df_raw    = df
                st.session_state.df_clean  = None
                st.session_state.df_source = uploaded.name
                st.session_state.results   = []
                st.session_state.best_model = None
                st.session_state.best_model_name = ""
                st.session_state.feature_cols = []
                st.session_state.target_col = None

                st.success(f"✓  Loaded **{uploaded.name}** successfully")
            else:
                df = st.session_state.df_raw
                st.info(f"ℹ  **{uploaded.name}** is already loaded.")

            # Stats
            stats = compute_basic_stats(df)
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: sm_card("Rows", fmt_num(stats["rows"]))
            with c2: sm_card("Columns", str(stats["cols"]))
            with c3: sm_card("Missing Values", str(stats["missing"]), color="var(--accent-warn)")
            with c4: sm_card("Duplicates", str(stats["duplicates"]), color="var(--accent-warn)")
            with c5: sm_card("Memory", f"{stats['memory_mb']:.1f} MB")

            divider()

            # Preview
            sec_title("DATASET PREVIEW")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

            divider()

            # Column types summary
            sec_title("COLUMN TYPES")
            num_c = df.select_dtypes(include=np.number).columns.tolist()
            cat_c = df.select_dtypes(include=object).columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1: sm_card("Numeric", str(len(num_c)), "columns")
            with c2: sm_card("Categorical", str(len(cat_c)), "columns")
            with c3: sm_card("Other", str(df.shape[1]-len(num_c)-len(cat_c)), "columns")

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns([2,2,4])
            with col_a:
                if st.button("→  Go to Data Analysis", use_container_width=True):
                    st.session_state.page = "Train Model"; st.rerun()
            with col_b:
                if st.button("→  Go to Train Model", use_container_width=True):
                    st.session_state.page = "Train Model"; st.rerun()

        except Exception as e:
            st.error(f"Could not read file: {e}")

    else:
        # Empty state
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-icon">⬆️</div>
          <div class="upload-title">Your dataset, your control</div>
          <div class="upload-desc">
            Drop a CSV or Excel file above.<br>
            SmartModel AI works exclusively with files you upload — no external data, no internet access.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Format guide
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="sm-card">
              <div class="card-eyebrow">Supported Formats</div>
              <div style="margin-top:.5rem">
                <div class="step-pill" style="margin:.15rem">.csv</div>
                <div class="step-pill" style="margin:.15rem">.xlsx</div>
                <div class="step-pill" style="margin:.15rem">.xls</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="sm-card">
              <div class="card-eyebrow">Requirements</div>
              <div style="color:var(--text-muted);font-size:.83rem;margin-top:.5rem;line-height:1.7">
                • First row must be column headers<br>
                • At least one numeric or target column<br>
                • Works best with 100+ rows
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL  (Analysis + Cleaning + Training combined)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Train Model":
    if st.session_state.df_raw is None:
        st.warning("No dataset loaded. Please upload a CSV or Excel file first.")
        if st.button("⬆️  Upload Dataset"):
            st.session_state.page = "Upload Dataset"; st.rerun()
        st.stop()

    df_raw = st.session_state.df_raw
    section_header("Train Model", f"Dataset: {st.session_state.df_source or 'Uploaded file'}")

    tab_analysis, tab_clean, tab_train = st.tabs(["📊 Data Analysis", "🧹 Data Cleaning", "⚡ AutoML Training"])

    # ── TAB: DATA ANALYSIS ──
    with tab_analysis:
        st.markdown("<br>", unsafe_allow_html=True)
        stats = compute_basic_stats(df_raw)
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: sm_card("Rows", fmt_num(stats["rows"]))
        with c2: sm_card("Columns", str(stats["cols"]))
        with c3: sm_card("Missing", str(stats["missing"]), color="var(--accent-warn)")
        with c4: sm_card("Duplicates", str(stats["duplicates"]), color="var(--accent-warn)")
        with c5: sm_card("Memory", f"{stats['memory_mb']:.1f} MB")

        inner1, inner2, inner3, inner4, inner5 = st.tabs(
            ["Overview","Missing Values","Correlation","Distributions","Target Analysis"])

        with inner1:
            st.dataframe(df_raw.head(30), use_container_width=True, hide_index=True)
            divider()
            info, describe = compute_overview_stats(df_raw)
            st.dataframe(info, use_container_width=True, hide_index=True)
            divider()
            st.dataframe(describe, use_container_width=True)

        with inner2:
            miss = compute_missing_summary(df_raw)
            if miss.empty:
                st.success("✓  No missing values — dataset is complete.")
            else:
                st.warning(f"{miss['Missing'].sum():,} missing values across {len(miss)} columns")
                fig = px.bar(miss, x="Missing", y="Column", orientation="h",
                             color="Missing",
                             color_continuous_scale=[[0,"#071428"],[.5,"#f59e0b"],[1,"#ef4444"]])
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(plotly_dark(fig, "Missing Values"), use_container_width=True)
                st.dataframe(miss, use_container_width=True, hide_index=True)

        with inner3:
            num = df_raw.select_dtypes(include=np.number)
            if num.shape[1] >= 2:
                fig = chart_correlation(df_raw)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation matrix.")

        with inner4:
            all_cols = df_raw.columns.tolist()
            sel = st.selectbox("Select column", all_cols, key="dist_col")
            if sel:
                fig = chart_distribution(df_raw[sel], sel)
                st.plotly_chart(fig, use_container_width=True)

            num_c = df_raw.select_dtypes(include=np.number).columns.tolist()[:12]
            if num_c:
                divider()
                sec_title("NUMERIC OVERVIEW")
                rows = (len(num_c) + 2) // 3
                for ri in range(rows):
                    chunk = num_c[ri*3:(ri+1)*3]
                    cols_ui = st.columns(len(chunk))
                    for ci, cname in enumerate(chunk):
                        with cols_ui[ci]:
                            fig2 = px.histogram(df_raw, x=cname, nbins=30,
                                                color_discrete_sequence=["#2d7dd2"], height=200)
                            plotly_dark(fig2, cname)
                            fig2.update_layout(margin=dict(l=5,r=5,t=35,b=5), showlegend=False, title_font_size=11)
                            st.plotly_chart(fig2, use_container_width=True, key=f"numeric_overview_{cname}")

        with inner5:
            target_sel = st.selectbox("Select target column", df_raw.columns.tolist(), key="target_analysis")
            if target_sel:
                prob = infer_problem(df_raw[target_sel])
                st.markdown(f'**Auto-detected:** {badge(prob, "clf" if prob=="Classification" else "reg")}',
                            unsafe_allow_html=True)
                dc1, dc2 = st.columns(2)
                with dc1:
                    fig = chart_distribution(df_raw[target_sel], target_sel)
                    st.plotly_chart(fig, use_container_width=True, key=f"target_dist_{target_sel}")
                with dc2:
                    vc = df_raw[target_sel].value_counts().reset_index()
                    vc.columns = [target_sel, "count"]
                    st.dataframe(vc.head(20), use_container_width=True, hide_index=True)

                st.session_state.target_col = target_sel
                st.session_state.problem_type = prob

    # ── TAB: DATA CLEANING ──
    with tab_clean:
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("cleaning_form"):
            c1, c2 = st.columns(2)
            with c1:
                remove_dups   = st.checkbox("Remove duplicate rows", value=True)
                encode_cats   = st.checkbox("Encode categorical columns", value=True)
                encoding_method = st.selectbox("Encoding method", ["label","onehot"],
                                               help="'label' assigns each category an integer. 'onehot' creates a binary column per category (better for non-ordinal categories, skipped for 2-class columns).")
                drop_outliers = st.checkbox("Remove outliers", value=False)
                outlier_method = st.selectbox("Outlier detection method", ["zscore","iqr"],
                                              help="'zscore' flags points beyond N standard deviations. 'iqr' flags points beyond 1.5× the interquartile range — more robust to skewed data.")
            with c2:
                fill_strategy = st.selectbox("Fill missing values strategy", ["median","mean","zero","drop"])
                scale         = st.selectbox("Feature scaling", ["standard","minmax","robust","none"])
                outlier_z     = st.slider("Z-score threshold for outliers", 2.0, 5.0, 3.0, 0.5,
                                          help="Only used when outlier method is 'zscore'.")
                feature_engineering = st.selectbox(
                    "Feature engineering", ["none","interactions","polynomial"],
                    help="'interactions' adds pairwise products of numeric columns (A×B). "
                         "'polynomial' additionally adds squared terms (A², B²). "
                         "Applied to up to the first 8 numeric columns to limit dimensionality."
                )

            apply_btn = st.form_submit_button("🧹  Apply Cleaning Pipeline", use_container_width=True)

        if apply_btn:
            with st.spinner("Running cleaning pipeline…"):
                df_clean, log = clean_dataframe(
                    df_raw, remove_dups=remove_dups, fill_strategy=fill_strategy,
                    encode_cats=encode_cats, encoding_method=encoding_method, scale=scale,
                    drop_outliers=drop_outliers, outlier_method=outlier_method, outlier_z=outlier_z,
                    feature_engineering=feature_engineering,
                )
            st.session_state.df_clean   = df_clean
            st.session_state.cleaning_log = log
            st.success(f"✓  Cleaning complete: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

        if st.session_state.df_clean is not None:
            df_c = st.session_state.df_clean
            c1,c2,c3,c4 = st.columns(4)
            with c1: sm_card("Rows After",  fmt_num(df_c.shape[0]))
            with c2: sm_card("Cols After",  str(df_c.shape[1]))
            with c3: sm_card("Missing Left",str(df_c.isnull().sum().sum()), color="var(--accent-ok)")
            with c4: sm_card("Rows Removed",str(df_raw.shape[0]-df_c.shape[0]), color="var(--accent-warn)")

            with st.expander("Cleaning Log"):
                for entry in st.session_state.cleaning_log:
                    icon = "✓" if "Removed" not in entry or "outlier" not in entry else "↓"
                    st.markdown(f"- {entry}")

            divider()
            sec_title("CLEANED DATA PREVIEW")
            st.dataframe(df_c.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("Configure options above and click 'Apply Cleaning Pipeline' to proceed.")

    # ── TAB: AUTOML TRAINING ──
    with tab_train:
        st.markdown("<br>", unsafe_allow_html=True)

        df_to_use = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

        if st.session_state.df_clean is None:
            st.info("ℹ  No cleaned dataset found. Training on raw data. Consider running the cleaning step first.")

        with st.form("train_form"):
            c1, c2 = st.columns(2)
            with c1:
                default_target_idx = 0
                if st.session_state.target_col and st.session_state.target_col in df_to_use.columns:
                    default_target_idx = df_to_use.columns.tolist().index(st.session_state.target_col)
                target_col = st.selectbox("Target column (what to predict)",
                                          df_to_use.columns.tolist(), index=default_target_idx)
                problem_type = st.selectbox("Problem type", ["Auto-detect","Classification","Regression"])

            with c2:
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                                      help="Fraction of data held out for evaluation")
                tune_mode = st.selectbox("Hyperparameter tuning",
                                         ["none","random","grid"],
                                         help="'random' is fast (15 iter), 'grid' is exhaustive but slow")

            c3, c4 = st.columns(2)
            with c3:
                cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 1,
                                     help="Stratified k-fold CV score reported for every model")
                use_stacking = st.checkbox("Include Stacking Ensemble", value=True,
                                           help="Trains a meta-learner on top of RF, ET, GB (and XGBoost if available)")
            with c4:
                st.markdown("""
                <div style="font-size:.78rem;color:var(--text-muted);margin-top:.3rem;line-height:1.8">
                  ℹ️ <b>New in this version</b><br>
                  • One-hot encoding option<br>
                  • IQR-based outlier detection<br>
                  • Feature engineering (interactions/polynomial)<br>
                  • Hist Gradient Boosting model<br>
                  • Stacking Ensemble model
                </div>""", unsafe_allow_html=True)

            feat_options = [c for c in df_to_use.columns if c != target_col]
            feature_cols = st.multiselect(
                "Feature columns (leave empty to use all)",
                feat_options, default=[],
                help="Select specific features or leave blank to use all available columns"
            )

            run_btn = st.form_submit_button("⚡  Launch AutoML", use_container_width=True)

        if run_btn:
            if target_col not in df_to_use.columns:
                st.error("Target column not found in dataset."); st.stop()

            feat_cols = feature_cols if feature_cols else [c for c in df_to_use.columns if c != target_col]

            X = df_to_use[feat_cols].copy()
            y = df_to_use[target_col].copy()

            mask = X.notna().all(axis=1) & y.notna()
            X, y = X[mask], y[mask]

            if problem_type == "Auto-detect":
                problem_type = infer_problem(y)
            st.session_state.problem_type = problem_type
            st.session_state.target_col = target_col

            le = None
            if problem_type == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                st.session_state.le = le

            X = X.select_dtypes(include=np.number)
            feat_cols = X.columns.tolist()
            st.session_state.feature_cols = feat_cols

            if X.empty:
                st.error("No numeric feature columns found. Run data cleaning with encoding first."); st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if problem_type=="Classification" else None,
            )
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            # Config summary
            badge_kind = "clf" if problem_type == "Classification" else "reg"
            st.markdown(f"""
            <div class="sm-card" style="margin-bottom:1rem">
              <div class="card-eyebrow">Training Configuration</div>
              <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.5rem;align-items:center">
                {badge(problem_type, badge_kind)}
                <span style="color:var(--text-muted);font-size:.82rem">Features: <b style="color:var(--text-primary)">{len(feat_cols)}</b></span>
                <span style="color:var(--text-muted);font-size:.82rem">Train rows: <b style="color:var(--text-primary)">{len(X_train):,}</b></span>
                <span style="color:var(--text-muted);font-size:.82rem">Test rows: <b style="color:var(--text-primary)">{len(X_test):,}</b></span>
                <span style="color:var(--text-muted);font-size:.82rem">Tuning: <b style="color:var(--text-primary)">{tune_mode}</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

            prog = st.progress(0, text="Initialising…")
            status = st.empty()

            results = run_automl(
                X_train, X_test, y_train, y_test,
                problem=problem_type, tune_mode=tune_mode,
                use_stacking=use_stacking,
                cv_folds=cv_folds,
                progress_bar=prog, status_text=status,
            )

            st.session_state.results = results
            best = next((r for r in results if r.get("_model") is not None), None)
            if best:
                st.session_state.best_model = best["_model"]
                st.session_state.best_model_name = best["Model"]

                save_history({
                    "date":    datetime.now().isoformat(),
                    "source":  st.session_state.df_source or "Unknown",
                    "target":  target_col,
                    "problem": problem_type,
                    "model":   best["Model"],
                    "score":   best.get("Score",0),
                    "features":feat_cols,
                    "n_train": len(X_train),
                    "n_test":  len(X_test),
                })

            status.empty()

            if best:
                st.success(f"✓  Training complete! Best model: **{best['Model']}** — Score: `{best.get('Score','—')}`")

                divider()
                sec_title("QUICK LEADERBOARD")

                valid = [r for r in results if r.get("Score",-9999) > -9999]
                medals = ["🥇","🥈","🥉"] + ["  "] * 20
                for i, r in enumerate(valid[:8]):
                    st.markdown(f"""
                    <div class="lb-row">
                      <div class="lb-rank">{medals[i]}</div>
                      <div class="lb-name">{r['Model']}</div>
                      <div class="lb-score">{r.get('Score','—')}</div>
                      <div class="lb-time">⏱ {r.get('Time(s)','—')}s</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("→  View Full Results", use_container_width=True):
                        st.session_state.page = "Predictions"; st.rerun()
                with col_b:
                    if st.button("→  Make Predictions", use_container_width=True):
                        st.session_state.page = "Predictions"; st.rerun()
            else:
                st.error("All models failed. Check your data for issues.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS  (Results + Predictions combined)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":
    if not st.session_state.results and st.session_state.best_model is None:
        st.warning("No trained model yet. Train a model first or load a saved one below.")

    section_header("Predictions & Results", "Evaluate your models and generate predictions on new data")

    tab_results, tab_viz, tab_predict, tab_export = st.tabs(
        ["🏆 Leaderboard", "📈 Visualizations", "🎯 Make Predictions", "💾 Export Model"])

    # ── LEADERBOARD ──
    with tab_results:
        if not st.session_state.results:
            st.info("No results yet. Run AutoML training first.")
        else:
            results  = st.session_state.results
            problem  = st.session_state.problem_type or "Classification"
            best_name = st.session_state.best_model_name

            best_r = next((r for r in results if r["Model"]==best_name), results[0] if results else {})
            badge_kind = "clf" if problem=="Classification" else "reg"

            # Best model card
            st.markdown(f"""
            <div class="sm-card sm-card-accent" style="margin-bottom:1.25rem">
              <div class="card-eyebrow">Best Model</div>
              <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.25rem">
                <div class="card-metric" style="font-size:1.5rem">{best_name}</div>
                <div>
                  {badge(problem, badge_kind)}
                  &nbsp;
                  <span class="badge badge-best">Score: {best_r.get('Score','—')}</span>
                  &nbsp;
                  <span class="badge badge-warn">⏱ {best_r.get('Time(s)','—')}s</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Imbalance warning
            if any(r.get("_imbalanced") for r in results):
                st.warning("⚠️ **Class imbalance detected** — models trained with `class_weight='balanced'`. "
                           "Balanced Accuracy and MCC are better metrics than raw Accuracy here.")

            # Chart
            fig = chart_leaderboard_bar(results, problem)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            divider()
            valid = [r for r in results if r.get("Score", -9999) > -9999]

            medals = ["🥇", "🥈", "🥉"] + [f"#{i+4}" for i in range(20)]
            for i, r in enumerate(valid):
                score_disp = r.get("Score", "—")
                cv_disp = f"{r['CV Score']} ±{r['CV Std']}" if r.get("CV Score") is not None else "—"
                if problem == "Classification":
                    extra = (
                        f'<span style="color:var(--text-muted);font-size:.78rem">'
                        f'F1:{r.get("F1","—")} · '
                        f'BalAcc:{r.get("Balanced Accuracy","—")} · '
                        f'AUC:{r.get("ROC-AUC","—")} · '
                        f'MCC:{r.get("MCC","—")} · '
                        f'CV:{cv_disp}</span>'
                    )
                else:
                    extra = (
                        f'<span style="color:var(--text-muted);font-size:.78rem">'
                        f'R²:{r.get("R²","—")} · '
                        f'RMSE:{r.get("RMSE","—")} · '
                        f'MAE:{r.get("MAE","—")} · '
                        f'MAPE:{r.get("MAPE","—")}% · '
                        f'CV:{cv_disp}</span>'
                    )
                params_disp = ""
                if r.get("_best_params"):
                    params_str = ", ".join(f"{k}={v}" for k, v in list(r["_best_params"].items())[:4])
                    params_disp = f'<div style="font-size:.72rem;color:var(--text-muted);margin-top:.15rem">🔧 {params_str}</div>'
                st.markdown(f"""
                <div class="lb-row" style="flex-direction:column;align-items:flex-start;padding:.6rem 1rem">
                  <div style="display:flex;align-items:center;gap:.75rem;width:100%">
                    <div class="lb-rank">{medals[i]}</div>
                    <div class="lb-name" style="flex:1">{r["Model"]}</div>
                    <div class="lb-score">{score_disp}</div>
                    <div class="lb-time">⏱ {r.get("Time(s)","—")}s</div>
                  </div>
                  <div style="padding-left:2.5rem;margin-top:.2rem">{extra}</div>
                  {params_disp}
                </div>""", unsafe_allow_html=True)

    # ── VISUALIZATIONS ──
    with tab_viz:
        if not st.session_state.results:
            st.info("No results yet. Run AutoML training first.")
        else:
            results = st.session_state.results
            problem = st.session_state.problem_type or "Classification"
            X_test  = st.session_state.X_test
            y_test  = st.session_state.y_test
            feat_cols = st.session_state.feature_cols

            valid_models = [r["Model"] for r in results if r.get("_model") is not None]
            if not valid_models:
                st.warning("No valid models to visualize."); st.stop()

            model_sel = st.selectbox("Select model to analyze", valid_models, key="viz_model_sel")
            sel_r = next((r for r in results if r["Model"]==model_sel), None)

            if sel_r:
                y_pred_sel = sel_r["_y_pred"]
                sel_model  = sel_r["_model"]

                if problem == "Classification":
                    v1, v2, v3, v4 = st.tabs(["Confusion Matrix","ROC Curve","P-R Curve","Feature Importance"])
                    with v1:
                        le = st.session_state.le
                        labels = le.classes_.tolist() if le else None
                        st.plotly_chart(chart_confusion(y_test, y_pred_sel, labels), use_container_width=True)
                    with v2:
                        if len(np.unique(y_test))==2 and hasattr(sel_model,"predict_proba"):
                            proba = sel_model.predict_proba(X_test)[:,1]
                            st.plotly_chart(chart_roc(y_test, proba, model_sel), use_container_width=True)
                        else:
                            st.info("ROC curve requires binary classification with probability support.")
                    with v3:
                        if len(np.unique(y_test))==2 and hasattr(sel_model,"predict_proba"):
                            proba = sel_model.predict_proba(X_test)[:,1]
                            st.plotly_chart(chart_prc(y_test, proba), use_container_width=True)
                        else:
                            st.info("P-R curve requires binary classification.")
                    with v4:
                        fig = chart_feature_importance(sel_model, feat_cols)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else:   st.info("This model doesn't expose feature importances.")
                else:
                    v1, v2, v3, v4 = st.tabs(["Actual vs Predicted","Residuals","Error Distribution","Feature Importance"])
                    with v1:
                        st.plotly_chart(chart_actual_vs_pred(y_test, y_pred_sel), use_container_width=True)
                    with v2:
                        st.plotly_chart(chart_residuals(y_test, y_pred_sel), use_container_width=True)
                    with v3:
                        st.plotly_chart(chart_error_dist(y_test, y_pred_sel), use_container_width=True)
                    with v4:
                        fig = chart_feature_importance(sel_model, feat_cols)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else:   st.info("This model doesn't expose feature importances.")

    # ── MAKE PREDICTIONS ──
    with tab_predict:
        # Load saved model option
        with st.expander("Load a saved model from disk"):
            saved = list(MODELS.glob("*.pkl")) + list(MODELS.glob("*.joblib"))
            if saved:
                sel_path = st.selectbox("Saved model files", [p.name for p in saved], key="load_model_sel")
                if st.button("Load Model", key="load_model_btn"):
                    fp = MODELS / sel_path
                    try:
                        if fp.suffix == ".pkl":
                            with open(fp,"rb") as f: st.session_state.best_model = pickle.load(f)
                        else:
                            st.session_state.best_model = joblib.load(fp)
                        st.session_state.best_model_name = fp.stem
                        st.success(f"Loaded **{fp.stem}**")
                    except Exception as e:
                        st.error(f"Could not load: {e}")
            else:
                st.info("No saved models found. Train and export a model first.")

        if st.session_state.best_model is None:
            st.info("Train a model first to make predictions.")
        else:
            st.markdown(f'**Active model:** `{st.session_state.best_model_name}`')
            divider()

            st.markdown("Upload a new dataset (same column structure as training data) to get predictions.")
            pred_upload = st.file_uploader("Upload prediction dataset", type=["csv","xlsx","xls"],
                                           key="pred_upload")

            if pred_upload:
                try:
                    ext = Path(pred_upload.name).suffix.lower()
                    df_pred = pd.read_csv(pred_upload) if ext==".csv" else pd.read_excel(pred_upload)

                    feat_cols = st.session_state.feature_cols
                    if feat_cols:
                        available = [c for c in feat_cols if c in df_pred.columns]
                        missing   = [c for c in feat_cols if c not in df_pred.columns]
                        if missing:
                            st.warning(f"Missing training columns: {missing}")
                        X_new = df_pred[available].select_dtypes(include=np.number)
                    else:
                        X_new = df_pred.select_dtypes(include=np.number)

                    if X_new.empty:
                        st.error("No numeric columns available for prediction."); st.stop()

                    preds = st.session_state.best_model.predict(X_new)

                    le = st.session_state.le
                    if le is not None:
                        try: preds = le.inverse_transform(preds)
                        except Exception: pass

                    df_out = df_pred.copy()
                    df_out["Prediction"] = preds

                    st.success(f"✓  Generated {len(preds):,} predictions")
                    st.dataframe(df_out.head(30), use_container_width=True, hide_index=True)

                    csv_bytes = df_out.to_csv(index=False).encode()
                    st.download_button("⬇️  Download predictions.csv", data=csv_bytes,
                                       file_name="predictions.csv", mime="text/csv",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}\n\n{traceback.format_exc()}")

    # ── EXPORT MODEL ──
    with tab_export:
        if st.session_state.best_model is None:
            st.info("No trained model available. Run AutoML first.")
        else:
            best_name = st.session_state.best_model_name
            best_model = st.session_state.best_model

            st.markdown(f"""
            <div class="sm-card" style="margin-bottom:1.25rem">
              <div class="card-eyebrow">Model Ready for Export</div>
              <div class="card-metric" style="font-size:1.4rem;margin-top:.25rem">{best_name}</div>
              <div class="card-sub">Choose your preferred serialization format below</div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="sm-card">
                  <div class="card-eyebrow">Pickle Format</div>
                  <div style="color:var(--text-muted);font-size:.82rem;margin:.4rem 0">
                    Standard Python serialization. Compatible with most ML frameworks.
                  </div>
                </div>""", unsafe_allow_html=True)
                pkl_buf = io.BytesIO()
                pickle.dump(best_model, pkl_buf)
                pkl_bytes = pkl_buf.getvalue()
                fname = f"{best_name.replace(' ','_')}.pkl"
                st.download_button(f"⬇️  Download {fname}", data=pkl_bytes,
                                   file_name=fname, mime="application/octet-stream",
                                   use_container_width=True)
                # Save to disk
                (MODELS / fname).write_bytes(pkl_bytes)

            with c2:
                st.markdown("""
                <div class="sm-card">
                  <div class="card-eyebrow">Joblib Format</div>
                  <div style="color:var(--text-muted);font-size:.82rem;margin:.4rem 0">
                    Optimized for scikit-learn models. Better for large numpy arrays.
                  </div>
                </div>""", unsafe_allow_html=True)
                jbl_buf = io.BytesIO()
                joblib.dump(best_model, jbl_buf)
                jbl_bytes = jbl_buf.getvalue()
                jname = f"{best_name.replace(' ','_')}.joblib"
                st.download_button(f"⬇️  Download {jname}", data=jbl_bytes,
                                   file_name=jname, mime="application/octet-stream",
                                   use_container_width=True)
                (MODELS / jname).write_bytes(jbl_bytes)

            divider()
            sec_title("SAVED MODELS")
            saved = list(MODELS.glob("*.pkl")) + list(MODELS.glob("*.joblib"))
            if saved:
                for mp in sorted(saved, reverse=True)[:10]:
                    st.markdown(f"""
                    <div class="lb-row">
                      <div class="lb-name">📦 {mp.name}</div>
                      <div class="lb-time">{mp.stat().st_size/1024:.0f} KB</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:var(--text-muted);font-size:.83rem">No saved models yet.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    section_header("Reports", "Generate and download detailed PDF reports of your model performance")

    if not HAS_PDF:
        st.warning("reportlab not installed. Run: `pip install reportlab`")
        st.stop()

    # History overview
    history = load_history()
    if history:
        c1, c2, c3 = st.columns(3)
        best_ever = max(history, key=lambda r: r.get("score",0))
        with c1: sm_card("Total Training Runs", str(len(history)))
        with c2: sm_card("Best Score Ever", f"{best_ever.get('score',0):.4f}", best_ever.get("model",""))
        with c3: sm_card("Unique Datasets", str(len(set(r.get("source","") for r in history))))

        divider()

    # PDF generation
    if st.session_state.df_raw is not None and st.session_state.results:
        df      = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        target  = st.session_state.target_col or "—"
        problem = st.session_state.problem_type or "—"
        results = st.session_state.results
        best    = st.session_state.best_model_name
        source  = st.session_state.df_source or "Unknown"

        st.markdown(f"""
        <div class="sm-card" style="margin-bottom:1rem">
          <div class="card-eyebrow">Current Session</div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:.4rem">
            <span style="color:var(--text-secondary);font-size:.83rem">Dataset: <b style="color:var(--text-primary)">{source[:40]}</b></span>
            <span style="color:var(--text-secondary);font-size:.83rem">Target: <b style="color:var(--text-primary)">{target}</b></span>
            <span style="color:var(--text-secondary);font-size:.83rem">Best: <b style="color:var(--text-primary)">{best}</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("📄  Generate PDF Report", use_container_width=False):
            with st.spinner("Building report…"):
                try:
                    pdf_bytes = generate_pdf_report(df, target, problem, results, best, source)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"SmartModelAI_Report_{ts}.pdf"
                    (REPORTS / fname).write_bytes(pdf_bytes)
                    st.download_button("⬇️  Download Report", data=pdf_bytes,
                                       file_name=fname, mime="application/pdf",
                                       use_container_width=False)
                    st.success(f"✓  Report saved")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
    else:
        st.info("Complete a training run to generate a performance report.")

    # Past reports
    existing = sorted(REPORTS.glob("*.pdf"), reverse=True)
    if existing:
        divider()
        sec_title("PREVIOUS REPORTS")
        for rp in existing[:10]:
            c1, c2 = st.columns([5,1])
            with c1:
                st.markdown(f"""
                <div class="lb-row">
                  <div class="lb-name">📄 {rp.name}</div>
                  <div class="lb-time">{rp.stat().st_size/1024:.0f} KB</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                with open(rp,"rb") as f: data = f.read()
                st.download_button("⬇️", data=data, file_name=rp.name,
                                   mime="application/pdf", key=f"dl_{rp.name}")

    # History table
    if history:
        divider()
        sec_title("TRAINING HISTORY")
        hist_df = pd.DataFrame([{
            "Date":    r.get("date","")[:19].replace("T"," "),
            "Dataset": r.get("source","")[:30],
            "Target":  r.get("target",""),
            "Problem": r.get("problem",""),
            "Model":   r.get("model",""),
            "Score":   round(r.get("score",0),4),
            "Rows":    r.get("n_train",0),
        } for r in history])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if len(history) > 1:
            fig = px.scatter(hist_df, x="Date", y="Score", color="Problem", text="Model",
                             color_discrete_sequence=["#2d7dd2","#38b6ff"])
            fig.update_traces(textposition="top center", textfont_size=9, marker_size=10)
            st.plotly_chart(plotly_dark(fig, "Score History"), use_container_width=True)

        if st.button("🗑  Clear History", key="clear_hist"):
            HISTORY_FILE.write_text("[]")
            st.success("History cleared."); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    section_header("Settings", "Customize your SmartModel AI experience")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="sm-card">
          <div class="card-eyebrow">Appearance</div>
        </div>""", unsafe_allow_html=True)

        theme = st.radio("Color theme", ["Dark","Light"],
                          index=0 if st.session_state.theme=="Dark" else 1,
                          horizontal=True)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    with c2:
        st.markdown("""
        <div class="sm-card">
          <div class="card-eyebrow">Session</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.df_raw is not None:
            st.markdown(f'<div style="color:var(--text-secondary);font-size:.83rem;margin:.5rem 0">Dataset: <b>{st.session_state.df_source}</b></div>', unsafe_allow_html=True)
        if st.session_state.best_model_name:
            st.markdown(f'<div style="color:var(--text-secondary);font-size:.83rem;margin:.5rem 0">Best model: <b>{st.session_state.best_model_name}</b></div>', unsafe_allow_html=True)

        if st.button("↺  Clear All Session Data"):
            for k in list(_DEFAULTS.keys()):
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

    divider()

    # Capabilities
    st.markdown('<div class="sec-title">Installed Libraries</div>', unsafe_allow_html=True)
    libs = [
        ("Scikit-Learn", True, "Core ML: Random Forest, SVM, Linear models, KNN, etc."),
        ("XGBoost", HAS_XGB, "Extreme gradient boosting — typically best for tabular data."),
        ("LightGBM", HAS_LGB, "Fast gradient boosting from Microsoft. Excellent on large datasets."),
        ("CatBoost", HAS_CAT, "Gradient boosting with native categorical feature support."),
        ("ReportLab", HAS_PDF, "PDF report generation. Install: pip install reportlab"),
    ]
    for name, ok, desc in libs:
        icon = "✓" if ok else "✗"
        color = "var(--accent-ok)" if ok else "var(--accent-err)"
        status = "Installed" if ok else "Not installed"
        install = "" if ok else f"<code>pip install {name.lower()}</code>"
        st.markdown(f"""
        <div class="lb-row">
          <div style="font-size:1rem;color:{color};min-width:1.5rem">{icon}</div>
          <div style="flex:1">
            <div style="font-size:.88rem;font-weight:600;color:var(--text-primary)">{name}</div>
            <div style="font-size:.77rem;color:var(--text-muted)">{desc}</div>
          </div>
          <div style="font-size:.75rem;color:{color};font-weight:700">{status}</div>
          {install}
        </div>""", unsafe_allow_html=True)

    divider()

    # Data policy
    st.markdown("""
    <div class="sm-card">
      <div class="card-eyebrow">Data Privacy Policy</div>
      <div style="color:var(--text-secondary);font-size:.83rem;line-height:1.8;margin-top:.5rem">
        ✓ &nbsp; SmartModel AI works <b>exclusively</b> with data you upload.<br>
        ✓ &nbsp; No external data is fetched, searched, or imported.<br>
        ✓ &nbsp; No Kaggle integration. No public dataset suggestions.<br>
        ✓ &nbsp; No model downloads from external repositories.<br>
        ✓ &nbsp; Your data never leaves this environment.<br>
        ✓ &nbsp; All models are trained on your data only.
      </div>
    </div>""", unsafe_allow_html=True)
