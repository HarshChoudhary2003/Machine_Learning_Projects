import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense AI — Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Master CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ══ RESET & BASE ══ */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #060b18 !important;
    color: #dce8f4 !important;   /* bright base text */
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 3rem !important; max-width: 1440px !important; }
section[data-testid="stSidebar"] { display: none; }

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1424; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 99px; }

/* ══ HERO ══ */
.hero-wrap {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #07101f 0%, #0f1e3d 45%, #0a1628 100%);
    border-radius: 28px; padding: 3.5rem 3rem 3rem;
    margin-bottom: 2.5rem;
    border: 1px solid rgba(56,189,248,0.18);
    box-shadow: 0 0 0 1px rgba(129,140,248,0.06),
                0 40px 80px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
/* Animated mesh gradient */
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 30%, rgba(56,189,248,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 80% 70%, rgba(139,92,246,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 60% 10%, rgba(20,184,166,0.07) 0%, transparent 60%);
    animation: hero-glow 10s ease-in-out infinite alternate;
}
@keyframes hero-glow {
    0%   { opacity: 0.7; transform: scale(1) translate(0,0); }
    50%  { opacity: 1;   transform: scale(1.05) translate(-1%,1%); }
    100% { opacity: 0.8; transform: scale(1.02) translate(1%,-1%); }
}
/* Floating orbs */
.hero-wrap::after {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    animation: float-orb 7s ease-in-out infinite alternate;
}
@keyframes float-orb {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(-30px, 20px) scale(1.15); }
}

.hero-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #38bdf8;
    position: relative; z-index: 1; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 28px; height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    border-radius: 2px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.6rem); font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #e2f4ff 0%, #93c5fd 30%, #818cf8 60%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    position: relative; z-index: 1;
    margin-bottom: 0.7rem;
    filter: drop-shadow(0 0 30px rgba(56,189,248,0.25));
}
.hero-sub {
    font-size: 1.05rem; line-height: 1.65;
    color: #c8ddf2;   /* bright subtitle */
    max-width: 620px; position: relative; z-index: 1; margin-bottom: 1.5rem;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; position: relative; z-index: 1; }
.pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.28);
    color: #7dd3fc;           /* ← bright sky blue, highly readable */
    padding: 0.32rem 0.95rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(.4,0,.2,1);
    cursor: default;
}
.pill:hover {
    background: rgba(56,189,248,0.18);
    border-color: rgba(56,189,248,0.5);
    color: #bae6fd;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56,189,248,0.2);
}

/* ══ STAT CARDS ══ */
.stat-card {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(15,28,55,0.9), rgba(10,18,38,0.95));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 1.6rem 1.8rem;
    backdrop-filter: blur(24px);
    transition: transform 0.35s cubic-bezier(.4,0,.2,1), box-shadow 0.35s;
    cursor: default;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    transition: left 0.6s ease;
}
.stat-card:hover::before { left: 100%; }
.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 24px 50px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.15);
}
.stat-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.stat-label {
    font-size: 0.73rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #a8c4e0;   /* clearly visible label */
    margin-bottom: 0.3rem;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem; font-weight: 800; line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-sub { font-size: 0.8rem; color: #b0c8e4; }   /* bright sub-text */

/* ══ SECTION HEADERS ══ */
.sec-head {
    display: flex; align-items: center; gap: 0.8rem;
    margin: 2.2rem 0 1.2rem;
}
.sec-head-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: #dde8f5;
    white-space: nowrap;
}
.sec-head-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.35), transparent);
}

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; gap: 0.4rem !important; padding: 0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    color: #8faacb !important;     /* ← visible inactive tab */
    transition: all 0.25s !important;
    border: 1px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(129,140,248,0.15)) !important;
    color: #7dd3fc !important;
    border-color: rgba(56,189,248,0.35) !important;
    box-shadow: 0 4px 16px rgba(56,189,248,0.12) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: #bcd4f0 !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ══ SLIDERS ══ */
.stSlider > label {
    font-size: 0.87rem !important; font-weight: 700 !important;
    color: #dce8f8 !important;    /* bright slider label */
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #8faacb !important; font-size: 0.75rem !important; }
[data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 0 0 4px rgba(56,189,248,0.3) !important;
}

/* ══ BUTTON ══ */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: #ffffff !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 0.02em !important;
    border: none !important; border-radius: 14px !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
    box-shadow: 0 6px 24px rgba(14,165,233,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
    position: relative !important; overflow: hidden !important;
}
.stButton > button::after {
    content: '' !important;
    position: absolute !important; inset: 0 !important;
    background: linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.08) 100%) !important;
    opacity: 0 !important; transition: opacity 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 36px rgba(14,165,233,0.45) !important;
}
.stButton > button:hover::after { opacity: 1 !important; }

/* ══ POLLUTANT PILLS (live preview) ══ */
.poll-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.55rem; margin: 0.8rem 0;
}
.poll-pill {
    background: rgba(13,22,45,0.8);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 0.55rem 0.5rem; text-align: center;
    transition: all 0.3s;
}
.poll-pill:hover {
    border-color: rgba(56,189,248,0.4);
    background: rgba(56,189,248,0.07);
    transform: translateY(-2px);
}
.pill-name { font-size: 0.68rem; font-weight: 700; color: #9dbad8; letter-spacing: 0.06em; text-transform: uppercase; }
.pill-val  { font-size: 1rem;   font-weight: 800; color: #e2f0ff; margin-top: 0.1rem; }

/* ══ RESULT CARD ══ */
.result-card {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(10,24,50,0.95), rgba(15,30,60,0.9));
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 24px; padding: 2rem 1.8rem; text-align: center;
    backdrop-filter: blur(30px);
    box-shadow: 0 0 80px rgba(56,189,248,0.07), inset 0 1px 0 rgba(255,255,255,0.06);
    animation: slide-up 0.5s cubic-bezier(.4,0,.2,1);
}
@keyframes slide-up {
    from { opacity:0; transform:translateY(24px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute; top: -80px; left: 50%; transform: translateX(-50%);
    width: 300px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%);
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { transform: translateX(-50%) scale(1); opacity: 0.6; }
    50%       { transform: translateX(-50%) scale(1.2); opacity: 1; }
}
.result-label-top {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #a8c4e0; margin-bottom: 0.6rem;
}
.aqi-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5.5rem; font-weight: 900; line-height: 1;
    text-shadow: 0 0 40px currentColor;
    animation: count-up 0.8s cubic-bezier(.4,0,.2,1);
}
@keyframes count-up { from { opacity:0; transform:scale(0.7); } to { opacity:1; transform:scale(1); } }
.aqi-cat { font-size: 1.2rem; font-weight: 700; margin-top: 0.4rem; }
.aqi-advice {
    margin-top: 1rem; padding: 0.85rem 1rem;
    background: rgba(255,255,255,0.06);
    border-radius: 12px; font-size: 0.9rem;
    color: #cce0f5;   /* high-contrast advice text */
    line-height: 1.65;
}
.aqi-divider {
    height: 1px; margin: 1rem 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.09), transparent);
}

/* ══ METRIC / INFO CARDS ══ */
.info-card {
    background: rgba(13,22,45,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    transition: all 0.3s; margin-bottom: 0.75rem;
}
.info-card:hover { transform: translateY(-3px); border-color: rgba(56,189,248,0.25); }
.ic-label { font-size: 0.71rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #a8c4e0; }
.ic-val   { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 800; line-height: 1.1; margin: 0.2rem 0; }
.ic-sub   { font-size: 0.8rem; color: #adc8e4; line-height: 1.4; }

/* ══ AQI REFERENCE CARD ══ */
.ref-card {
    border-radius: 16px; padding: 1.1rem 1.3rem;
    background: rgba(13,22,45,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-left-width: 4px;
    margin-bottom: 0.75rem;
    transition: transform 0.3s, box-shadow 0.3s;
}
.ref-card:hover { transform: translateX(4px); box-shadow: -4px 0 20px rgba(0,0,0,0.3); }
.ref-range { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.ref-name  { font-size: 1rem; font-weight: 700; color: #e8f2ff; margin: 0.15rem 0; }
.ref-desc  { font-size: 0.8rem; color: #b0c8e4; line-height: 1.4; }

/* ══ TECH CARD ══ */
.tech-card {
    background: rgba(13,22,45,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 1.4rem 1rem; text-align: center;
    transition: all 0.35s;
}
.tech-card:hover {
    transform: translateY(-5px);
    border-color: rgba(129,140,248,0.4);
    box-shadow: 0 16px 40px rgba(129,140,248,0.12);
}
.tech-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.tech-name { font-weight: 700; color: #e8f2ff; font-size: 0.95rem; margin-bottom: 0.2rem; }
.tech-desc { font-size: 0.78rem; color: #adc8e4; }

/* ══ GLOW DIVIDER ══ */
.glow-divider {
    height: 1px; margin: 0.5rem 0;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.25), rgba(129,140,248,0.25), transparent);
}

/* ══ SELECT / INPUT widgets ══ */
[data-baseweb="select"] * { color: #c5d8f0 !important; background: #0d1624 !important; }
[data-baseweb="input"] *  { color: #c5d8f0 !important; background: #0d1624 !important; }
.stMultiSelect [data-baseweb="tag"] { background: rgba(56,189,248,0.15) !important; }
.stMultiSelect [data-baseweb="tag"] span { color: #7dd3fc !important; }

/* ══ EXPANDER ══ */
.stExpander { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 14px !important; }
.stExpander summary { color: #c5d8f0 !important; font-weight: 600 !important; }

/* ══ PLOTLY ══ */
[data-testid="stPlotlyChart"] { border-radius: 18px; overflow: hidden; }

/* ══ DATAFRAME ══ */
.stDataFrame { border-radius: 14px; overflow: hidden; }
[data-testid="stDataFrameResizable"] th { background: rgba(13,22,45,0.9) !important; color: #7dd3fc !important; }
[data-testid="stDataFrameResizable"] td { color: #c5d8f0 !important; }

/* ══ FOOTER ══ */
.footer-bar {
    text-align: center; padding: 2rem 0 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 3rem;
    font-size: 0.85rem; color: #7b9ab8;  /* visible footer */
}
.footer-bar a { color: #38bdf8; text-decoration: none; font-weight: 600; }
.footer-bar a:hover { color: #bae6fd; }
.footer-bar strong { color: #a8c4e0; }

/* ══ WARNING / ERROR ══ */
.warn-box {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.28);
    border-left: 4px solid #ef4444; border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem; color: #fecaca;   /* bright, very readable */
    font-size: 0.9rem; line-height: 1.65; font-weight: 500;
}
.warn-box code { background: rgba(239,68,68,0.18); padding: 0.1rem 0.4rem; border-radius: 5px; color: #fda4af; }
.tip-box {
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
    border-left: 4px solid #38bdf8; border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem; color: #e0f2fe;   /* bright, very readable */
    font-size: 0.9rem; line-height: 1.65; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────
def aqi_info(aqi):
    if   aqi <= 50:  return "Good",                     "#22c55e", "😊", "Air quality is satisfactory. Enjoy outdoor activities freely!"
    elif aqi <= 100: return "Moderate",                 "#f59e0b", "😐", "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged exertion."
    elif aqi <= 150: return "Unhealthy for Sensitive",  "#f97316", "😷", "Sensitive groups may experience health effects. The general public is unlikely to be affected."
    elif aqi <= 200: return "Unhealthy",                "#ef4444", "🤢", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."
    elif aqi <= 300: return "Very Unhealthy",           "#a855f7", "☠️", "Health alert — everyone may experience more serious health effects. Avoid outdoor activities."
    else:            return "Hazardous",                "#dc2626", "💀", "Health emergency: entire population is likely to be affected. Stay indoors!"

POLLUTANTS = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]

POLL_META = {
    "PM2.5":   ("Fine Particulate Matter",    "μg/m³", 0.0, 500.0, 45.0),
    "PM10":    ("Coarse Particulate Matter",  "μg/m³", 0.0, 600.0, 80.0),
    "NO":      ("Nitric Oxide",               "μg/m³", 0.0, 400.0, 10.0),
    "NO2":     ("Nitrogen Dioxide",           "μg/m³", 0.0, 400.0, 20.0),
    "NOx":     ("Nitrogen Oxides",            "ppb",   0.0, 400.0, 25.0),
    "NH3":     ("Ammonia",                    "μg/m³", 0.0, 200.0, 15.0),
    "CO":      ("Carbon Monoxide",            "mg/m³", 0.0, 100.0, 1.0),
    "SO2":     ("Sulphur Dioxide",            "μg/m³", 0.0, 300.0, 18.0),
    "O3":      ("Ozone",                      "μg/m³", 0.0, 300.0, 40.0),
    "Benzene": ("Benzene",                    "μg/m³", 0.0, 100.0, 3.0),
    "Toluene": ("Toluene",                    "μg/m³", 0.0, 200.0, 8.0),
    "Xylene":  ("Xylene",                     "μg/m³", 0.0, 100.0, 2.0),
}

BUCKET_CLR = {
    "Good":"#22c55e","Satisfactory":"#86efac",
    "Moderate":"#f59e0b","Poor":"#f97316",
    "Very Poor":"#ef4444","Severe":"#a855f7",
}

# ─── Load ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:    return tf.keras.models.load_model("model.h5", compile=False)
    except: return None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("city_day.csv"); df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]); return df
    except: return None

@st.cache_resource(show_spinner=False)
def fit_scaler(df):
    sc = StandardScaler(); sc.fit(df[POLLUTANTS]); return sc

model  = load_model()
df     = load_data()
scaler = fit_scaler(df) if df is not None else None

# plotly common dark layout
def dark_layout(**kw):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0e4f7", family="Inter"),
        **kw
    )

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Deep Learning · Environmental AI</div>
  <div class="hero-title">🌫️ AirSense AI</div>
  <div class="hero-sub">
    Predict the <strong>Air Quality Index</strong> using a deep Artificial Neural Network
    trained on real pollution data from 26 Indian cities (2015–2020).
    Enter pollutant levels and get an instant AQI forecast with health guidance.
  </div>
  <div class="hero-pills">
    <span class="pill">🧠 Neural Network</span>
    <span class="pill">📊 Real-time Prediction</span>
    <span class="pill">🏙️ 26 Indian Cities</span>
    <span class="pill">📅 2015 – 2020</span>
    <span class="pill">⚡ TensorFlow + Keras</span>
    <span class="pill">🐍 Python</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── STAT CARDS ──────────────────────────────────────────────────────────────
if df is not None:
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        (c1,"📁","Total Records",f"{len(df):,}","Cleaned samples","#38bdf8"),
        (c2,"🏙️","Cities Covered",str(df['City'].nunique()),"Across India","#818cf8"),
        (c3,"📊","National Avg AQI",f"{df['AQI'].mean():.0f}","Mean AQI value","#f59e0b"),
        (c4,"🔴","Peak AQI",f"{df['AQI'].max():.0f}","Highest measured","#f87171"),
    ]
    for col,icon,label,val,sub,clr in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-icon">{icon}</div>
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="color:{clr}">{val}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<div class='glow-divider' style='margin:1.8rem 0'></div>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predict AQI", "📊  Data Explorer", "🧬  About & Model"])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.05, 0.95], gap="large")

    # ── INPUTS ──────────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="sec-head"><span class="sec-head-text">🎛️ Pollutant Concentrations</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="tip-box">Adjust each pollutant slider. Values auto-update the live preview and AQI prediction on the right.</div>', unsafe_allow_html=True)

        inputs = {}
        c_a, c_b = st.columns(2)
        for idx, p in enumerate(POLLUTANTS):
            _, unit, lo, hi, deft = POLL_META[p]
            col = c_a if idx < 6 else c_b
            with col:
                inputs[p] = st.slider(f"{p} ({unit})", lo, hi, deft, step=0.1, key=f"s_{p}")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict Air Quality Index", key="pred_btn")

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="sec-head"><span class="sec-head-text">⚡ Live Results</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)

        # Live pill grid
        pill_html = "".join(
            f'<div class="poll-pill"><div class="pill-name">{p}</div><div class="pill-val">{inputs[p]:.1f}</div></div>'
            for p in POLLUTANTS
        )
        st.markdown(f'<div class="poll-grid">{pill_html}</div>', unsafe_allow_html=True)
        st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

        # Prediction
        if model is not None and scaler is not None:
            fv    = np.array([[inputs[p] for p in POLLUTANTS]])
            scaled = scaler.transform(fv)
            pred  = max(0.0, float(model.predict(scaled, verbose=0)[0][0]))
            cat, clr, emo, adv = aqi_info(pred)

            st.markdown(f"""
            <div class="result-card">
              <div class="result-label-top">Predicted AQI</div>
              <div class="aqi-big" style="color:{clr}">{pred:.0f}</div>
              <div class="aqi-cat" style="color:{clr}">{emo}&nbsp; {cat}</div>
              <div class="aqi-divider"></div>
              <div class="aqi-advice">{adv}</div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"font":{"size":52,"color":clr,"family":"Space Grotesk"},"suffix":""},
                gauge={
                    "axis":{"range":[0,500],"tickwidth":1,"tickcolor":"#3d5575","tickfont":{"color":"#6b87a8","size":10}},
                    "bar":{"color":clr,"thickness":0.28},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[
                        {"range":[0,50],   "color":"rgba(34,197,94,0.12)"},
                        {"range":[50,100], "color":"rgba(245,158,11,0.12)"},
                        {"range":[100,150],"color":"rgba(249,115,22,0.12)"},
                        {"range":[150,200],"color":"rgba(239,68,68,0.12)"},
                        {"range":[200,300],"color":"rgba(168,85,247,0.12)"},
                        {"range":[300,500],"color":"rgba(220,38,38,0.12)"},
                    ],
                    "threshold":{"line":{"color":clr,"width":4},"thickness":0.78,"value":pred},
                },
                domain={"x":[0,1],"y":[0,1]},
            ))
            fig_g.update_layout(**dark_layout(height=270, margin=dict(l=20,r=20,t=30,b=10)))
            st.plotly_chart(fig_g, use_container_width=True)

            # Radar
            norms = [min(inputs[p]/POLL_META[p][3], 1.0) for p in POLLUTANTS]
            fig_r = go.Figure(go.Scatterpolar(
                r=norms+[norms[0]], theta=POLLUTANTS+[POLLUTANTS[0]],
                fill="toself",
                fillcolor="rgba(56,189,248,0.12)",
                line=dict(color="#38bdf8", width=2.2),
            ))
            fig_r.update_layout(
                **dark_layout(height=310, margin=dict(l=40,r=40,t=45,b=15),
                    title=dict(text="Pollutant Profile Radar",font=dict(color="#b0c8e8",size=13),x=0.5)),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.05)",
                                   tickfont=dict(color="#4a6080",size=8), visible=True),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#8faacb",size=10)),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        else:
            st.markdown("""<div class="warn-box">⚠️ <strong>Model not loaded.</strong>
            Ensure <code>model.h5</code> and <code>city_day.csv</code> are in the same folder as <code>app.py</code>.
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is None:
        st.markdown('<div class="warn-box">❌ <code>city_day.csv</code> not found.</div>', unsafe_allow_html=True)
    else:
        # Filters
        st.markdown('<div class="sec-head"><span class="sec-head-text">🗂️ Filters</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        f1, f2 = st.columns([1, 2])
        cities_list = sorted(df["City"].unique())
        with f1:
            sel_cities = st.multiselect("Cities", cities_list, default=cities_list[:5], key="ms_city")
        with f2:
            min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
            date_r = st.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)

        if not sel_cities:
            st.warning("Select at least one city.")
        else:
            fdf = df[df["City"].isin(sel_cities)].copy()
            if len(date_r) == 2:
                fdf = fdf[(fdf["Date"].dt.date >= date_r[0]) & (fdf["Date"].dt.date <= date_r[1])]

            # ── Timeline ──
            st.markdown('<div class="sec-head"><span class="sec-head-text">📅 AQI Timeline</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
            fig_ln = px.line(fdf, x="Date", y="AQI", color="City",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_ln.update_traces(line_width=2)
            fig_ln.update_layout(
                **dark_layout(height=340, margin=dict(l=10,r=10,t=20,b=20)),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b87a8")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b87a8")),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0c8e8")),
            )
            st.plotly_chart(fig_ln, use_container_width=True)

            ca, cb = st.columns(2)

            # ── Donut ──
            with ca:
                st.markdown('<div class="sec-head"><span class="sec-head-text">🥧 Category Split</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
                bc = fdf["AQI_Bucket"].value_counts().reset_index()
                bc.columns = ["Bucket","Count"]
                fig_d = go.Figure(go.Pie(
                    labels=bc["Bucket"], values=bc["Count"], hole=0.58,
                    marker=dict(colors=[BUCKET_CLR.get(b,"#475569") for b in bc["Bucket"]],
                                line=dict(color="#060b18", width=2)),
                    textfont=dict(color="#dde8f5", size=12),
                ))
                fig_d.update_layout(**dark_layout(height=300, margin=dict(l=0,r=0,t=15,b=0),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0c8e8", size=11))))
                st.plotly_chart(fig_d, use_container_width=True)

            # ── Bar ──
            with cb:
                st.markdown('<div class="sec-head"><span class="sec-head-text">🏆 City Avg AQI</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
                cavg = fdf.groupby("City")["AQI"].mean().sort_values().reset_index()
                fig_b = go.Figure(go.Bar(
                    y=cavg["City"], x=cavg["AQI"], orientation="h",
                    marker=dict(color=cavg["AQI"],
                                colorscale=[[0,"#22c55e"],[0.35,"#f59e0b"],[0.65,"#f97316"],[1,"#ef4444"]]),
                    text=[f"{v:.0f}" for v in cavg["AQI"]], textposition="outside",
                    textfont=dict(color="#c5d8f0"),
                ))
                fig_b.update_layout(
                    **dark_layout(height=300, margin=dict(l=0,r=40,t=10,b=10)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b87a8")),
                    yaxis=dict(tickfont=dict(color="#c5d8f0")),
                )
                st.plotly_chart(fig_b, use_container_width=True)

            # ── Heatmap ──
            st.markdown('<div class="sec-head"><span class="sec-head-text">🔥 Correlation Heatmap</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
            corr = fdf[POLLUTANTS+["AQI"]].corr()
            fig_h = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0,"#1e3a5f"],[0.25,"#0ea5e9"],[0.5,"#060b18"],[0.75,"#9333ea"],[1,"#ec4899"]],
                text=np.round(corr.values,2), texttemplate="%{text}",
                textfont=dict(size=9, color="#e2f0ff"),
                zmin=-1, zmax=1,
                colorbar=dict(tickfont=dict(color="#8faacb"), outlinecolor="rgba(0,0,0,0)"),
            ))
            fig_h.update_layout(
                **dark_layout(height=430, margin=dict(l=10,r=10,t=10,b=10)),
                xaxis=dict(tickfont=dict(color="#8faacb", size=10)),
                yaxis=dict(tickfont=dict(color="#8faacb", size=10)),
            )
            st.plotly_chart(fig_h, use_container_width=True)

            # ── Monthly trend ──
            st.markdown('<div class="sec-head"><span class="sec-head-text">📆 Monthly Avg AQI</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
            fdf2 = fdf.copy(); fdf2["Month"] = fdf2["Date"].dt.to_period("M").astype(str)
            mth = fdf2.groupby("Month")["AQI"].mean().reset_index()
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                x=mth["Month"], y=mth["AQI"],
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=5, color="#818cf8", line=dict(width=1.5, color="#38bdf8")),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
            ))
            fig_m.update_layout(
                **dark_layout(height=290, margin=dict(l=10,r=10,t=10,b=60)),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)",tickfont=dict(color="#6b87a8"),tickangle=-45),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",tickfont=dict(color="#6b87a8"),title="Avg AQI",
                            title_font=dict(color="#8faacb")),
            )
            st.plotly_chart(fig_m, use_container_width=True)

            with st.expander("🗃️ Raw Data Table"):
                st.dataframe(fdf.sort_values("Date",ascending=False).head(300),
                             use_container_width=True, height=280)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    a1, a2 = st.columns([2, 1], gap="large")

    with a1:
        st.markdown('<div class="sec-head"><span class="sec-head-text">🧬 Model Architecture</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        layers_ = ["Input  (12 Pollutant Features)", "Dense 128 + ReLU + Dropout", "Dense 64  + ReLU + Dropout", "Output  → AQI Value"]
        szs_    = [12, 128, 64, 1]
        clrs_   = ["#0ea5e9","#818cf8","#c084fc","#f59e0b"]
        fig_a   = go.Figure()
        for i,(nm,sz,cr) in enumerate(zip(layers_,szs_,clrs_)):
            fig_a.add_trace(go.Bar(
                x=[sz], y=[i], orientation="h", width=0.55,
                marker=dict(color=cr, opacity=0.85, line=dict(width=0)),
                text=[f"  {nm}  "], textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=13, family="Space Grotesk"),
            ))
        fig_a.update_layout(
            **dark_layout(height=300, margin=dict(l=10,r=10,t=10,b=10)),
            xaxis=dict(visible=False, range=[0,140]),
            yaxis=dict(visible=False),
            showlegend=False, barmode="overlay",
        )
        st.plotly_chart(fig_a, use_container_width=True)

        # AQI Reference
        st.markdown('<div class="sec-head"><span class="sec-head-text">📊 AQI Category Reference</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        aqi_ref = [
            ("0 – 50",   "Good",                    "#22c55e","😊","Safe for everyone. Great day to be outside!"),
            ("51 – 100", "Moderate",                "#f59e0b","😐","Acceptable air; very sensitive people should limit exertion."),
            ("101 – 150","Unhealthy for Sensitive", "#f97316","😷","Sensitive groups (elderly, children, asthma) at risk."),
            ("151 – 200","Unhealthy",               "#ef4444","🤢","General public may experience health effects."),
            ("201 – 300","Very Unhealthy",          "#a855f7","☠️","Everyone likely affected. Avoid outdoor activities."),
            ("300+",     "Hazardous",               "#dc2626","💀","Emergency conditions. Stay indoors immediately."),
        ]
        r1, r2 = st.columns(2)
        for idx,(rng,nm,cr,emo,desc) in enumerate(aqi_ref):
            with (r1 if idx%2==0 else r2):
                st.markdown(f"""
                <div class="ref-card" style="border-left-color:{cr}">
                  <div style="font-size:1.4rem;margin-bottom:0.3rem">{emo}</div>
                  <div class="ref-range" style="color:{cr}">{rng}</div>
                  <div class="ref-name">{nm}</div>
                  <div class="ref-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="sec-head"><span class="sec-head-text">📐 Model Details</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        for lbl,val,sub,cr in [
            ("Input Features","12","PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene","#38bdf8"),
            ("Hidden Layers","2","Dense + ReLU + Dropout (regularisation)","#818cf8"),
            ("Output","AQI","Continuous regression value (0 – 500+)","#f59e0b"),
            ("Optimiser","Adam","Adaptive learning rate optimiser","#22c55e"),
        ]:
            st.markdown(f"""
            <div class="info-card">
              <div class="ic-label">{lbl}</div>
              <div class="ic-val" style="color:{cr}">{val}</div>
              <div class="ic-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head"><span class="sec-head-text">🛠️ Tech Stack</span><div class="sec-head-line"></div></div>', unsafe_allow_html=True)
        techs = [("🧠","TensorFlow / Keras","Neural network training & inference"),
                 ("🐼","Pandas / NumPy","Data wrangling & numerical ops"),
                 ("📊","Plotly","Interactive charts & visualisations"),
                 ("🚀","Streamlit","Web application framework")]
        t1,t2 = st.columns(2)
        for i,(ico,nm,dsc) in enumerate(techs):
            with (t1 if i%2==0 else t2):
                st.markdown(f"""
                <div class="tech-card">
                  <div class="tech-icon">{ico}</div>
                  <div class="tech-name">{nm}</div>
                  <div class="tech-desc">{dsc}</div>
                </div>""", unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  🌫️ <strong>AirSense AI</strong> &nbsp;·&nbsp;
  Built with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a>
  &nbsp;·&nbsp; Powered by TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
